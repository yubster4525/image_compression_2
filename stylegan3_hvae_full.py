import os
import sys
import time
import argparse
import glob
from tqdm import tqdm

# Add StyleGAN3 repo to path
sys.path.insert(0, os.path.abspath('stylegan3'))

# Import StyleGAN3's dependencies
from torch_utils import misc
import dnnlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pickle
import lpips
import os.path as osp
import torchvision.transforms as transforms

# Full VGG-style Hierarchical VAE Encoder for StyleGAN3
class HVAE_VGG_Encoder(nn.Module):
    """
    Full VGG-style HVAE encoder for StyleGAN3 with hierarchical feature extraction.
    Processes images at different resolutions to capture multi-scale features.
    """
    def __init__(
        self,
        img_resolution=1024,
        img_channels=3,
        w_dim=512,
        num_ws=16,
        block_split=(5, 12),
        channel_base=32768,
        channel_max=512,
        use_fp16=False,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.block_split = block_split
        self.use_fp16 = use_fp16
        
        # Calculate number of resolution levels
        self.num_layers = int(np.log2(img_resolution))
        
        # Calculate channel counts per resolution
        channels = {}
        for res in range(self.num_layers + 1):
            channels[res] = min(channel_max, channel_base // (2 ** (self.num_layers - res)))
        
        # Input layer
        self.from_rgb = nn.Conv2d(img_channels, channels[0], kernel_size=3, padding=1)
        
        # Main encoder blocks
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            in_channels = channels[i]
            out_channels = channels[i+1] if i < self.num_layers-1 else channels[i]
            
            block = VGGBlock(in_channels, out_channels)
            self.blocks.append(block)
            
        # Calculate hierarchical layer indices for feature extraction
        # Block 2 (early features), Block 5 (middle features), and final layer (global features)
        self.hierarchy_blocks = {
            'fine': 1,      # Early block for fine details
            'medium': 4,    # Middle block for medium-scale features
            'global': self.num_layers - 1  # Final block for global structure
        }
        
        # Calculate number of W vectors for each hierarchy level
        self.num_ws_global = block_split[0]
        self.num_ws_medium = block_split[1] - block_split[0]
        self.num_ws_fine = num_ws - block_split[1]
        
        # Create feature projectors for each hierarchy level
        self.global_projector = HierarchyProjector(
            channels[self.hierarchy_blocks['global']], 
            w_dim, 
            self.num_ws_global
        )
        
        self.medium_projector = HierarchyProjector(
            channels[self.hierarchy_blocks['medium']], 
            w_dim, 
            self.num_ws_medium
        )
        
        self.fine_projector = HierarchyProjector(
            channels[self.hierarchy_blocks['fine']], 
            w_dim, 
            self.num_ws_fine
        )
    
    def forward(self, x):
        """
        Forward pass through the HVAE encoder.
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        Returns:
            w_plus: W+ latent codes of shape [batch_size, num_ws, w_dim]
            means: Mean vectors for the latent codes
            logvars: Log variance vectors for the latent codes
        """
        batch_size = x.shape[0]
        
        # Dictionary to store hierarchy features
        hierarchy_features = {}
        
        # Print input shape for debugging
        print(f"HVAE input shape: {x.shape}")
        
        # Initial convolution
        x = self.from_rgb(x)
        
        # Forward through blocks, capturing hierarchy features
        for i, block in enumerate(self.blocks):
            # Check if we've reached the end of useful resolution
            if x.shape[2] <= 1 or x.shape[3] <= 1:
                print(f"Stopping at block {i} due to small feature map: {x.shape}")
                break
                
            x = block(x)
            print(f"After block {i}, shape: {x.shape}")
            
            # Store features at hierarchy points
            if i == self.hierarchy_blocks['fine']:
                hierarchy_features['fine'] = x
                print(f"Stored fine features: {x.shape}")
            elif i == self.hierarchy_blocks['medium']:
                hierarchy_features['medium'] = x
                print(f"Stored medium features: {x.shape}")
        
        # Final features
        hierarchy_features['global'] = x
        print(f"Stored global features: {x.shape}")
        
        # Make sure all hierarchy features exist, even with small input sizes
        if 'fine' not in hierarchy_features:
            print("Warning: Creating fine features from global")
            hierarchy_features['fine'] = x
        
        if 'medium' not in hierarchy_features:
            print("Warning: Creating medium features from global")
            hierarchy_features['medium'] = x
        
        # Project features to latent space
        global_ws, global_means, global_logvars = self.global_projector(hierarchy_features['global'])
        medium_ws, medium_means, medium_logvars = self.medium_projector(hierarchy_features['medium'])
        fine_ws, fine_means, fine_logvars = self.fine_projector(hierarchy_features['fine'])
        
        # Concatenate W vectors in correct order
        w_plus = torch.cat([global_ws, medium_ws, fine_ws], dim=1)
        means = torch.cat([global_means, medium_means, fine_means], dim=1)
        logvars = torch.cat([global_logvars, medium_logvars, fine_logvars], dim=1)
        
        return w_plus, means, logvars


class VGGBlock(nn.Module):
    """VGG-style convolution block with two convolutions and a pooling layer."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Use layer normalization for small feature maps, instance norm otherwise
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2)
        
        # Only apply pooling if feature map is big enough (at least 2x2)
        if x.shape[2] > 1 and x.shape[3] > 1:
            x = self.pool(x)
        
        return x


class HierarchyProjector(nn.Module):
    """Projects features from a specific hierarchy level to W space vectors."""
    def __init__(self, in_channels, w_dim, num_ws):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.in_channels = in_channels
        
        # Global average pooling 
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Adaptive projection network that works with any input size
        # Instead of fixed size, we'll check dimensions at runtime and use a flexible approach
        self.fc1 = nn.Linear(in_channels, 256)  # Smaller intermediate size
        self.act = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(256, num_ws * w_dim * 2)  # *2 for mean and logvar
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Print input shape for debugging
        print(f"HierarchyProjector input shape: {x.shape}")
        
        # Global average pooling and flatten
        x = self.pool(x)
        x = x.view(batch_size, -1)
        
        # Print after pooling for debugging
        print(f"After pooling shape: {x.shape}")
        
        # Make sure dimensions match by checking and adapting if needed
        in_features = x.shape[1]
        if in_features != self.in_channels:
            print(f"Warning: Expected {self.in_channels} features but got {in_features}")
            # Create new linear layer on-the-fly if dimensions don't match
            device = x.device
            self.fc1 = nn.Linear(in_features, 256).to(device)
            
        # Forward through MLP
        x = self.act(self.fc1(x))
        w_params = self.fc2(x)
        
        # Reshape to [batch_size, num_ws, w_dim*2]
        w_params = w_params.view(batch_size, self.num_ws, self.w_dim * 2)
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(w_params, 2, dim=2)
        
        # Apply reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        w = mean + eps * std
        
        return w, mean, logvar


class StyleGAN3Compressor(nn.Module):
    """
    Full StyleGAN3 compressor that combines the HVAE encoder with StyleGAN3 generator.
    Supports quantized compression and reconstruction.
    """
    def __init__(self, encoder, generator, training_resolution=None):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.training_resolution = training_resolution
        
        # Freeze generator weights
        for param in generator.parameters():
            param.requires_grad = False
    
    def forward(self, x, noise_mode='const'):
        """
        Forward pass: encode image to W+ space and reconstruct with generator.
        Returns reconstructed image and W+ latent.
        """
        # Encode to W+ space
        w_plus, _, _ = self.encoder(x)
        
        # Generate image from W+ latent
        img = self.generator.synthesis(w_plus, noise_mode=noise_mode)
        
        # Resize output to match input if necessary
        if self.training_resolution is not None and img.shape[2] != x.shape[2]:
            img = F.interpolate(img, size=(x.shape[2], x.shape[3]), 
                                mode='bilinear', align_corners=False)
        
        return img, w_plus
    
    def encode(self, x, deterministic=False):
        """
        Encode an image to W+ space.
        Args:
            x: Input image tensor
            deterministic: If True, return mean vectors instead of samples
        Returns:
            W+ latent tensor
        """
        w_plus, means, _ = self.encoder(x)
        return means if deterministic else w_plus
    
    def compress(self, x, quantization_bits=8, deterministic=True):
        """
        Compress an image to quantized W+ vectors.
        Args:
            x: Input image tensor
            quantization_bits: Bit depth for quantization
            deterministic: Use deterministic encoding
        Returns:
            Quantized W+ vectors
        """
        # Encode image
        if deterministic:
            w_plus, means, _ = self.encoder(x)
            w_plus = means  # Use means for deterministic encoding
        else:
            w_plus, _, _ = self.encoder(x)
        
        # Apply quantization
        scale = (2 ** quantization_bits) - 1
        w_scaled = (w_plus + 1) * 0.5  # Scale from [-1, 1] to [0, 1]
        w_quantized = torch.round(w_scaled * scale) / scale
        w_quantized = w_quantized * 2 - 1  # Scale back to [-1, 1]
        
        return w_quantized
    
    def decompress(self, w_plus, noise_mode='const'):
        """
        Decompress W+ vectors to an image.
        Args:
            w_plus: W+ latent vectors
            noise_mode: Noise mode for StyleGAN3 synthesis
        Returns:
            Reconstructed image
        """
        return self.generator.synthesis(w_plus, noise_mode=noise_mode)
    
    def save_compressed(self, x, filename, quantization_bits=8, deterministic=True):
        """
        Save compressed representation of an image.
        Args:
            x: Input image tensor
            filename: Output filename (.npz)
            quantization_bits: Bit depth for quantization
            deterministic: Use deterministic encoding
        """
        # Compress image
        w_quantized = self.compress(x, quantization_bits, deterministic)
        
        # Convert to numpy
        w_quantized_np = w_quantized.detach().cpu().numpy()
        
        # Calculate compression statistics
        orig_size = x.numel() * 4  # Assuming 32-bit floats
        comp_size = w_quantized.numel() * (quantization_bits / 8)
        
        # Save compressed data
        np.savez_compressed(
            filename,
            w=w_quantized_np,
            resolution=x.shape[2:4],
            bits=quantization_bits,
            orig_size=orig_size,
            comp_size=comp_size,
            compression_ratio=orig_size/comp_size
        )
        
        return orig_size, comp_size, orig_size/comp_size
    
    def load_compressed(self, filename, noise_mode='const'):
        """
        Load and decompress an image from a compressed file.
        Args:
            filename: Input filename (.npz)
            noise_mode: Noise mode for StyleGAN3 synthesis
        Returns:
            Reconstructed image
        """
        # Load compressed data
        data = np.load(filename)
        w_quantized = torch.tensor(data['w']).to(next(self.generator.parameters()).device)
        
        # Decompress
        with torch.no_grad():
            img = self.decompress(w_quantized, noise_mode=noise_mode)
        
        return img, data['compression_ratio']


def train_hvae_encoder(
    generator_pkl,
    output_dir='./output',
    training_resolution=256,
    batch_size=4,
    max_resolution=1024,
    num_epochs=100,
    lr=1e-4,
    kl_weight=0.01,
    perceptual_weight=0.8,
    rec_weight=1.0,
    fp16=False,
    resume_from=None,
    device_override=None,
    num_workers=4,
    use_random_latents=True,
    noise_mode='const',
    save_every=10,
    train_samples=50,
    dataset_path=None,
    is_imagenet=False,
    val_dataset_path=None,
    **kwargs,
):
    """
    Train the HVAE encoder for StyleGAN3.
    
    Args:
        generator_pkl: Path to the StyleGAN3 generator pickle
        output_dir: Directory to save results
        training_resolution: Training resolution (lower = faster training)
        batch_size: Training batch size
        max_resolution: Maximum resolution for the encoder
        num_epochs: Number of training epochs
        lr: Learning rate
        kl_weight: Weight for KL divergence loss
        perceptual_weight: Weight for perceptual loss
        rec_weight: Weight for reconstruction loss
        fp16: Use mixed precision training
        resume_from: Resume from checkpoint
        device_override: Override device selection
        num_workers: Number of dataloader workers
        use_random_latents: Use random latents for synthetic training data
        noise_mode: Noise mode for StyleGAN3 synthesis
        save_every: Save checkpoints every N epochs
        dataset_path: Path to real image dataset (if provided, uses real images instead of StyleGAN3 samples)
        is_imagenet: Whether the dataset has ImageNet folder structure
        val_dataset_path: Path to validation dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'checkpoints'), exist_ok=True)
    
    # Setup device
    if device_override is not None:
        device = torch.device(device_override)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print(f"Using device: {device}")
    
    # Load StyleGAN3 generator
    print(f"Loading StyleGAN3 generator from {generator_pkl}")
    with open(generator_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    
    # Print generator info
    print(f"StyleGAN3 generator info:")
    print(f"  Resolution: {G.img_resolution}x{G.img_resolution}")
    print(f"  W dimensionality: {G.w_dim}")
    print(f"  Number of W vectors: {G.num_ws}")
    
    # Create encoder - full capacity for CUDA
    encoder = HVAE_VGG_Encoder(
        img_resolution=max_resolution,
        img_channels=G.img_channels,
        w_dim=G.w_dim,
        num_ws=G.num_ws,
        block_split=(5, 12),
        use_fp16=fp16,
        # Use full capacity for RTX 4080 Super
        channel_base=32768,
        channel_max=512,
    ).to(device)
    print(f"Created HVAE encoder with max resolution {max_resolution}x{max_resolution}")
    
    # Create compressor
    compressor = StyleGAN3Compressor(
        encoder, G, training_resolution=training_resolution
    ).to(device)
    print("Created StyleGAN3 compressor")
    
    # Create optimizer
    optimizer = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Setup mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if fp16 and device.type == 'cuda' else None
    
    # Setup perceptual loss
    percep = lpips.LPIPS(net='vgg').to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from is not None and osp.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Setup data - either real dataset or synthetic from StyleGAN3
    if dataset_path is not None:
        print(f"Using real image dataset from {dataset_path}")
        
        # Create real image dataset
        train_dataset = ImageDataset(
            dataset_path,
            resolution=training_resolution,
            is_imagenet=is_imagenet
        )
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        # If using real data, we need a slightly different training loop
        using_real_data = True
        num_train_samples = len(train_dataset)
        print(f"Using {num_train_samples} real images for training")
        
        # Save some samples for reference
        sample_indices = list(range(min(5, len(train_dataset))))
        for i, idx in enumerate(sample_indices):
            sample_img = train_dataset[idx].unsqueeze(0)  # Add batch dimension
            save_tensor_as_image(
                sample_img[0].cpu(),
                osp.join(output_dir, f'samples/real_train_sample_{i}.png')
            )
    else:
        print(f"Using synthetic data from StyleGAN3")
        # Generate synthetic training data
        num_train_samples = train_samples  # Number of synthetic images for training
        print(f"Generating {num_train_samples} training samples...")
        
        # Generate random latents
        torch.manual_seed(42)  # For reproducibility
        z_train = torch.randn(num_train_samples, G.z_dim).to(device)
        
        # Generate images with StyleGAN3
        train_images = []
        train_w = []
        
        with torch.no_grad():
            for i in range(0, num_train_samples, batch_size):
                z_batch = z_train[i:i+batch_size]
                
                # Generate images
                if use_random_latents:
                    # Random latents through mapping network
                    w_batch = G.mapping(z_batch, None)
                    img_batch = G.synthesis(w_batch, noise_mode=noise_mode)
                else:
                    # Direct generation
                    img_batch = G(z_batch, None, noise_mode=noise_mode)
                    w_batch = G.mapping(z_batch, None)  # Still get W for reference
                
                # Resize to training resolution if needed
                if img_batch.shape[2] != training_resolution:
                    img_batch = F.interpolate(
                        img_batch, 
                        size=(training_resolution, training_resolution),
                        mode='bilinear',
                        align_corners=False
                    )
                
                train_images.append(img_batch.detach())
                train_w.append(w_batch.detach())
        
        # Concatenate batches
        train_images = torch.cat(train_images, dim=0)
        train_w = torch.cat(train_w, dim=0)
        
        print(f"Training dataset: {train_images.shape}")
        
        # Save some training samples
        for i in range(min(5, num_train_samples)):
            save_tensor_as_image(
                train_images[i].cpu(),
                osp.join(output_dir, f'samples/synthetic_train_sample_{i}.png')
            )
        
        # Create synthetic dataset and loader
        train_dataset = SyntheticDataset(train_images, train_w)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Synthetic data is already in memory, no need for workers
            pin_memory=False
        )
        
        using_real_data = False
    
    # Create validation dataset if specified
    val_loader = None
    if val_dataset_path is not None:
        print(f"Using validation dataset from {val_dataset_path}")
        
        # Create validation dataset
        val_dataset = ImageDataset(
            val_dataset_path,
            resolution=training_resolution,
            is_imagenet=is_imagenet
        )
        
        # Create validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"Using {len(val_dataset)} images for validation")
    
    # Extract W average from StyleGAN for KL divergence
    w_avg = G.mapping.w_avg.unsqueeze(0).unsqueeze(0).to(device)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    
    # Training metrics history
    history = {
        'rec_loss': [],
        'kl_loss': [],
        'perceptual_loss': [],
        'total_loss': [],
        'epoch_time': []
    }
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Set encoder to training mode
        encoder.train()
        
        # Training metrics for this epoch
        epoch_rec_loss = 0
        epoch_kl_loss = 0
        epoch_perceptual_loss = 0
        epoch_total_loss = 0
        num_batches = 0
        total_samples = 0
        
        # Train using data loader
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Get batch of images (for real and synthetic data)
            if using_real_data:
                batch_images = batch_data.to(device)
            else:
                # For synthetic data, the loader provides both images and w vectors
                batch_images = batch_data[0].to(device) if isinstance(batch_data, tuple) else batch_data.to(device)
            
            batch_size_actual = batch_images.shape[0]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                # Encode and reconstruct
                reconstructed_imgs, w_plus = compressor(batch_images)
                
                # Reconstruction loss (L2)
                rec_loss = F.mse_loss(batch_images, reconstructed_imgs)
                
                # Perceptual loss (LPIPS)
                perceptual_loss = percep(batch_images, reconstructed_imgs).mean()
                
                # KL divergence
                _, means, logvars = encoder(batch_images)
                kl_loss = 0.5 * torch.mean(torch.sum(
                    torch.pow((means - w_avg), 2) + 
                    torch.exp(logvars) - logvars - 1,
                    dim=[1, 2]
                ))
                
                # Total loss - full version with all components
                loss = rec_weight * rec_loss + \
                       perceptual_weight * perceptual_loss + \
                       kl_weight * kl_loss
            
            # Backward pass with mixed precision if enabled
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update metrics
            epoch_rec_loss += rec_loss.item() * batch_size_actual
            epoch_kl_loss += kl_loss.item() * batch_size_actual
            epoch_perceptual_loss += perceptual_loss.item() * batch_size_actual
            epoch_total_loss += loss.item() * batch_size_actual
            num_batches += 1
            total_samples += batch_size_actual
            
            # Print progress occasionally for large datasets
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Rec: {rec_loss.item():.4f} | "
                      f"KL: {kl_loss.item():.4f} | "
                      f"Perceptual: {perceptual_loss.item():.4f}")
        
        # Compute epoch metrics
        epoch_rec_loss /= total_samples
        epoch_kl_loss /= total_samples
        epoch_perceptual_loss /= total_samples
        epoch_total_loss /= total_samples
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history['rec_loss'].append(epoch_rec_loss)
        history['kl_loss'].append(epoch_kl_loss)
        history['perceptual_loss'].append(epoch_perceptual_loss)
        history['total_loss'].append(epoch_total_loss)
        history['epoch_time'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {epoch_total_loss:.4f} | "
              f"Rec: {epoch_rec_loss:.4f} | "
              f"KL: {epoch_kl_loss:.4f} | "
              f"Perceptual: {epoch_perceptual_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Run validation if validation dataset is available
        if val_loader is not None:
            encoder.eval()
            
            val_rec_loss = 0
            val_kl_loss = 0
            val_perceptual_loss = 0
            val_total_loss = 0
            val_total_samples = 0
            
            print("Running validation...")
            with torch.no_grad():
                for batch_data in tqdm(val_loader, desc="Validation"):
                    val_batch_images = batch_data.to(device)
                    val_batch_size = val_batch_images.shape[0]
                    
                    # Encode and reconstruct
                    reconstructed_imgs, w_plus = compressor(val_batch_images)
                    
                    # Reconstruction loss (L2)
                    rec_loss = F.mse_loss(val_batch_images, reconstructed_imgs)
                    
                    # Perceptual loss (LPIPS)
                    perceptual_loss = percep(val_batch_images, reconstructed_imgs).mean()
                    
                    # KL divergence
                    _, means, logvars = encoder(val_batch_images)
                    kl_loss = 0.5 * torch.mean(torch.sum(
                        torch.pow((means - w_avg), 2) + 
                        torch.exp(logvars) - logvars - 1,
                        dim=[1, 2]
                    ))
                    
                    # Total loss
                    loss = rec_weight * rec_loss + \
                           perceptual_weight * perceptual_loss + \
                           kl_weight * kl_loss
                    
                    # Update metrics
                    val_rec_loss += rec_loss.item() * val_batch_size
                    val_kl_loss += kl_loss.item() * val_batch_size
                    val_perceptual_loss += perceptual_loss.item() * val_batch_size
                    val_total_loss += loss.item() * val_batch_size
                    val_total_samples += val_batch_size
            
            # Compute validation metrics
            val_rec_loss /= val_total_samples
            val_kl_loss /= val_total_samples
            val_perceptual_loss /= val_total_samples
            val_total_loss /= val_total_samples
            
            # Print validation results
            print(f"Validation | "
                  f"Loss: {val_total_loss:.4f} | "
                  f"Rec: {val_rec_loss:.4f} | "
                  f"KL: {val_kl_loss:.4f} | "
                  f"Perceptual: {val_perceptual_loss:.4f}")
            
            # Save validation metrics
            if 'val_loss' not in history:
                history['val_loss'] = []
                history['val_rec_loss'] = []
                history['val_kl_loss'] = []
                history['val_perceptual_loss'] = []
            
            history['val_loss'].append(val_total_loss)
            history['val_rec_loss'].append(val_rec_loss)
            history['val_kl_loss'].append(val_kl_loss)
            history['val_perceptual_loss'].append(val_perceptual_loss)
        
        # Save samples and checkpoint periodically - full quality for CUDA
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            # Set encoder to eval mode
            encoder.eval()
            
            # Generate samples with full quality
            with torch.no_grad():
                # Get sample images for visualization (either from dataset or validation set)
                if using_real_data:
                    # Sample from training dataset
                    sample_indices = list(range(min(3, len(train_dataset))))
                    sample_images = torch.stack([train_dataset[idx] for idx in sample_indices]).to(device)
                else:
                    # Use synthetic samples
                    sample_indices = [0, min(1, len(train_images)-1), min(2, len(train_images)-1)]
                    sample_images = train_images[sample_indices[:min(3, len(train_images))]].to(device)
                
                # Encode and reconstruct
                reconstructed_imgs, w_plus = compressor(sample_images)
                
                # Compress with quantization (8 bits)
                quantized_w = compressor.compress(sample_images, quantization_bits=8)
                quantized_imgs = compressor.decompress(quantized_w)
                
                # Save samples
                for i in range(len(sample_images)):
                    # Original
                    save_tensor_as_image(
                        sample_images[i].detach().cpu(),
                        osp.join(output_dir, f'samples/epoch_{epoch+1}_sample_{i}_original.png')
                    )
                    
                    # Reconstructed
                    save_tensor_as_image(
                        reconstructed_imgs[i].detach().cpu(),
                        osp.join(output_dir, f'samples/epoch_{epoch+1}_sample_{i}_reconstructed.png')
                    )
                    
                    # Quantized (8 bits)
                    save_tensor_as_image(
                        quantized_imgs[i].detach().cpu(),
                        osp.join(output_dir, f'samples/epoch_{epoch+1}_sample_{i}_quantized_8bit.png')
                    )
                
                # If we have a validation set, also save some validation samples
                if val_loader is not None:
                    # Get some validation samples
                    val_batch = next(iter(val_loader))
                    val_samples = val_batch[:min(3, len(val_batch))].to(device)
                    
                    # Encode and reconstruct
                    val_reconstructed, val_w_plus = compressor(val_samples)
                    
                    # Compress with quantization (8 bits)
                    val_quantized_w = compressor.compress(val_samples, quantization_bits=8)
                    val_quantized_imgs = compressor.decompress(val_quantized_w)
                    
                    # Save validation samples
                    for i in range(len(val_samples)):
                        # Original
                        save_tensor_as_image(
                            val_samples[i].detach().cpu(),
                            osp.join(output_dir, f'samples/epoch_{epoch+1}_val_{i}_original.png')
                        )
                        
                        # Reconstructed
                        save_tensor_as_image(
                            val_reconstructed[i].detach().cpu(),
                            osp.join(output_dir, f'samples/epoch_{epoch+1}_val_{i}_reconstructed.png')
                        )
                        
                        # Quantized (8 bits)
                        save_tensor_as_image(
                            val_quantized_imgs[i].detach().cpu(),
                            osp.join(output_dir, f'samples/epoch_{epoch+1}_val_{i}_quantized_8bit.png')
                        )
                
                print(f"Saved visualization samples for epoch {epoch+1}")
            
            # Save checkpoint
            checkpoint_path = osp.join(output_dir, f'checkpoints/epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_total_loss,
                'history': history,
                'config': {
                    'max_resolution': max_resolution,
                    'img_channels': G.img_channels,
                    'w_dim': G.w_dim,
                    'num_ws': G.num_ws,
                    'block_split': (5, 12),
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = osp.join(output_dir, 'hvae_encoder_final.pt')
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': {
            'max_resolution': max_resolution,
            'img_channels': G.img_channels,
            'w_dim': G.w_dim,
            'num_ws': G.num_ws,
            'block_split': (5, 12),
            'history': history,
        }
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    return encoder, history


def save_tensor_as_image(tensor, filename):
    """Convert a tensor to a PIL Image and save it."""
    # Convert tensor to numpy array
    img = tensor.numpy()
    
    # Scale from [-1, 1] to [0, 255]
    img = ((img.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    
    # Create PIL Image and save
    Image.fromarray(img).save(filename)


class ImageDataset(Dataset):
    """Dataset class for training with real images including ImageNet."""
    def __init__(self, image_folder, resolution=256, is_imagenet=False, recursive=True, 
                 file_extensions=('.png', '.jpg', '.jpeg')):
        """
        Args:
            image_folder: Path to folder containing images
            resolution: Resolution to resize images to
            is_imagenet: If True, treats as ImageNet-style dataset with class folders
            recursive: Whether to recursively search for images in subdirectories
            file_extensions: Tuple of valid file extensions to include
        """
        self.image_folder = image_folder
        self.resolution = resolution
        self.is_imagenet = is_imagenet
        self.file_extensions = file_extensions
        
        # Find all images
        self.image_paths = []
        
        if is_imagenet or recursive:
            # For ImageNet-style datasets with class folders or recursive search
            for root, dirs, files in os.walk(image_folder):
                for file in files:
                    if file.lower().endswith(self.file_extensions):
                        self.image_paths.append(os.path.join(root, file))
        else:
            # For flat directory structure
            for ext in self.file_extensions:
                self.image_paths.extend(glob.glob(os.path.join(image_folder, f'*{ext}')))
                self.image_paths.extend(glob.glob(os.path.join(image_folder, f'*{ext.upper()}')))
        
        # Setup transformations
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
        
        print(f"Found {len(self.image_paths)} images in {image_folder}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and preprocess an image."""
        try:
            # Load image file
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            img_tensor = self.transform(img)
            
            return img_tensor
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return an empty tensor on error and get next one
            if idx + 1 < len(self.image_paths):
                return self.__getitem__(idx + 1)
            else:
                # Return an empty tensor at correct shape as last resort
                return torch.zeros(3, self.resolution, self.resolution)


class SyntheticDataset(Dataset):
    """Dataset of images generated by StyleGAN3."""
    def __init__(self, images, w_vectors=None):
        """
        Args:
            images: Tensor of images [N, C, H, W]
            w_vectors: Optional tensor of W vectors [N, num_ws, w_dim]
        """
        self.images = images
        self.w_vectors = w_vectors
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Return an image and optionally its W vector."""
        if self.w_vectors is not None:
            return self.images[idx], self.w_vectors[idx]
        else:
            return self.images[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StyleGAN3 HVAE Encoder")
    parser.add_argument("--generator", type=str, default="models/stylegan3-t-ffhq-1024x1024.pkl",
                        help="Path to StyleGAN3 generator pickle")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.01,
                        help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.8,
                        help="Perceptual loss weight")
    parser.add_argument("--rec_weight", type=float, default=1.0,
                        help="Reconstruction loss weight")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device selection")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoints every N epochs")
    parser.add_argument("--train_samples", type=int, default=50,
                        help="Number of training samples to generate from StyleGAN3")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to training dataset (ImageNet, FFHQ, etc.). If provided, uses real images instead of StyleGAN3 samples")
    parser.add_argument("--imagenet", action="store_true",
                        help="Treat dataset as ImageNet with class subdirectories")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to validation dataset")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Run training
    encoder, history = train_hvae_encoder(
        generator_pkl=args.generator,
        output_dir=args.output,
        training_resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        rec_weight=args.rec_weight,
        fp16=args.fp16,
        resume_from=args.resume,
        device_override=args.device,
        save_every=args.save_every,
        train_samples=args.train_samples,
        dataset_path=args.dataset,
        is_imagenet=args.imagenet,
        val_dataset_path=args.val_dataset,
        num_workers=args.workers,
    )
    
    print("Training completed successfully!")