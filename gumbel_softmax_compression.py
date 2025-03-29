"""
Gumbel-Softmax Discretization for StyleGAN3-HVAE Compression

This script implements a Gumbel-Softmax based discretization layer between 
the HVAE encoder and StyleGAN3 generator. This approach allows for:
- Differentiable discretization during training
- More efficient training of the discrete latent space
- Better compression performance with discrete codes

The Gumbel-Softmax trick enables training discrete latent variables with
straight-through gradients, improving compression performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm

from stylegan3_hvae_full import HVAE_VGG_Encoder, StyleGAN3Compressor, save_tensor_as_image

# Gumbel-Softmax discretization layer
class GumbelSoftmaxDiscretization(nn.Module):
    """
    Implements Gumbel-Softmax discretization for latent vectors.
    
    This allows differentiable discretization during training while
    using hard quantization during inference.
    """
    def __init__(
        self, 
        latent_dim=512,
        n_embeddings=256,  # Number of discrete values (eg. 256 for 8-bit)
        temperature=1.0,
        straight_through=True,
        learnable_temp=True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        self.initial_temp = temperature
        self.straight_through = straight_through
        
        # Create codebook embeddings 
        # Linearly spaced values from -1 to 1 (StyleGAN's W space range)
        self.register_buffer(
            'codebook', 
            torch.linspace(-1, 1, n_embeddings).float()
        )
        
        # Learnable temperature parameter
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.ones(1) * np.log(temperature))
        else:
            self.register_buffer('log_temperature', torch.ones(1) * np.log(temperature))
            
        # For tracking usage statistics
        self.register_buffer('usage', torch.zeros(n_embeddings))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def update_temp(self, anneal_rate=0.00003, min_temp=0.5):
        """Anneal the temperature parameter for curriculum learning"""
        with torch.no_grad():
            self.log_temperature.clamp_(min=np.log(min_temp))
            self.log_temperature -= anneal_rate
    
    def forward(self, z, hard=None):
        """
        Forward pass with Gumbel-Softmax discretization.
        
        Args:
            z: Input continuous latent vectors [batch_size, num_ws, w_dim]
            hard: Whether to use hard discretization (defaults to self.training)
            
        Returns:
            discretized: Discretized latent vectors
            perplexity: Perplexity of the discretization (soft assignment entropy)
            encoding_indices: Indices of the closest codebook entries (for stats)
        """
        batch_size, num_ws, w_dim = z.shape
        
        # Default hard value based on training mode
        if hard is None:
            hard = not self.training
            
        # Reshape to [batch_size*num_ws*w_dim, 1]
        flat_z = z.reshape(-1, 1)
        
        # Calculate distances to codebook entries
        # This creates a [batch*num_ws*w_dim, n_embeddings] tensor of distances
        distances = torch.abs(flat_z - self.codebook.reshape(1, -1))
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        # Apply Gumbel-Softmax
        soft_onehot = F.gumbel_softmax(
            logits, 
            tau=self.temperature, 
            hard=hard,
            dim=1
        )
        
        # Compute discretized values by multiplying with codebook
        # [batch*num_ws*w_dim, n_embeddings] @ [n_embeddings, 1]
        discretized_flat = torch.matmul(soft_onehot, self.codebook.reshape(-1, 1))
        
        # Reshape back to original dimensions
        discretized = discretized_flat.reshape(batch_size, num_ws, w_dim)
        
        # Compute encoding indices for monitoring
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Update usage statistics during training
        if self.training:
            self.usage.scatter_add_(0, encoding_indices, 
                                  torch.ones_like(encoding_indices, dtype=torch.float))
        
        # Calculate perplexity (soft assignment entropy)
        avg_probs = soft_onehot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return discretized, perplexity, encoding_indices
    
    def get_code_usage(self):
        """Get normalized code usage statistics"""
        total = self.usage.sum().float()
        if total > 0:
            return self.usage / total
        else:
            return self.usage


class GumbelSoftmaxCompressor(nn.Module):
    """
    StyleGAN3 compressor with Gumbel-Softmax discretization.
    Combines the HVAE encoder with a Gumbel-Softmax discretization layer
    and the StyleGAN3 generator.
    """
    def __init__(
        self, 
        encoder, 
        generator, 
        n_embeddings=256,
        temperature=1.0,
        straight_through=True,
        training_resolution=None
    ):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.training_resolution = training_resolution
        
        # Create discretization layer per latent dimension
        self.discretization = GumbelSoftmaxDiscretization(
            latent_dim=encoder.w_dim,
            n_embeddings=n_embeddings,
            temperature=temperature,
            straight_through=straight_through
        )
        
        # Freeze generator weights
        for param in generator.parameters():
            param.requires_grad = False
    
    def forward(self, x, noise_mode='const'):
        """
        Forward pass: encode, discretize, and reconstruct.
        
        Args:
            x: Input image tensor
            noise_mode: Noise mode for StyleGAN3 synthesis
            
        Returns:
            img: Reconstructed image
            w_plus: Continuous W+ latent
            w_discrete: Discretized W+ latent
            perplexity: Perplexity of the discretization
        """
        # Encode to W+ space
        w_plus, means, _ = self.encoder(x)
        
        # Use means for more stable discretization
        w_discrete, perplexity, _ = self.discretization(means)
        
        # Generate image from discretized W+ latent
        img = self.generator.synthesis(w_discrete, noise_mode=noise_mode)
        
        # Resize output to match input if necessary
        if self.training_resolution is not None and img.shape[2] != x.shape[2]:
            img = F.interpolate(img, size=(x.shape[2], x.shape[3]), 
                              mode='bilinear', align_corners=False)
        
        return img, w_plus, w_discrete, perplexity
    
    def encode(self, x, deterministic=True):
        """Encode an image to discretized W+ space"""
        w_plus, means, _ = self.encoder(x)
        
        if deterministic:
            w_discrete, _, _ = self.discretization(means, hard=True)
        else:
            w_discrete, _, _ = self.discretization(w_plus, hard=True)
            
        return w_discrete
    
    def compress(self, x, discrete_bits=8):
        """
        Compress an image to quantized indices.
        
        Args:
            x: Input image tensor
            discrete_bits: Bit precision for encoding indices
            
        Returns:
            codes: Quantized code indices, ready for further entropy coding
        """
        with torch.no_grad():
            # Encode image
            w_plus, means, _ = self.encoder(x)
            
            # Discretize latent
            w_discrete, _, indices = self.discretization(means, hard=True)
            
            # Get indices (can be entropy-coded further)
            batch_size, num_ws, w_dim = w_plus.shape
            codes = indices.reshape(batch_size, num_ws, w_dim)
            
            return codes.cpu()
    
    def decompress(self, codes, noise_mode='const'):
        """
        Decompress indices to an image.
        
        Args:
            codes: Quantized code indices [batch, num_ws, w_dim]
            noise_mode: Noise mode for StyleGAN3 synthesis
            
        Returns:
            Reconstructed image
        """
        with torch.no_grad():
            # Get device
            device = next(self.discretization.parameters()).device
            codes = codes.to(device)
            
            # Convert indices to latent values using codebook
            batch_size, num_ws, w_dim = codes.shape
            flat_codes = codes.reshape(-1)
            
            # Look up values in the codebook
            w_discrete_flat = self.discretization.codebook[flat_codes]
            w_discrete = w_discrete_flat.reshape(batch_size, num_ws, w_dim)
            
            # Generate image
            img = self.generator.synthesis(w_discrete, noise_mode=noise_mode)
            
            return img
    
    def save_compressed(self, x, filename, discrete_bits=8):
        """
        Save compressed representation of an image.
        
        Args:
            x: Input image tensor
            filename: Output filename (.npz)
            discrete_bits: Bits for discretization
            
        Returns:
            Compression statistics
        """
        # Compress image
        codes = self.compress(x, discrete_bits=discrete_bits)
        
        # Convert to numpy
        codes_np = codes.numpy()
        
        # Calculate compression statistics
        orig_size = x.numel() * 4  # Assuming 32-bit floats
        comp_size = codes_np.size * (np.log2(self.discretization.n_embeddings) / 8)
        
        # Save compressed data
        np.savez_compressed(
            filename,
            codes=codes_np,
            n_embeddings=self.discretization.n_embeddings,
            resolution=x.shape[2:4],
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
        codes = torch.from_numpy(data['codes'])
        
        # Decompress
        img = self.decompress(codes, noise_mode=noise_mode)
        
        return img, data['compression_ratio']


def train_gumbel_discretized_hvae(
    generator_pkl,
    output_dir='./output_gumbel',
    training_resolution=256,
    batch_size=4,
    num_epochs=100,
    lr=1e-4,
    temperature=1.0,
    temp_anneal_rate=0.00003,
    min_temperature=0.5,
    n_embeddings=256,  # 8-bit (2^8)
    kl_weight=0.01,
    perceptual_weight=0.8,
    gumbel_weight=1.0,
    rec_weight=1.0,
    fp16=False,
    resume_from=None,
    device_override=None,
    **kwargs
):
    """
    Train the HVAE encoder with Gumbel-Softmax discretization.
    
    Args:
        generator_pkl: Path to StyleGAN3 generator pickle
        output_dir: Directory to save results
        training_resolution: Training resolution
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        temperature: Initial temperature for Gumbel-Softmax
        temp_anneal_rate: Temperature annealing rate
        min_temperature: Minimum temperature
        n_embeddings: Number of discrete embeddings (eg. 256 for 8-bit)
        kl_weight: Weight for KL divergence loss
        perceptual_weight: Weight for perceptual loss
        gumbel_weight: Weight for Gumbel-Softmax perplexity loss
        rec_weight: Weight for reconstruction loss
        fp16: Use mixed precision
        resume_from: Resume from checkpoint
        device_override: Override device selection
    """
    import time
    import lpips
    import os.path as osp
    
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
    
    # Create encoder - full capacity for CUDA
    encoder = HVAE_VGG_Encoder(
        img_resolution=1024,
        img_channels=G.img_channels,
        w_dim=G.w_dim,
        num_ws=G.num_ws,
        block_split=(5, 12),
        channel_base=32768,
        channel_max=512,
    ).to(device)
    
    # Create Gumbel-Softmax compressor
    compressor = GumbelSoftmaxCompressor(
        encoder, 
        G, 
        n_embeddings=n_embeddings,
        temperature=temperature,
        training_resolution=training_resolution
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + 
        list(compressor.discretization.parameters()),
        lr=lr, betas=(0.9, 0.999)
    )
    
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
        compressor.discretization.load_state_dict(checkpoint['discretization_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Create synthetic training data
    num_train_samples = 50  # Number of synthetic images for training
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
            
            # Random latents through mapping network
            w_batch = G.mapping(z_batch, None)
            img_batch = G.synthesis(w_batch, noise_mode='const')
            
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
            osp.join(output_dir, f'samples/train_sample_{i}.png')
        )
    
    # Extract W average from StyleGAN for KL divergence
    w_avg = G.mapping.w_avg.unsqueeze(0).unsqueeze(0).to(device)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    
    # Training metrics history
    history = {
        'rec_loss': [],
        'kl_loss': [],
        'perceptual_loss': [],
        'perplexity_loss': [],
        'total_loss': [],
        'perplexity': [],
        'temperature': [],
        'epoch_time': []
    }
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Set encoder to training mode
        compressor.train()
        
        # Shuffle data for this epoch
        indices = torch.randperm(num_train_samples)
        
        # Training metrics for this epoch
        epoch_rec_loss = 0
        epoch_kl_loss = 0
        epoch_perceptual_loss = 0
        epoch_perplexity_loss = 0
        epoch_total_loss = 0
        epoch_perplexity = 0
        num_batches = 0
        
        # Train in batches
        for i in range(0, num_train_samples, batch_size):
            # Get batch indices
            batch_indices = indices[i:i+batch_size]
            batch_size_actual = len(batch_indices)
            
            # Get batch data
            batch_images = train_images[batch_indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                # Encode, discretize, and reconstruct
                reconstructed_imgs, w_plus, w_discrete, perplexity = compressor(batch_images)
                
                # Reconstruction loss (L2)
                rec_loss = F.mse_loss(batch_images, reconstructed_imgs)
                
                # Perceptual loss (LPIPS)
                perceptual_loss = percep(batch_images, reconstructed_imgs).mean()
                
                # KL divergence loss
                _, means, logvars = encoder(batch_images)
                kl_loss = 0.5 * torch.mean(torch.sum(
                    torch.pow((means - w_avg), 2) + 
                    torch.exp(logvars) - logvars - 1,
                    dim=[1, 2]
                ))
                
                # Gumbel perplexity maximization loss (encourages using all codes)
                # We want high perplexity (utilization of all codes)
                ideal_perplexity = torch.tensor(n_embeddings, device=device)
                perplexity_loss = F.mse_loss(perplexity, ideal_perplexity)
                
                # Total loss
                loss = rec_weight * rec_loss + \
                       perceptual_weight * perceptual_loss + \
                       kl_weight * kl_loss + \
                       gumbel_weight * perplexity_loss
            
            # Backward pass
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
            epoch_perplexity_loss += perplexity_loss.item() * batch_size_actual
            epoch_total_loss += loss.item() * batch_size_actual
            epoch_perplexity += perplexity.item()
            num_batches += 1
        
        # Update temperature
        compressor.discretization.update_temp(
            anneal_rate=temp_anneal_rate,
            min_temp=min_temperature
        )
        current_temp = compressor.discretization.temperature.item()
        
        # Compute epoch metrics
        epoch_rec_loss /= num_train_samples
        epoch_kl_loss /= num_train_samples
        epoch_perceptual_loss /= num_train_samples
        epoch_perplexity_loss /= num_train_samples
        epoch_total_loss /= num_train_samples
        epoch_perplexity /= num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history['rec_loss'].append(epoch_rec_loss)
        history['kl_loss'].append(epoch_kl_loss)
        history['perceptual_loss'].append(epoch_perceptual_loss)
        history['perplexity_loss'].append(epoch_perplexity_loss)
        history['total_loss'].append(epoch_total_loss)
        history['perplexity'].append(epoch_perplexity)
        history['temperature'].append(current_temp)
        history['epoch_time'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {epoch_total_loss:.4f} | "
              f"Rec: {epoch_rec_loss:.4f} | "
              f"KL: {epoch_kl_loss:.4f} | "
              f"Perceptual: {epoch_perceptual_loss:.4f} | "
              f"Perplexity: {epoch_perplexity:.1f}/{n_embeddings} | "
              f"Temp: {current_temp:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save samples and checkpoint periodically
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            # Set model to eval mode
            compressor.eval()
            
            # Generate samples
            with torch.no_grad():
                # Select a few samples for visualization
                sample_indices = [0, min(1, len(train_images)-1), min(2, len(train_images)-1)]
                sample_images = train_images[sample_indices[:min(3, len(train_images))]]
                
                # Encode and reconstruct with discretization
                reconstructed_imgs, _, _, _ = compressor(sample_images)
                
                # Compress and decompress with hard quantization
                compressed_codes = compressor.compress(sample_images)
                decompressed_imgs = compressor.decompress(compressed_codes)
                
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
                    
                    # Hard quantized
                    save_tensor_as_image(
                        decompressed_imgs[i].detach().cpu(),
                        osp.join(output_dir, f'samples/epoch_{epoch+1}_sample_{i}_quantized.png')
                    )
                
                # Show code usage statistics
                code_usage = compressor.discretization.get_code_usage().cpu().numpy()
                
                print(f"Code usage: min={code_usage.min():.4f}, max={code_usage.max():.4f}, "
                      f"unused={torch.sum(compressor.discretization.usage == 0).item()}/{n_embeddings}")
                
                print(f"Saved visualization samples for epoch {epoch+1}")
            
            # Save checkpoint
            checkpoint_path = osp.join(output_dir, f'checkpoints/epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'discretization_state_dict': compressor.discretization.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_total_loss,
                'history': history,
                'config': {
                    'training_resolution': training_resolution,
                    'img_channels': G.img_channels,
                    'w_dim': G.w_dim,
                    'num_ws': G.num_ws,
                    'block_split': (5, 12),
                    'n_embeddings': n_embeddings
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = osp.join(output_dir, 'gumbel_hvae_final.pt')
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'discretization_state_dict': compressor.discretization.state_dict(),
        'config': {
            'training_resolution': training_resolution,
            'img_channels': G.img_channels,
            'w_dim': G.w_dim,
            'num_ws': G.num_ws,
            'block_split': (5, 12),
            'n_embeddings': n_embeddings,
            'history': history,
        }
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    return encoder, compressor.discretization, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StyleGAN3 HVAE Encoder with Gumbel-Softmax Discretization")
    parser.add_argument("--generator", type=str, default="models/stylegan3-t-ffhq-1024x1024.pkl",
                        help="Path to StyleGAN3 generator pickle")
    parser.add_argument("--output", type=str, default="./output_gumbel",
                        help="Output directory")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Initial Gumbel-Softmax temperature")
    parser.add_argument("--min_temperature", type=float, default=0.5,
                        help="Minimum Gumbel-Softmax temperature")
    parser.add_argument("--temp_anneal_rate", type=float, default=0.00003,
                        help="Temperature annealing rate")
    parser.add_argument("--n_embeddings", type=int, default=256,
                        help="Number of discrete embeddings (default: 256 for 8-bit)")
    parser.add_argument("--kl_weight", type=float, default=0.01,
                        help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.8,
                        help="Perceptual loss weight")
    parser.add_argument("--gumbel_weight", type=float, default=1.0,
                        help="Gumbel-Softmax perplexity loss weight")
    parser.add_argument("--rec_weight", type=float, default=1.0,
                        help="Reconstruction loss weight")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device selection")
    
    args = parser.parse_args()
    
    # Run training
    encoder, discretization, history = train_gumbel_discretized_hvae(
        generator_pkl=args.generator,
        output_dir=args.output,
        training_resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        min_temperature=args.min_temperature,
        temp_anneal_rate=args.temp_anneal_rate,
        n_embeddings=args.n_embeddings,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        gumbel_weight=args.gumbel_weight,
        rec_weight=args.rec_weight,
        fp16=args.fp16,
        resume_from=args.resume,
        device_override=args.device
    )
    
    print("Gumbel-Softmax HVAE training completed successfully!")