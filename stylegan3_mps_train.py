import os
import sys
# Add StyleGAN3 repo to path
sys.path.insert(0, os.path.abspath('stylegan3'))

# Import StyleGAN3's dependencies
from torch_utils import misc
import dnnlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm

# Simple encoder for StyleGAN3 with reduced memory requirements
class StyleGAN3Encoder(nn.Module):
    def __init__(self, img_resolution=64, img_channels=3, w_dim=512, num_ws=16, 
                 block_split=(4, 12)):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.block_split = block_split
        
        # Simple encoder with 6 convolutional layers for 64x64 images
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        # Instance normalization
        self.norm1 = nn.InstanceNorm2d(32)
        self.norm2 = nn.InstanceNorm2d(64)
        self.norm3 = nn.InstanceNorm2d(128)
        self.norm4 = nn.InstanceNorm2d(256)
        self.norm5 = nn.InstanceNorm2d(512)
        
        # Calculate W vector allocations for hierarchical encoding
        self.num_ws_coarse = block_split[0]
        self.num_ws_medium = block_split[1] - block_split[0]
        self.num_ws_fine = num_ws - block_split[1]
        
        # Project features to W+ space
        # Assuming 64x64 input, final feature map is 1x1x512
        self.coarse_w = nn.Linear(512, self.num_ws_coarse * w_dim * 2)
        self.medium_w = nn.Linear(512, self.num_ws_medium * w_dim * 2)
        self.fine_w = nn.Linear(512, self.num_ws_fine * w_dim * 2)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Encoder backbone
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)  # 32x32
        f1 = x  # Save feature map for fine details
        
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2)  # 16x16
        f2 = x  # Save feature map for medium details
        
        x = F.leaky_relu(self.norm3(self.conv3(x)), 0.2)  # 8x8
        x = F.leaky_relu(self.norm4(self.conv4(x)), 0.2)  # 4x4
        x = F.leaky_relu(self.norm5(self.conv5(x)), 0.2)  # 2x2
        x = F.leaky_relu(self.conv6(x), 0.2)              # 1x1
        
        # Global average pooling (already 1x1)
        x = x.reshape(batch_size, -1)
        
        # Project features to W+ space for coarse features
        coarse_params = self.coarse_w(x)
        coarse_params = coarse_params.reshape(batch_size, self.num_ws_coarse, self.w_dim * 2)
        coarse_mean, coarse_logvar = torch.chunk(coarse_params, 2, dim=2)
        
        # Project features to W+ space for medium features
        medium_params = self.medium_w(x)
        medium_params = medium_params.reshape(batch_size, self.num_ws_medium, self.w_dim * 2)
        medium_mean, medium_logvar = torch.chunk(medium_params, 2, dim=2)
        
        # Project features to W+ space for fine features
        fine_params = self.fine_w(x)
        fine_params = fine_params.reshape(batch_size, self.num_ws_fine, self.w_dim * 2)
        fine_mean, fine_logvar = torch.chunk(fine_params, 2, dim=2)
        
        # Concatenate all latent vectors
        means = torch.cat([coarse_mean, medium_mean, fine_mean], dim=1)
        logvars = torch.cat([coarse_logvar, medium_logvar, fine_logvar], dim=1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        w_plus = means + eps * std
        
        return w_plus, means, logvars


class StyleGAN3Compressor(nn.Module):
    def __init__(self, encoder, generator, output_resolution=None):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.output_resolution = output_resolution
        
        # Freeze generator weights
        for param in generator.parameters():
            param.requires_grad = False
    
    def forward(self, x, noise_mode='const'):
        # Encode image to W+ space
        w_plus, _, _ = self.encoder(x)
        
        # Generate reconstruction with StyleGAN3
        with torch.no_grad():
            img = self.generator.synthesis(w_plus, noise_mode=noise_mode)
            
            # Resize output if needed
            if self.output_resolution is not None and img.shape[2] != self.output_resolution:
                img = F.interpolate(img, size=(self.output_resolution, self.output_resolution), 
                                    mode='bilinear', align_corners=False)
        
        return img, w_plus


def train_stylegan3_encoder_mps():
    print("Training StyleGAN3 encoder with MPS")
    
    # Use MPS
    device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Set small resolution for efficient training
    resolution = 64
    print(f"Using {resolution}x{resolution} resolution for training")
    
    # Load StyleGAN3 generator
    generator_pkl = 'models/stylegan3-t-ffhq-1024x1024.pkl'
    print(f"Loading StyleGAN3 generator from {generator_pkl}")
    
    with open(generator_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    
    # Print generator info
    print(f"StyleGAN3 generator info:")
    print(f"  Resolution: {G.img_resolution}x{G.img_resolution}")
    print(f"  W dimensionality: {G.w_dim}")
    print(f"  Number of W vectors: {G.num_ws}")
    
    # Create encoder
    encoder = StyleGAN3Encoder(
        img_resolution=resolution,
        img_channels=3,
        w_dim=G.w_dim,
        num_ws=G.num_ws,
        block_split=(5, 12)
    ).to(device)
    print(f"Created encoder for {resolution}x{resolution} resolution")
    
    # Create compressor
    compressor = StyleGAN3Compressor(encoder, G, output_resolution=resolution).to(device)
    print("Created compressor")
    
    # Generate training data - small number of samples
    num_samples = 2
    print(f"Generating {num_samples} training samples")
    
    # Generate images with StyleGAN3
    torch.manual_seed(42)  # For reproducibility
    z_samples = torch.randn(num_samples, G.z_dim).to(device)
    
    with torch.no_grad():
        # Generate at full resolution
        train_images = G(z_samples, None)
        
        # Resize to training resolution
        train_images_resized = F.interpolate(
            train_images, 
            size=(resolution, resolution),
            mode='bilinear',
            align_corners=False
        )
    
    print(f"Generated images with shape: {train_images_resized.shape}")
    
    # Save a training sample
    save_tensor_as_image(train_images_resized[0].cpu(), "stylegan3_train_sample.png")
    
    # Create optimizer
    optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
    
    # Extract W average from StyleGAN for KL divergence
    w_avg = G.mapping.w_avg.unsqueeze(0).unsqueeze(0).to(device)
    
    # Training loop
    num_epochs = 50
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed_imgs, w_plus = compressor(train_images_resized)
        
        # Reconstruction loss (L2)
        rec_loss = F.mse_loss(train_images_resized, reconstructed_imgs)
        
        # KL divergence
        _, means, logvars = encoder(train_images_resized)
        kl_loss = 0.5 * torch.mean(torch.sum(
            torch.pow((means - w_avg), 2) + 
            torch.exp(logvars) - logvars - 1,
            dim=[1, 2]
        ))
        
        # Total loss
        loss = rec_loss + 0.01 * kl_loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Rec: {rec_loss.item():.4f} | KL: {kl_loss.item():.4f}")
            
            # Save current reconstruction
            save_tensor_as_image(reconstructed_imgs[0].detach().cpu(), f"stylegan3_recon_epoch_{epoch+1}.png")
    
    print("Training completed!")
    
    # Final evaluation
    with torch.no_grad():
        final_reconstructed, _ = compressor(train_images_resized)
        final_loss = F.mse_loss(train_images_resized, final_reconstructed)
    
    print(f"Final reconstruction loss: {final_loss.item():.4f}")
    
    # Save original and final reconstructed images
    save_tensor_as_image(train_images_resized[0].cpu(), "stylegan3_final_original.png")
    save_tensor_as_image(final_reconstructed[0].cpu(), "stylegan3_final_reconstructed.png")
    
    # Save model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': {
            'img_resolution': resolution,
            'img_channels': 3,
            'w_dim': G.w_dim,
            'num_ws': G.num_ws,
            'block_split': (5, 12),
        }
    }, 'stylegan3_encoder.pt')
    print("Saved model as stylegan3_encoder.pt")
    
    return "StyleGAN3 training completed successfully with MPS"


def save_tensor_as_image(tensor, filename):
    """Convert a tensor to a PIL Image and save it."""
    # Convert tensor to numpy array
    img = tensor.numpy()
    
    # Scale from [-1, 1] to [0, 255]
    img = ((img.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    
    # Create PIL Image and save
    Image.fromarray(img).save(filename)


if __name__ == "__main__":
    result = train_stylegan3_encoder_mps()
    print(result)