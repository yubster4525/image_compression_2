import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import gc

# Add StyleGAN3 repo to path
sys.path.insert(0, os.path.abspath('stylegan3'))

# Import StyleGAN3's dependencies
try:
    from torch_utils import misc
    import dnnlib
except ImportError:
    print("Warning: StyleGAN3 dependencies not found. Make sure StyleGAN3 repository is available.")

class SimpleEncoder(nn.Module):
    """Simple encoder for StyleGAN3 that works reliably with MPS."""
    def __init__(self, img_resolution=64, img_channels=3, w_dim=512, num_ws=16):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        
        # Simple CNN encoder
        self.encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, 32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.GroupNorm(16, 64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2),
            nn.GroupNorm(16, 128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2),
            nn.GroupNorm(32, 256),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 2x2
            nn.LeakyReLU(0.2),
            nn.GroupNorm(32, 512),
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # 1x1
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
        )
        
        # Calculate latent dimension
        latent_dim = num_ws * w_dim
        
        # MLP to predict W
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, latent_dim),
        )
        
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # Check if input resolution matches expected resolution
        if x.shape[2] != self.img_resolution or x.shape[3] != self.img_resolution:
            x = F.interpolate(x, size=(self.img_resolution, self.img_resolution), 
                             mode='bilinear', align_corners=False)
        
        # Encode
        features = self.encoder(x)
        
        # Generate W
        w = self.fc(features)
        
        # Reshape to [batch_size, num_ws, w_dim]
        w = w.view(batch_size, self.num_ws, self.w_dim)
        
        return w


class SimpleCompressor(nn.Module):
    """Simple compressor that combines encoder and StyleGAN3 generator."""
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
        w_plus = self.encoder(x)
        
        # Generate with StyleGAN3
        with torch.no_grad():
            img = self.generator.synthesis(w_plus, noise_mode=noise_mode)
            
            # Resize if needed
            if (self.output_resolution is not None and 
                img.shape[2] != self.output_resolution):
                img = F.interpolate(img, size=(self.output_resolution, self.output_resolution), 
                                   mode='bilinear', align_corners=False)
        
        return img, w_plus


def train_simple_encoder(
    generator_pkl,
    output_dir='./output',
    resolution=64,
    batch_size=1,
    epochs=10,
    train_samples=5,
    lr=1e-4
):
    """Train a simple StyleGAN3 encoder with minimal memory usage."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Setup device - use CPU for reliability in testing
    device = torch.device('mps')
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
    
    # Create encoder
    encoder = SimpleEncoder(
        img_resolution=resolution,
        img_channels=G.img_channels,
        w_dim=G.w_dim,
        num_ws=G.num_ws
    ).to(device)
    print(f"Created simple encoder for {resolution}x{resolution} images")
    
    # Create compressor
    compressor = SimpleCompressor(
        encoder, G, output_resolution=resolution
    ).to(device)
    print("Created compressor")
    
    # Create optimizer
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    
    # Generate training samples
    print(f"Generating {train_samples} training samples...")
    
    # Generate random latents
    torch.manual_seed(42)  # For reproducibility
    z_train = torch.randn(train_samples, G.z_dim).to(device)
    
    # Generate images
    train_images = []
    train_ws = []
    
    with torch.no_grad():
        for i in range(0, train_samples, batch_size):
            # Get batch
            batch_size_actual = min(batch_size, train_samples - i)
            z_batch = z_train[i:i+batch_size_actual]
            
            # Generate W vectors
            w_batch = G.mapping(z_batch, None)
            
            # Generate images
            img_batch = G.synthesis(w_batch)
            
            # Resize to training resolution
            if img_batch.shape[2] != resolution:
                img_batch = F.interpolate(
                    img_batch, 
                    size=(resolution, resolution),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Add to training set
            train_images.append(img_batch.detach())
            train_ws.append(w_batch.detach())
            
            # Save memory
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            gc.collect()
    
    # Concatenate batches
    train_images = torch.cat(train_images, dim=0)
    train_ws = torch.cat(train_ws, dim=0)
    
    print(f"Training dataset: {train_images.shape}")
    
    # Save some training samples
    for i in range(min(2, train_samples)):
        save_tensor_as_image(
            train_images[i].cpu(),
            os.path.join(output_dir, f'samples/train_sample_{i}.png')
        )
    
    # Training loop
    print(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Shuffle data
        indices = torch.randperm(train_samples)
        
        # Train in batches
        total_loss = 0
        
        for i in range(0, train_samples, batch_size):
            # Get batch
            batch_indices = indices[i:i+batch_size]
            batch_size_actual = len(batch_indices)
            
            # Get batch data
            batch_images = train_images[batch_indices]
            batch_ws = train_ws[batch_indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, pred_ws = compressor(batch_images)
            
            # Compute loss - just L2 on W vectors for simplicity
            loss = F.mse_loss(pred_ws, batch_ws)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item() * batch_size_actual
        
        # Compute average loss
        avg_loss = total_loss / train_samples
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        # Generate samples every few epochs
        if epoch % 2 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                # Get first training image
                sample_img = train_images[0:1]
                
                # Encode and reconstruct
                recon_img, _ = compressor(sample_img)
                
                # Save images
                save_tensor_as_image(
                    sample_img[0].cpu(),
                    os.path.join(output_dir, f'samples/epoch_{epoch+1}_original.png')
                )
                save_tensor_as_image(
                    recon_img[0].cpu(),
                    os.path.join(output_dir, f'samples/epoch_{epoch+1}_reconstructed.png')
                )
    
    print("Training completed!")
    
    # Save model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': {
            'img_resolution': resolution,
            'img_channels': G.img_channels,
            'w_dim': G.w_dim,
            'num_ws': G.num_ws,
        }
    }, os.path.join(output_dir, 'simple_encoder.pt'))
    
    print(f"Model saved to {os.path.join(output_dir, 'simple_encoder.pt')}")
    
    return encoder


def save_tensor_as_image(tensor, filename):
    """Convert a tensor to a PIL Image and save it."""
    # Convert tensor to numpy array
    img = tensor.numpy()
    
    # Scale from [-1, 1] to [0, 255]
    img = ((img.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    
    # Create PIL Image and save
    Image.fromarray(img).save(filename)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a simple StyleGAN3 encoder")
    parser.add_argument("--generator", type=str, required=True, help="Path to StyleGAN3 generator pickle")
    parser.add_argument("--output", type=str, default="./simple_output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=64, help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--train_samples", type=int, default=5, help="Number of training samples")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Train encoder
    encoder = train_simple_encoder(
        generator_pkl=args.generator,
        output_dir=args.output,
        resolution=args.resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_samples=args.train_samples,
        lr=args.lr
    )