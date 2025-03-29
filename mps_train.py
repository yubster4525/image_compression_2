import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import pickle

# Add StyleGAN3 repo to path
sys.path.insert(0, os.path.abspath('stylegan3'))

# Import StyleGAN3's dependencies if needed
try:
    from torch_utils import misc
    import dnnlib
except ImportError:
    print("Warning: StyleGAN3 dependencies not found. Generator features may not work.")

# Simple encoder for StyleGAN3
class SimpleEncoder(nn.Module):
    def __init__(self, img_resolution=64, img_channels=3, w_dim=512, num_ws=16):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        
        # Simple encoder with 4 convolutional layers
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Final size after convolutions
        self.final_resolution = img_resolution // 16
        self.final_channels = 256
        
        # Project to W+ space
        self.to_w = nn.Linear(self.final_resolution * self.final_resolution * self.final_channels, 
                             num_ws * w_dim * 2)  # *2 for mean and logvar
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Encoder backbone
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Project to W+ space
        w_params = self.to_w(x)
        
        # Reshape to [batch_size, num_ws, w_dim * 2]
        w_params = w_params.view(batch_size, self.num_ws, self.w_dim * 2)
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(w_params, 2, dim=2)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        w_plus = mean + eps * std
        
        return w_plus, mean, logvar

def run_mps_training():
    print("Running minimalist MPS training test")
    
    # Use MPS
    device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Create a simple 64x64 noise image for testing
    resolution = 64
    batch_size = 1
    
    # Generate random images directly instead of loading StyleGAN
    print(f"Creating random test images at {resolution}x{resolution}")
    test_images = torch.randn(batch_size, 3, resolution, resolution).to(device)
    
    # Create encoder with small dimensions
    encoder = SimpleEncoder(
        img_resolution=resolution,
        img_channels=3,
        w_dim=64,  # Reduced from 512
        num_ws=8   # Reduced from 16
    ).to(device)
    print("Created simple encoder")
    
    # Simple decoder (just for testing)
    decoder = nn.Sequential(
        nn.Linear(64 * 8, 256 * 4 * 4),  # w_dim * num_ws -> features
        nn.LeakyReLU(0.2),
        nn.Unflatten(1, (256, 4, 4)),
        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        nn.Tanh()
    ).to(device)
    print("Created simple decoder")
    
    # Create optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    
    # Create prior (W average)
    w_avg = torch.zeros(1, 1, 64).to(device)
    
    # Mini training loop
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        w_plus, means, logvars = encoder(test_images)
        
        # Reshape for decoder
        w_flat = w_plus.reshape(batch_size, -1)
        reconstructed = decoder(w_flat)
        
        # Calculate losses
        rec_loss = F.mse_loss(test_images, reconstructed)
        
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
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Rec: {rec_loss.item():.4f} | KL: {kl_loss.item():.4f}")
    
    print("Training completed successfully!")
    
    # Save results
    save_tensor_as_image(test_images[0].cpu(), "mps_test_original.png")
    save_tensor_as_image(reconstructed[0].detach().cpu(), "mps_test_reconstructed.png")
    
    return "MPS training completed successfully"

def save_tensor_as_image(tensor, filename):
    """Convert a tensor to a PIL Image and save it."""
    # Convert tensor to numpy array
    img = tensor.numpy()
    
    # Scale from [-1, 1] to [0, 255]
    img = ((img.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    
    # Create PIL Image and save
    Image.fromarray(img).save(filename)

if __name__ == "__main__":
    result = run_mps_training()
    print(result)