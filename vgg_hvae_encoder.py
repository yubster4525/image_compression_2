import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGG_HVAE_Encoder(nn.Module):
    """
    VGG-like encoder architecture designed to work with StyleGAN3.
    Features hierarchical structure that extracts features at block 2 and 5,
    and maps them to StyleGAN3's W+ latent space.
    """
    def __init__(
        self,
        img_resolution=1024,      # Input resolution
        img_channels=3,           # Number of input color channels
        w_dim=512,                # Intermediate latent (W) dimensionality in StyleGAN3
        num_ws=16,                # Number of intermediate latents in StyleGAN3 W+ space
        block_split=[5, 12],      # Where to split W vector indices for hierarchical encoding
        architecture='vgg',       # Architecture: 'vgg' or 'resnet'
        channel_base=32768,       # Overall multiplier for the number of channels
        channel_max=512,          # Maximum number of channels in any layer
    ):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.block_split = block_split
        
        # Ensure block split is valid
        assert block_split[0] < block_split[1] < num_ws, "Block split must divide W+ space into 3 parts"
        
        # Calculate number of W vectors in each hierarchy level
        self.num_ws_coarse = block_split[0]  # W vectors for coarse features
        self.num_ws_medium = block_split[1] - block_split[0]  # W vectors for medium features
        self.num_ws_fine = num_ws - block_split[1]  # W vectors for fine features
        
        # Calculate number of layers based on input resolution
        # For 1024x1024, we need 10 layers to reduce to 1x1
        # Each layer reduces size by factor of 2
        self.num_layers = int(np.log2(img_resolution))
        
        # VGG-like blocks with increasing channel counts
        channels = {
            0: min(channel_max, channel_base // (2 ** (self.num_layers - 1))),  # First layer
        }
        
        # Calculate number of channels per resolution
        for res in range(1, self.num_layers):
            channels[res] = min(channel_max, channel_base // (2 ** (self.num_layers - 1 - res)))
        
        # Build VGG-like architecture
        self.blocks = nn.ModuleList()
        self.from_rgb = nn.Conv2d(img_channels, channels[0], kernel_size=3, padding=1)
        
        # Create VGG blocks
        in_channels = channels[0]
        for i in range(self.num_layers - 1):
            out_channels = channels[i+1]
            block = VGGBlock(in_channels, out_channels)
            self.blocks.append(block)
            in_channels = out_channels
        
        # Create hierarchical feature extractors for each resolution level
        # We'll extract features at block 2 (early) and block 5 (mid/late)
        self.block2_idx = 1  # Block 2 (after 2 pooling layers) - resolution/4
        self.block5_idx = 4  # Block 5 (after 5 pooling layers) - resolution/32
        
        # Define lateral connections for hierarchical feature extraction
        block2_channels = channels[self.block2_idx + 1]  # Output channels of block 2
        block5_channels = channels[self.block5_idx + 1]  # Output channels of block 5
        final_channels = channels[self.num_layers - 1]   # Output channels of final block
        
        # Define feature projectors for each hierarchical level
        # Coarse features from final output
        self.coarse_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(final_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.num_ws_coarse * w_dim * 2)  # *2 for mean and logvar
        )
        
        # Medium features from block 5
        self.medium_projector = nn.Sequential(
            nn.Conv2d(block5_channels, block5_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(block5_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.num_ws_medium * w_dim * 2)  # *2 for mean and logvar
        )
        
        # Fine features from block 2
        self.fine_projector = nn.Sequential(
            nn.Conv2d(block2_channels, block2_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(block2_channels, block2_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(block2_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.num_ws_fine * w_dim * 2)  # *2 for mean and logvar
        )
        
    def forward(self, x):
        """
        Forward pass of the encoder.
        Args:
            x: Input image tensor of shape [batch_size, img_channels, img_resolution, img_resolution]
        Returns:
            w_plus: W+ latent tensor of shape [batch_size, num_ws, w_dim]
        """
        batch_size = x.shape[0]
        
        # Initial convolution
        x = self.from_rgb(x)
        
        # Store feature maps at different hierarchical levels
        features = {}
        
        # Forward through VGG blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Store features at specified blocks
            if i == self.block2_idx:
                features['fine'] = x
            elif i == self.block5_idx:
                features['medium'] = x
        
        # Final features for coarse level
        features['coarse'] = x
        
        # Project features to W+ space distributions (mean and logvar)
        coarse_params = self.coarse_projector(features['coarse'])
        medium_params = self.medium_projector(features['medium'])
        fine_params = self.fine_projector(features['fine'])
        
        # Split into mean and logvar
        coarse_mean, coarse_logvar = torch.chunk(
            coarse_params.view(batch_size, self.num_ws_coarse, self.w_dim * 2),
            2, dim=2
        )
        
        medium_mean, medium_logvar = torch.chunk(
            medium_params.view(batch_size, self.num_ws_medium, self.w_dim * 2),
            2, dim=2
        )
        
        fine_mean, fine_logvar = torch.chunk(
            fine_params.view(batch_size, self.num_ws_fine, self.w_dim * 2),
            2, dim=2
        )
        
        # Concatenate all means and logvars
        means = torch.cat([coarse_mean, medium_mean, fine_mean], dim=1)
        logvars = torch.cat([coarse_logvar, medium_logvar, fine_logvar], dim=1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        w_plus = means + eps * std
        
        return w_plus, means, logvars
    
    def encode_deterministic(self, x):
        """
        Encode without sampling (returns means only).
        Useful for evaluation or when you don't want variational sampling.
        """
        w_plus, means, _ = self(x)
        return means


class VGGBlock(nn.Module):
    """
    VGG-style block with two convolutions and a pooling layer.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Use average pooling like StyleGAN
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2)
        x = self.pool(x)
        return x


def create_stylegan3_compatible_encoder(G=None):
    """
    Create an encoder compatible with a given StyleGAN3 generator.
    Args:
        G: StyleGAN3 generator (if provided, will extract parameters from it)
    Returns:
        encoder: VGG_HVAE_Encoder instance
    """
    if G is not None:
        # Extract parameters from StyleGAN3 generator
        img_resolution = G.img_resolution
        img_channels = G.img_channels
        w_dim = G.w_dim
        num_ws = G.num_ws
        
        # Create encoder with matching parameters
        encoder = VGG_HVAE_Encoder(
            img_resolution=img_resolution,
            img_channels=img_channels,
            w_dim=w_dim,
            num_ws=num_ws
        )
    else:
        # Default parameters for a typical StyleGAN3 model
        encoder = VGG_HVAE_Encoder(
            img_resolution=1024,
            img_channels=3,
            w_dim=512,
            num_ws=16  # Updated based on our test for the 1024x1024 model
        )
    
    return encoder


class StyleGAN3Compressor(nn.Module):
    """
    Complete compressor model combining the VGG_HVAE_Encoder with a StyleGAN3 generator.
    """
    def __init__(self, encoder, generator):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        
        # Ensure generator weights are frozen
        for param in self.generator.parameters():
            param.requires_grad = False
    
    def forward(self, x, truncation_psi=1.0, noise_mode='const'):
        """
        Forward pass: encode image to W+ space and reconstruct with generator.
        Args:
            x: Input image tensor
            truncation_psi: Truncation factor for W space
            noise_mode: Noise mode for StyleGAN3 synthesis
        Returns:
            y: Reconstructed image
            w_plus: W+ latent representing the input image
        """
        # Encode to W+ space
        w_plus, _, _ = self.encoder(x)
        
        # Generate image from W+ latent
        y = self.generator.synthesis(w_plus, noise_mode=noise_mode)
        
        return y, w_plus
    
    def encode(self, x, deterministic=False):
        """
        Encode an image to W+ space.
        Args:
            x: Input image tensor
            deterministic: If True, returns mean vectors without sampling
        Returns:
            w_plus: W+ latent
        """
        if deterministic:
            return self.encoder.encode_deterministic(x)
        else:
            w_plus, _, _ = self.encoder(x)
            return w_plus
    
    def compress(self, x, quantization_bits=8, deterministic=True):
        """
        Compress an image to quantized W+ vectors.
        Args:
            x: Input image tensor
            quantization_bits: Number of bits per dimension
            deterministic: Whether to use deterministic encoding
        Returns:
            w_quantized: Quantized W+ vectors
        """
        # Encode image
        if deterministic:
            w_plus = self.encoder.encode_deterministic(x)
        else:
            w_plus, _, _ = self.encoder(x)
        
        # Quantize W+ vectors
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
            image: Reconstructed image
        """
        return self.generator.synthesis(w_plus, noise_mode=noise_mode)


# Example usage
if __name__ == "__main__":
    import pickle
    
    # Load StyleGAN3 generator
    with open('stylegan3-t-ffhq-1024x1024.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    # Create compatible encoder
    encoder = create_stylegan3_compatible_encoder(G)
    
    # Create compressor
    compressor = StyleGAN3Compressor(encoder, G)
    
    # Test with random input
    x = torch.randn(1, 3, 1024, 1024).cuda()
    y, w_plus = compressor(x)
    
    print(f"Input shape: {x.shape}")
    print(f"W+ shape: {w_plus.shape}")
    print(f"Output shape: {y.shape}")