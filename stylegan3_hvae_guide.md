# StyleGAN3 + HVAE for Image Compression: A Practical Guide

## 1. Understanding the Basics

### What is StyleGAN3?
StyleGAN3 is a generative model that creates high-quality images from random noise. It works by:
- Taking random latent codes (Z-space, 512 dimensions)
- Transforming them through a mapping network to W-space
- Using a synthesis network to generate images

Key insight: StyleGAN3 doesn't have an encoder by default - it only generates images.

### What is HVAE?
A Hierarchical Variational Autoencoder (HVAE) is a type of encoder-decoder that:
- Encodes images into a compressed latent representation
- Adds variational constraints for better latent organization
- Uses hierarchical structure to capture details at different levels

## 1.1 Compatible Hyperparameters

### StyleGAN3 Key Hyperparameters
| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `z_dim` | 512 | Dimensionality of initial latent space |
| `w_dim` | 512 | Dimensionality of intermediate latent space |
| `num_ws` | 14 (1024×1024 model) | Number of W vectors in W+ space |
| `img_resolution` | 1024 or 512 | Output image resolution |
| `img_channels` | 3 | Number of output color channels |
| `channel_base` | 32768 | Overall multiplier for the number of channels |
| `channel_max` | 512 | Maximum number of channels in any layer |

### HVAE Encoder Hyperparameters (must match StyleGAN3)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `w_dim` | 512 | **Must match** StyleGAN3's `w_dim` |
| `num_ws` | 14 (1024×1024 model) | **Must match** StyleGAN3's `num_ws` |
| `img_resolution` | 1024 or 512 | **Must match** StyleGAN3's `img_resolution` |
| `img_channels` | 3 | **Must match** StyleGAN3's `img_channels` |
| `encoder_channels_base` | 32768 | Can be adjusted for encoder capacity |
| `encoder_channels_max` | 512 | Maximum channels in encoder |

### Training Hyperparameters
| Parameter | Recommended Range | Description |
|-----------|------------------|-------------|
| `learning_rate` | 1e-4 to 5e-4 | Adam optimizer learning rate |
| `beta1` | 0.9 | Adam optimizer beta1 |
| `beta2` | 0.999 | Adam optimizer beta2 |
| `batch_size` | 8-32 | Depends on GPU memory |
| `kl_weight` | 0.005-0.02 | Weight for KL divergence loss |
| `perceptual_weight` | 0.1-1.0 | Weight for LPIPS perceptual loss |
| `truncation_psi` | 0.5-0.7 | Truncation trick parameter for sampling |

### Compression Hyperparameters
| Parameter | Range | Effect |
|-----------|-------|--------|
| `quantization_bits` | 4-12 | Bits per latent dimension |
| `quantization_level` | 2-12 | Related to step size (1.0/(2^bits)) |
| `progressive_coding` | boolean | Whether to use progressive compression |
| `entropy_coder` | 'arithmetic', 'range', 'ans' | Type of entropy coding |

## 2. The Integration Approach

### The Big Picture
```
┌─────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ Image   │───►│ HVAE Encoder │───►│ Compressed │───►│ StyleGAN3    │───►│ Reconstructed │
│ Input   │    │ (Custom)     │    │ Latent     │    │ Generator    │    │ Image         │
└─────────┘    └──────────────┘    └────────────┘    └──────────────┘    └──────────────┘
```

### How It Works Step-by-Step
1. **Encoding**: Your custom HVAE encoder converts real images to StyleGAN3's W+ latent space
2. **Compression**: The W+ vectors are quantized and compressed
3. **Decoding**: StyleGAN3's pre-trained generator reconstructs the image from W+ vectors

## 3. Practical Implementation

### Step 1: Set Up StyleGAN3
```python
# Load pre-trained StyleGAN3 generator
with open('stylegan3-t-ffhq-1024x1024.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

# Freeze the generator weights
for param in G.parameters():
    param.requires_grad = False
```

### Step 2: Create Your HVAE Encoder
```python
class HVAEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Mirror StyleGAN3's synthesis network in reverse
        self.encoder_blocks = nn.ModuleList([
            # Example: Series of Conv2d layers with downsampling
            EncoderBlock(channels_in=3, channels_out=64),  # 1024 → 512
            EncoderBlock(channels_in=64, channels_out=128),  # 512 → 256
            # ... more blocks
            EncoderBlock(channels_in=512, channels_out=512),  # 16 → 8
        ])
        
        # Final layers to produce W+ space vectors (14 style vectors for 1024×1024)
        self.to_w = nn.ModuleList([
            nn.Linear(512, 512*2)  # 2x dims for mean and logvar
            for _ in range(14)  # 14 different style vectors for 1024×1024 model
        ])
    
    def forward(self, x):
        # Extract features
        features = x
        for block in self.encoder_blocks:
            features = block(features)
            
        # Create W vectors with variational sampling
        w_vectors = []
        for i in range(14):
            mean_logvar = self.to_w[i](features.view(features.size(0), -1))
            mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            w = mean + eps * std
            w_vectors.append(w)
            
        # Stack W vectors to create W+ tensor
        return torch.stack(w_vectors, dim=1)  # [batch, 14, 512]
```

### Step 3: Create a Combined Model
```python
class StyleGAN3Compressor(torch.nn.Module):
    def __init__(self, encoder, generator):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
    
    def forward(self, x):
        # Encode input image to W+ space
        w_plus = self.encoder(x)
        
        # Generate reconstructed image
        img = self.generator.synthesis(w_plus)
        
        return img, w_plus
    
    def compress(self, x, quantization_level=8):
        # Encode to W+ space 
        w_plus = self.encoder(x)
        
        # Simple quantization (in a real system, use entropy coding too)
        step_size = 1.0 / (2**quantization_level)
        w_plus_quantized = torch.round(w_plus / step_size) * step_size
        
        return w_plus_quantized
    
    def decompress(self, w_plus):
        # Generate image from W+ vectors
        return self.generator.synthesis(w_plus)
```

### Step 4: Train the HVAE Encoder
```python
def train_hvae_encoder(compressor, dataloader, num_epochs):
    # Configuration settings
    config = {
        'lr': 0.0001,
        'betas': (0.9, 0.999),
        'kl_weight': 0.01,
        'perceptual_weight': 0.8,
        'feature_match_weight': 0.2,
        'rec_weight': 1.0,
        'truncation_psi': 0.7,
        'truncation_cutoff': 8,
    }
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        compressor.encoder.parameters(), 
        lr=config['lr'], 
        betas=config['betas']
    )
    
    # Initialize perceptual loss if needed
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    
    # Prepare for feature matching if used
    if config['feature_match_weight'] > 0:
        # Access discriminator features - if available in your generator pickle
        discriminator = pickle.load(open('stylegan3.pkl', 'rb'))['D'].cuda()
        
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.cuda()
            
            # Forward pass
            reconstructed_imgs, w_plus = compressor(real_imgs)
            
            # Calculate losses
            # 1. Reconstruction loss (MSE)
            rec_loss = F.mse_loss(real_imgs, reconstructed_imgs)
            
            # 2. Perceptual loss (LPIPS)
            perceptual_loss = lpips_fn(real_imgs, reconstructed_imgs).mean()
            
            # 3. KL divergence (variational regularization)
            # Get StyleGAN's average W vector
            w_avg = compressor.generator.mapping.w_avg.unsqueeze(0).unsqueeze(0)
            
            # Calculate KL divergence for variational constraint
            kl_loss = 0.5 * torch.mean(torch.sum(
                torch.pow((w_plus - w_avg), 2) + 
                torch.exp(w_plus) - 1 - w_plus, 
                dim=[1, 2]))
            
            # 4. Feature matching loss (optional)
            feature_match_loss = torch.tensor(0.0).to(real_imgs.device)
            if config['feature_match_weight'] > 0:
                # Extract discriminator features for real and reconstructed
                real_features = discriminator.extract_features(real_imgs)
                recon_features = discriminator.extract_features(reconstructed_imgs)
                
                # Calculate L2 distance between feature maps
                for real_feat, recon_feat in zip(real_features, recon_features):
                    feature_match_loss += F.mse_loss(real_feat, recon_feat)
            
            # 5. Total weighted loss
            loss = (
                config['rec_weight'] * rec_loss + 
                config['kl_weight'] * kl_loss + 
                config['perceptual_weight'] * perceptual_loss +
                config['feature_match_weight'] * feature_match_loss
            )
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log training progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Rec: {rec_loss.item():.4f} | "
                      f"KL: {kl_loss.item():.4f} | "
                      f"Perceptual: {perceptual_loss.item():.4f}")
                
        # After each epoch, save a sample reconstruction
        if epoch % 5 == 0:
            save_image_grid(
                torch.cat([real_imgs[:4], reconstructed_imgs[:4]], dim=0),
                f'samples/epoch_{epoch:04d}.png',
                nrow=4
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': compressor.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f'checkpoints/encoder_epoch_{epoch:04d}.pt')
```

## 4. Practical Compression Steps

### Compressing a Single Image
```python
# Load or create the compression model
compressor = StyleGAN3Compressor(hvae_encoder, stylegan_generator)

# Load an image
img = load_and_preprocess_image("input.png")  # Should scale to [-1, 1] range

# Compress the image (returns W+ latent)
compressed_latents = compressor.compress(img, quantization_level=8)

# At this point, you'd normally apply entropy coding (arithmetic coding)
# Save compressed latents to file
save_compressed_data(compressed_latents, "compressed.bin")
```

### Decompressing a Single Image
```python
# Load compressed data
compressed_latents = load_compressed_data("compressed.bin")

# Decompress to get image
reconstructed_img = compressor.decompress(compressed_latents)

# Save or display the image
save_image(reconstructed_img, "reconstructed.png")
```

## 5. Key Advantages of This Approach

1. **Better Compression Ratio**: StyleGAN3 captures detailed priors about natural images
2. **High Perceptual Quality**: Generates realistic results even at low bitrates
3. **Flexibility**: Can adjust compression rate by changing quantization
4. **Domain-Specific**: Can be optimized for specific types of images (faces, cars, etc.)

## 6. How to Access StyleGAN3's Structure

StyleGAN3's generator has two main parts:
1. **Mapping Network**: Access via `G.mapping`
2. **Synthesis Network**: Access via `G.synthesis`

To see the detailed architecture:
```python
# Print StyleGAN3 structure
print(G)

# Access specific parts
z_dim = G.z_dim  # Typically 512
w_dim = G.w_dim  # Typically 512
num_ws = G.num_ws  # Number of style vectors (14 for 1024×1024)
```

## 7. Additional Tips

1. **Start simple**: Begin by training only to reconstruct the first few layers
2. **Progressive training**: Add more layers to the W+ representation gradually
3. **Pretrain the encoder**: Use standard autoencoder training, then add variational properties
4. **Experiment with layers**: Different W+ layers control different details:
   - Early layers: Overall structure and composition
   - Middle layers: Medium-scale features
   - Late layers: Fine details and textures

## 8. Advanced Hyperparameter Configurations

### LayerWise Compression Settings

For optimal compression, each W vector layer can be allocated different bit precision based on its perceptual importance:

| Layer Index | Typical Bit Allocation | Perceptual Impact |
|-------------|------------------------|-------------------|
| 0-1 | 10-12 bits | Global structure, pose, composition |
| 2-5 | 8-10 bits | Intermediate features, broad textures |
| 6-10 | 6-8 bits | Medium details, facial features |
| 11-13 | 4-6 bits | Fine details, skin texture, hair |

### HVAE Encoder Architecture Details

| Layer | Output Size | Channels | Kernel | Stride | Notes |
|-------|------------|----------|--------|--------|-------|
| Input | H×W×3 | 3 | - | - | Normalized to [-1, 1] |
| Conv1 | H/2×W/2×64 | 64 | 3×3 | 2 | LeakyReLU(0.2) |
| Conv2 | H/4×W/4×128 | 128 | 3×3 | 2 | LeakyReLU(0.2), IN |
| Conv3 | H/8×W/8×256 | 256 | 3×3 | 2 | LeakyReLU(0.2), IN |
| Conv4 | H/16×W/16×512 | 512 | 3×3 | 2 | LeakyReLU(0.2), IN |
| Conv5 | H/32×W/32×512 | 512 | 3×3 | 2 | LeakyReLU(0.2), IN |
| Conv6 | H/64×W/64×512 | 512 | 3×3 | 2 | LeakyReLU(0.2), IN |
| GAP | 1×1×512 | 512 | - | - | Global Average Pooling |
| FC-W+ | 14×512×2 | - | - | - | Linear layers for mean & logvar |

Where IN = Instance Normalization, GAP = Global Average Pooling

### Loss Function Weights for Different Applications

| Application | Rec Loss | KL Loss | Perceptual Loss | Feature Match Loss |
|-------------|----------|---------|-----------------|-------------------|
| High Quality | 1.0 | 0.005 | 0.8 | 0.2 |
| High Compression | 0.6 | 0.02 | 1.0 | 0.1 |
| Artistic | 0.4 | 0.01 | 1.2 | 0.0 |
| Photo-realistic | 1.0 | 0.005 | 1.0 | 0.5 |

### Multi-Scale HVAE Configurations

For improved hierarchical encoding:

```python
class MultiScaleHVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_64 = EncoderBlock(resolution=64)  # Coarse features
        self.encoder_128 = EncoderBlock(resolution=128)  # Medium features
        self.encoder_256 = EncoderBlock(resolution=256)  # Fine features
        
        # W vector allocations by resolution
        self.w_alloc = {
            'coarse': [0, 1, 2, 3],      # From 64×64 encoder
            'medium': [4, 5, 6, 7, 8],   # From 128×128 encoder
            'fine': [9, 10, 11, 12, 13]  # From 256×256 encoder
        }
        
        # Project each feature level to appropriate W vectors
        self.to_w_coarse = nn.ModuleList([
            nn.Linear(512, 512*2) for _ in range(len(self.w_alloc['coarse']))
        ])
        
        # Similar projections for medium and fine levels
    
    def forward(self, x):
        # Multi-scale encoding at different resolutions
        # Returns W+ with appropriate hierarchical structure
```