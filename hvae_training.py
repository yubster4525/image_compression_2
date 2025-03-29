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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import pickle
import numpy as np
import lpips
from tqdm import tqdm

from vgg_hvae_encoder import VGG_HVAE_Encoder, StyleGAN3Compressor

# Define image dataset class
class ImageDataset(Dataset):
    def __init__(self, image_folder, resolution=1024):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.png")) + \
                          glob.glob(os.path.join(image_folder, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), 0  # Return 0 as a dummy label


def train_hvae_encoder(
    generator_pkl,
    data_path,
    output_dir='./output',
    epochs=50,
    batch_size=4,
    lr=1e-4,
    resume_from=None,
    img_resolution=1024,
    block_split=[4, 10],  # Split W+ space at indices 4 and 10
    kl_weight=0.01,
    perceptual_weight=0.8,
    rec_weight=1.0,
    noise_mode='const',
    save_every=5,
    fp16=False,
):
    """
    Train a VGG-HVAE encoder for StyleGAN3 image compression.
    
    Args:
        generator_pkl: Path to StyleGAN3 generator pickle
        data_path: Path to training images
        output_dir: Directory to save models and samples
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        resume_from: Path to checkpoint to resume from
        img_resolution: Image resolution
        block_split: Indices to split W+ space for hierarchical encoding
        kl_weight: Weight for KL divergence loss
        perceptual_weight: Weight for LPIPS perceptual loss
        rec_weight: Weight for reconstruction loss
        noise_mode: Noise mode for StyleGAN3 synthesis
        save_every: Save checkpoints every N epochs
        fp16: Whether to use mixed precision
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load StyleGAN3 generator
    print(f"Loading StyleGAN3 generator from {generator_pkl}")
    with open(generator_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    
    # Print generator info
    print(f"StyleGAN3 generator info:")
    print(f"  Resolution: {G.img_resolution}x{G.img_resolution}")
    print(f"  W dimensionality: {G.w_dim}")
    print(f"  Number of W vectors: {G.num_ws}")
    
    # Create encoder with matching parameters
    encoder = VGG_HVAE_Encoder(
        img_resolution=G.img_resolution,
        img_channels=G.img_channels,
        w_dim=G.w_dim,
        num_ws=G.num_ws,
        block_split=block_split,
    ).to(device)
    
    # Create compressor
    compressor = StyleGAN3Compressor(encoder, G).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Load checkpoint if specified
    start_epoch = 0
    if resume_from is not None and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Setup dataset and dataloader
    dataset = ImageDataset(data_path, resolution=img_resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Setup perceptual loss
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    # Extract W average from StyleGAN for KL divergence
    w_avg = G.mapping.w_avg.unsqueeze(0).unsqueeze(0).to(device)
    
    # Training loop
    print(f"Starting training for {epochs} epochs")
    for epoch in range(start_epoch, epochs):
        encoder.train()
        
        # Progress bar
        progress_bar = tqdm(dataloader)
        
        # Training metrics
        epoch_rec_loss = 0
        epoch_perceptual_loss = 0
        epoch_kl_loss = 0
        epoch_total_loss = 0
        num_batches = 0
        
        for batch_idx, (real_imgs, _) in enumerate(progress_bar):
            real_imgs = real_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            with torch.cuda.amp.autocast() if fp16 else torch.no_grad():
                # Encode and reconstruct
                reconstructed_imgs, w_plus = compressor(real_imgs, noise_mode=noise_mode)
                
                # Reconstruction loss (L2)
                rec_loss = F.mse_loss(real_imgs, reconstructed_imgs)
                
                # Perceptual loss (LPIPS)
                perceptual_loss = lpips_fn(real_imgs, reconstructed_imgs).mean()
                
                # KL divergence
                _, means, logvars = compressor.encoder(real_imgs)
                kl_loss = 0.5 * torch.mean(torch.sum(
                    torch.pow((means - w_avg), 2) + 
                    torch.exp(logvars) - logvars - 1,
                    dim=[1, 2]
                ))
                
                # Total loss
                loss = rec_weight * rec_loss + \
                       perceptual_weight * perceptual_loss + \
                       kl_weight * kl_loss
            
            # Backward pass with mixed precision if enabled
            if fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update metrics
            epoch_rec_loss += rec_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"MSE: {rec_loss.item():.4f} | "
                f"LPIPS: {perceptual_loss.item():.4f} | "
                f"KL: {kl_loss.item():.4f}"
            )
        
        # Compute epoch metrics
        epoch_rec_loss /= num_batches
        epoch_perceptual_loss /= num_batches
        epoch_kl_loss /= num_batches
        epoch_total_loss /= num_batches
        
        print(f"Epoch {epoch+1} summary:")
        print(f"  Total Loss: {epoch_total_loss:.6f}")
        print(f"  MSE Loss: {epoch_rec_loss:.6f}")
        print(f"  LPIPS Loss: {epoch_perceptual_loss:.6f}")
        print(f"  KL Loss: {epoch_kl_loss:.6f}")
        
        # Save samples and checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            # Generate samples
            encoder.eval()
            with torch.no_grad():
                # Get a batch of images for visualization
                sample_imgs = next(iter(dataloader))[0][:4].to(device)
                
                # Original vs. reconstructed
                reconstructed_imgs, _ = compressor(sample_imgs, noise_mode=noise_mode)
                
                # Convert to numpy for saving
                sample_imgs = (sample_imgs.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
                reconstructed_imgs = (reconstructed_imgs.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
                
                # Create grid
                grid = []
                for i in range(len(sample_imgs)):
                    grid.append(np.concatenate([sample_imgs[i], reconstructed_imgs[i]], axis=1))
                grid = np.concatenate(grid, axis=0)
                
                # Save grid
                import imageio
                grid = (grid * 255).astype(np.uint8)
                imageio.imsave(
                    os.path.join(output_dir, 'samples', f'epoch_{epoch+1:04d}.png'),
                    grid
                )
            
            # Save checkpoint
            torch.save(
                {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_total_loss,
                },
                os.path.join(output_dir, 'checkpoints', f'encoder_epoch_{epoch+1:04d}.pt')
            )
    
    # Save final model
    torch.save(
        {
            'encoder': encoder.state_dict(),
            'config': {
                'img_resolution': img_resolution,
                'img_channels': G.img_channels,
                'w_dim': G.w_dim,
                'num_ws': G.num_ws,
                'block_split': block_split,
            }
        },
        os.path.join(output_dir, 'hvae_encoder_final.pt')
    )
    
    print("Training complete!")
    return encoder


def test_compression(
    encoder_path,
    generator_pkl,
    test_image_path,
    output_dir='./compression_test',
    quantization_bits=8,
    noise_mode='const',
):
    """
    Test image compression using a trained HVAE encoder and StyleGAN3 generator.
    
    Args:
        encoder_path: Path to trained encoder checkpoint
        generator_pkl: Path to StyleGAN3 generator pickle
        test_image_path: Path to test image
        output_dir: Directory to save results
        quantization_bits: Number of bits for quantization
        noise_mode: Noise mode for StyleGAN3 synthesis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load StyleGAN3 generator
    with open(generator_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    
    # Load encoder
    checkpoint = torch.load(encoder_path)
    if 'encoder' in checkpoint:
        encoder_state_dict = checkpoint['encoder']
        config = checkpoint['config']
        encoder = VGG_HVAE_Encoder(
            img_resolution=config['img_resolution'],
            img_channels=config['img_channels'],
            w_dim=config['w_dim'],
            num_ws=config['num_ws'],
            block_split=config['block_split'],
        ).to(device)
        encoder.load_state_dict(encoder_state_dict)
    else:
        encoder_state_dict = checkpoint['encoder_state_dict']
        encoder = VGG_HVAE_Encoder(
            img_resolution=G.img_resolution,
            img_channels=G.img_channels,
            w_dim=G.w_dim,
            num_ws=G.num_ws,
        ).to(device)
        encoder.load_state_dict(encoder_state_dict)
    
    # Create compressor
    compressor = StyleGAN3Compressor(encoder, G).to(device)
    compressor.eval()
    
    # Load and preprocess test image
    transform = transforms.Compose([
        transforms.Resize((G.img_resolution, G.img_resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # to [-1, 1]
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Compress and decompress
    with torch.no_grad():
        # Original reconstruction (no quantization)
        recon_no_quant, w_plus = compressor(image_tensor, noise_mode=noise_mode)
        
        # Compress with quantization
        w_quantized = compressor.compress(image_tensor, quantization_bits=quantization_bits)
        
        # Decompress
        recon_quantized = compressor.decompress(w_quantized, noise_mode=noise_mode)
    
    # Calculate compression rate
    total_w_dims = w_plus.numel()
    bytes_per_w_dim = quantization_bits / 8
    compressed_size_bytes = total_w_dims * bytes_per_w_dim
    original_size_bytes = image_tensor.numel() * 4  # Assuming 32-bit float per element
    compression_ratio = original_size_bytes / compressed_size_bytes
    bpp = compressed_size_bytes * 8 / (G.img_resolution * G.img_resolution)  # Bits per pixel
    
    print(f"Compression results:")
    print(f"  Original size: {original_size_bytes/1024:.2f} KB")
    print(f"  Compressed size: {compressed_size_bytes/1024:.2f} KB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Bits per pixel: {bpp:.4f}")
    
    # Save original and reconstructed images
    original_np = (image_tensor.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    recon_no_quant_np = (recon_no_quant.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    recon_quantized_np = (recon_quantized.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    
    import imageio
    imageio.imsave(os.path.join(output_dir, 'original.png'), (original_np * 255).astype(np.uint8))
    imageio.imsave(os.path.join(output_dir, 'reconstructed_no_quant.png'), (recon_no_quant_np * 255).astype(np.uint8))
    imageio.imsave(os.path.join(output_dir, f'reconstructed_{quantization_bits}bit.png'), (recon_quantized_np * 255).astype(np.uint8))
    
    # Create comparison grid
    grid = np.concatenate([original_np, recon_no_quant_np, recon_quantized_np], axis=1)
    imageio.imsave(os.path.join(output_dir, 'comparison.png'), (grid * 255).astype(np.uint8))
    
    # Calculate metrics
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Convert to uint8 for metric calculation
    original_uint8 = (original_np * 255).astype(np.uint8)
    recon_no_quant_uint8 = (recon_no_quant_np * 255).astype(np.uint8)
    recon_quantized_uint8 = (recon_quantized_np * 255).astype(np.uint8)
    
    # Calculate PSNR and SSIM
    psnr_no_quant = psnr(original_uint8, recon_no_quant_uint8)
    ssim_no_quant = ssim(original_uint8, recon_no_quant_uint8, channel_axis=2, data_range=255)
    
    psnr_quantized = psnr(original_uint8, recon_quantized_uint8)
    ssim_quantized = ssim(original_uint8, recon_quantized_uint8, channel_axis=2, data_range=255)
    
    print(f"Quality metrics (no quantization):")
    print(f"  PSNR: {psnr_no_quant:.2f} dB")
    print(f"  SSIM: {ssim_no_quant:.4f}")
    
    print(f"Quality metrics ({quantization_bits}-bit quantization):")
    print(f"  PSNR: {psnr_quantized:.2f} dB")
    print(f"  SSIM: {ssim_quantized:.4f}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Compression settings:\n")
        f.write(f"  Quantization bits: {quantization_bits}\n")
        f.write(f"  W dimensionality: {G.w_dim}\n")
        f.write(f"  Number of W vectors: {G.num_ws}\n\n")
        
        f.write(f"Compression results:\n")
        f.write(f"  Original size: {original_size_bytes/1024:.2f} KB\n")
        f.write(f"  Compressed size: {compressed_size_bytes/1024:.2f} KB\n")
        f.write(f"  Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"  Bits per pixel: {bpp:.4f}\n\n")
        
        f.write(f"Quality metrics (no quantization):\n")
        f.write(f"  PSNR: {psnr_no_quant:.2f} dB\n")
        f.write(f"  SSIM: {ssim_no_quant:.4f}\n\n")
        
        f.write(f"Quality metrics ({quantization_bits}-bit quantization):\n")
        f.write(f"  PSNR: {psnr_quantized:.2f} dB\n")
        f.write(f"  SSIM: {ssim_quantized:.4f}\n")
    
    return compression_ratio, psnr_quantized, ssim_quantized


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and test HVAE encoder for StyleGAN3")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], required=True)
    parser.add_argument("--generator", type=str, required=True, help="Path to StyleGAN3 generator pickle")
    parser.add_argument("--data", type=str, help="Path to training images")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--block_split", type=str, default="4,10", help="Indices to split W+ space")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="KL loss weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.8, help="Perceptual loss weight")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--noise_mode", type=str, default="const", help="Noise mode for StyleGAN3")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoints every N epochs")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    
    # Testing arguments
    parser.add_argument("--encoder", type=str, help="Path to trained encoder checkpoint")
    parser.add_argument("--test_image", type=str, help="Path to test image")
    parser.add_argument("--quant_bits", type=int, default=8, help="Quantization bits")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        assert args.data is not None, "Training data path must be provided"
        
        block_split = [int(x) for x in args.block_split.split(',')]
        assert len(block_split) == 2, "Block split must have two values"
        
        train_hvae_encoder(
            generator_pkl=args.generator,
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_from=args.resume,
            img_resolution=args.resolution,
            block_split=block_split,
            kl_weight=args.kl_weight,
            perceptual_weight=args.perceptual_weight,
            rec_weight=args.rec_weight,
            noise_mode=args.noise_mode,
            save_every=args.save_every,
            fp16=args.fp16,
        )
    
    elif args.mode == 'test':
        assert args.encoder is not None, "Encoder checkpoint path must be provided"
        assert args.test_image is not None, "Test image path must be provided"
        
        test_compression(
            encoder_path=args.encoder,
            generator_pkl=args.generator,
            test_image_path=args.test_image,
            output_dir=args.output,
            quantization_bits=args.quant_bits,
            noise_mode=args.noise_mode,
        )