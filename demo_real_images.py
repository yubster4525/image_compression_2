"""
Demo script to demonstrate training the StyleGAN3 HVAE on a small set of real images.
Can work with any folder of images, not just ImageNet.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from stylegan3_hvae_full import train_hvae_encoder, ImageDataset

def create_result_grid(original_dir, reconstructed_dir, output_path, num_samples=5):
    """
    Create a grid comparing original images with their reconstructions.
    Args:
        original_dir: Directory containing original images
        reconstructed_dir: Directory containing reconstructed images
        output_path: Path to save the comparison grid image
        num_samples: Number of samples to include in the grid
    """
    # Get image paths
    orig_imgs = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')])
    recon_imgs = sorted([f for f in os.listdir(reconstructed_dir) if f.endswith('.png') and 'reconstructed' in f])
    
    # Limit to specified number of samples
    orig_imgs = orig_imgs[:num_samples]
    recon_imgs = recon_imgs[:num_samples]
    
    # Create figure
    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
    
    # Plot images
    for i in range(num_samples):
        if i < len(orig_imgs):
            # Original
            orig_img = Image.open(os.path.join(original_dir, orig_imgs[i]))
            axs[0, i].imshow(np.array(orig_img))
            axs[0, i].set_title('Original')
            axs[0, i].axis('off')
            
            # Reconstructed
            if i < len(recon_imgs):
                recon_img = Image.open(os.path.join(reconstructed_dir, recon_imgs[i]))
                axs[1, i].imshow(np.array(recon_img))
                axs[1, i].set_title('Reconstructed')
                axs[1, i].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved comparison grid to {output_path}")

def demo_train_on_real_images(
    generator_pkl,
    dataset_path,
    output_dir='./real_images_output',
    is_imagenet=False,
    resolution=256,
    batch_size=4,
    num_epochs=5,
    device=None,
    num_workers=4,
):
    """
    Run a demo training run on a small set of real images.
    
    Args:
        generator_pkl: Path to StyleGAN3 generator pickle
        dataset_path: Path to folder of real images
        output_dir: Directory to save results
        is_imagenet: Whether dataset has ImageNet folder structure
        resolution: Training resolution
        batch_size: Batch size
        num_epochs: Number of epochs
        device: Device to use (cuda, mps, cpu)
        num_workers: Number of dataloader workers
    """
    print(f"\n✨ StyleGAN3 HVAE Training Demo on Real Images ✨")
    print(f"Dataset: {dataset_path}")
    print(f"Generator: {generator_pkl}")
    print(f"Output directory: {output_dir}")
    print(f"Training resolution: {resolution}x{resolution}")
    print(f"Training for {num_epochs} epochs\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the model
    encoder, history = train_hvae_encoder(
        generator_pkl=generator_pkl,
        output_dir=output_dir,
        training_resolution=resolution,
        batch_size=batch_size,
        max_resolution=1024,  # Use full resolution encoder for better quality
        num_epochs=num_epochs,
        lr=1e-4,
        kl_weight=0.01,
        perceptual_weight=0.8,
        rec_weight=1.0,
        fp16=torch.cuda.is_available(),  # Use FP16 if CUDA is available
        device_override=device,
        num_workers=num_workers,
        save_every=1,  # Save every epoch for the demo
        dataset_path=dataset_path,
        is_imagenet=is_imagenet,
    )
    
    print("\n✅ Training completed!")
    
    # Create comparison grid
    create_result_grid(
        original_dir=output_dir + '/samples',
        reconstructed_dir=output_dir + '/samples',
        output_path=output_dir + '/results_comparison.png'
    )
    
    # Visualize training losses
    plt.figure(figsize=(10, 6))
    plt.plot(history['rec_loss'], label='Reconstruction Loss')
    plt.plot(history['perceptual_loss'], label='Perceptual Loss')
    plt.plot(history['kl_loss'], label='KL Loss')
    plt.plot(history['total_loss'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir + '/training_losses.png', dpi=300)
    
    print(f"\nResults and visualizations saved to {output_dir}")
    print("Demo completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: Train StyleGAN3 HVAE on Real Images")
    parser.add_argument("--generator", type=str, default="models/stylegan3-t-ffhq-1024x1024.pkl",
                        help="Path to StyleGAN3 generator pickle")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to folder of real images")
    parser.add_argument("--output", type=str, default="./real_images_output",
                        help="Output directory")
    parser.add_argument("--imagenet", action="store_true",
                        help="Treat dataset as ImageNet with class subdirectories")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    demo_train_on_real_images(
        generator_pkl=args.generator,
        dataset_path=args.dataset,
        output_dir=args.output,
        is_imagenet=args.imagenet,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        num_workers=args.workers,
    )