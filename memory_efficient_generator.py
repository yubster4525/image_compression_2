import os
import sys
import argparse
import torch
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
    print("Warning: StyleGAN3 dependencies not found. Make sure the StyleGAN3 repository is available.")

class MemoryEfficientGenerator:
    """
    Memory-efficient wrapper for StyleGAN3 generator that allows generating
    images one by one or in small batches to manage memory usage.
    """
    def __init__(
        self,
        generator_pkl,
        output_dir='./generated_images',
        resolution=None,
        device=None,
        truncation_psi=0.7,
        noise_mode='const',
    ):
        self.output_dir = output_dir
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        self.resolution = resolution
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup device
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using MPS (Metal Performance Shaders) on Mac")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        print(f"Using device: {self.device}")
        
        # Load generator
        print(f"Loading StyleGAN3 generator from {generator_pkl}")
        with open(generator_pkl, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(self.device)
        
        # Print generator info
        print(f"StyleGAN3 generator info:")
        print(f"  Resolution: {self.G.img_resolution}x{self.G.img_resolution}")
        print(f"  W dimensionality: {self.G.w_dim}")
        print(f"  Number of W vectors: {self.G.num_ws}")
    
    def generate_images(
        self,
        num_images,
        batch_size=1,
        seed=None,
        save_images=True,
        clear_memory=True,
        class_idx=None,
    ):
        """
        Generate images one by one or in small batches to manage memory efficiently.
        
        Args:
            num_images: Number of images to generate
            batch_size: Batch size for generation (use 1 for lowest memory usage)
            seed: Random seed for reproducibility
            save_images: Whether to save images to disk
            clear_memory: Whether to clear CUDA/MPS memory after each batch
            class_idx: Class index for conditional generation (if applicable)
            
        Returns:
            List of generated images (PIL.Image objects) if save_images=False,
            otherwise list of saved image paths
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate images in batches
        all_images = []
        saved_paths = []
        
        for i in tqdm(range(0, num_images, batch_size), desc="Generating images"):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_images - i)
            
            # Generate random latents
            z = torch.randn(current_batch_size, self.G.z_dim).to(self.device)
            
            # Generate images
            with torch.no_grad():
                # Apply truncation trick (optional)
                ws = self.G.mapping(z, None, truncation_psi=self.truncation_psi)
                
                # Generate images
                img = self.G.synthesis(ws, noise_mode=self.noise_mode)
                
                # Resize if needed
                if self.resolution is not None and self.resolution != self.G.img_resolution:
                    img = torch.nn.functional.interpolate(
                        img, 
                        size=(self.resolution, self.resolution),
                        mode='bilinear',
                        align_corners=False
                    )
            
            # Convert to numpy and process
            imgs_np = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            
            # Process each image in the batch
            for j in range(current_batch_size):
                img_idx = i + j
                if save_images:
                    # Save image
                    img_path = os.path.join(self.output_dir, f"image_{img_idx:05d}.png")
                    Image.fromarray(imgs_np[j]).save(img_path)
                    saved_paths.append(img_path)
                else:
                    # Add to list
                    all_images.append(Image.fromarray(imgs_np[j]))
            
            # Clear memory if requested
            if clear_memory:
                del z, ws, img, imgs_np
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        return saved_paths if save_images else all_images
    
    def generate_from_seeds(
        self,
        seeds,
        batch_size=1,
        save_images=True,
        clear_memory=True,
        class_idx=None,
    ):
        """
        Generate images from specific seeds.
        
        Args:
            seeds: List of integer random seeds
            batch_size: Batch size for generation (use 1 for lowest memory usage)
            save_images: Whether to save images to disk
            clear_memory: Whether to clear memory after each batch
            class_idx: Class index for conditional generation (if applicable)
            
        Returns:
            List of generated images or saved image paths
        """
        # Generate images in batches
        all_images = []
        saved_paths = []
        
        for i in tqdm(range(0, len(seeds), batch_size), desc="Generating from seeds"):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, len(seeds) - i)
            current_seeds = seeds[i:i+current_batch_size]
            
            # Generate latents from seeds
            z_batch = []
            for seed in current_seeds:
                torch.manual_seed(seed)
                z = torch.randn(1, self.G.z_dim, device=self.device)
                z_batch.append(z)
            
            z = torch.cat(z_batch, dim=0)
            
            # Generate images
            with torch.no_grad():
                # Apply truncation trick (optional)
                ws = self.G.mapping(z, None, truncation_psi=self.truncation_psi)
                
                # Generate images
                img = self.G.synthesis(ws, noise_mode=self.noise_mode)
                
                # Resize if needed
                if self.resolution is not None and self.resolution != self.G.img_resolution:
                    img = torch.nn.functional.interpolate(
                        img, 
                        size=(self.resolution, self.resolution),
                        mode='bilinear',
                        align_corners=False
                    )
            
            # Convert to numpy and process
            imgs_np = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            
            # Process each image in the batch
            for j in range(current_batch_size):
                seed = current_seeds[j]
                if save_images:
                    # Save image
                    img_path = os.path.join(self.output_dir, f"seed_{seed}.png")
                    Image.fromarray(imgs_np[j]).save(img_path)
                    saved_paths.append(img_path)
                else:
                    # Add to list
                    all_images.append(Image.fromarray(imgs_np[j]))
            
            # Clear memory if requested
            if clear_memory:
                del z, z_batch, ws, img, imgs_np
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        return saved_paths if save_images else all_images
    
    def generate_single_image(self, seed=None, save=True, filename=None):
        """
        Generate a single image with minimal memory usage.
        
        Args:
            seed: Random seed (None for random)
            save: Whether to save the image
            filename: Custom filename (default: random_seed.png)
            
        Returns:
            PIL.Image object and path if saved
        """
        # Set seed
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        torch.manual_seed(seed)
        
        # Generate random latent
        z = torch.randn(1, self.G.z_dim, device=self.device)
        
        # Generate image
        with torch.no_grad():
            ws = self.G.mapping(z, None, truncation_psi=self.truncation_psi)
            img = self.G.synthesis(ws, noise_mode=self.noise_mode)
            
            # Resize if needed
            if self.resolution is not None and self.resolution != self.G.img_resolution:
                img = torch.nn.functional.interpolate(
                    img, 
                    size=(self.resolution, self.resolution),
                    mode='bilinear',
                    align_corners=False
                )
        
        # Convert to PIL image
        img_np = (img[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(img_np)
        
        # Save if requested
        path = None
        if save:
            if filename is None:
                filename = f"seed_{seed}.png"
            path = os.path.join(self.output_dir, filename)
            pil_img.save(path)
            print(f"Saved image to {path}")
        
        # Clear memory
        del z, ws, img, img_np
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return pil_img, path, seed


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient StyleGAN3 image generator")
    parser.add_argument("--generator", type=str, required=True, help="Path to StyleGAN3 generator pickle")
    parser.add_argument("--output", type=str, default="./generated_images", help="Output directory for generated images")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (use 1 for lowest memory usage)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds to use")
    parser.add_argument("--resolution", type=int, default=None, help="Output resolution (default: same as generator)")
    parser.add_argument("--truncation_psi", type=float, default=0.7, help="Truncation psi parameter")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--single", action="store_true", help="Generate a single image with minimal memory usage")
    
    args = parser.parse_args()
    
    # Create generator
    generator = MemoryEfficientGenerator(
        generator_pkl=args.generator,
        output_dir=args.output,
        resolution=args.resolution,
        device=args.device,
        truncation_psi=args.truncation_psi,
    )
    
    if args.single:
        # Generate a single image
        seed = np.random.randint(0, 2**32 - 1) if args.seeds is None else int(args.seeds.split(',')[0])
        generator.generate_single_image(seed=seed)
    elif args.seeds is not None:
        # Generate from seeds
        seeds = [int(s) for s in args.seeds.split(',')]
        paths = generator.generate_from_seeds(
            seeds=seeds,
            batch_size=args.batch_size,
        )
        print(f"Generated {len(paths)} images from provided seeds")
    else:
        # Generate multiple images
        paths = generator.generate_images(
            num_images=args.num_images,
            batch_size=args.batch_size,
        )
        print(f"Generated {len(paths)} images")


if __name__ == "__main__":
    main()