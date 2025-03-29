import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np

# Import our custom dataset classes
from stylegan3_hvae_full import ImageDataset

def save_tensor_as_image(tensor, filename):
    """Convert a tensor to a PIL Image and save it."""
    # Convert tensor to numpy array
    img = tensor.numpy()
    
    # Scale from [-1, 1] to [0, 255]
    img = ((img.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    
    # Create PIL Image and save
    Image.fromarray(img).save(filename)

def test_image_dataset(dataset_path, output_dir='./test_output', batch_size=4, num_workers=4, 
                     resolution=256, is_imagenet=False, max_images=10):
    """
    Test loading and processing images from a dataset.
    
    Args:
        dataset_path: Path to the image dataset
        output_dir: Directory to save output samples
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        resolution: Image resolution
        is_imagenet: Whether dataset has ImageNet folder structure
        max_images: Maximum number of images to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing dataset loading from {dataset_path}")
    
    # Create dataset
    dataset = ImageDataset(
        dataset_path,
        resolution=resolution,
        is_imagenet=is_imagenet
    )
    
    print(f"Found {len(dataset)} images")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Process a few batches
    print(f"Processing {min(max_images, len(dataset))} images")
    
    num_processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Save each image in the batch
        for i in range(batch.size(0)):
            if num_processed >= max_images:
                break
                
            # Save image
            save_tensor_as_image(
                batch[i].cpu(),
                os.path.join(output_dir, f'sample_{num_processed}.png')
            )
            
            num_processed += 1
        
        if num_processed >= max_images:
            break
    
    print(f"Saved {num_processed} images to {output_dir}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(dataset)}")
    
    # Check a few random images for dimensions and value range
    idx = torch.randint(0, len(dataset), (5,))
    for i in idx:
        img = dataset[i.item()]
        print(f"Image {i.item()} shape: {img.shape}, min: {img.min().item():.4f}, max: {img.max().item():.4f}")
    
    # Test image loading speed
    print("\nTesting loading speed...")
    
    # Time loading all images in a batch
    batch_size = min(100, len(dataset))
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    for batch in tqdm(loader):
        print(f"Loaded batch of {batch.size(0)} images, shape: {batch.shape}")
        break
    
    print("Image dataset test completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ImageNet dataset loading")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to image dataset")
    parser.add_argument("--output", type=str, default="./test_output",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--imagenet", action="store_true",
                        help="Treat dataset as ImageNet with class subdirectories")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--max_images", type=int, default=10,
                        help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    test_image_dataset(
        dataset_path=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
        resolution=args.resolution,
        is_imagenet=args.imagenet,
        max_images=args.max_images
    )