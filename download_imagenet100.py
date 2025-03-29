#!/usr/bin/env python3
"""
Script to download and organize the ImageNet-100 dataset from Kaggle.
Uses the Kaggle API for downloading and then organizes the files in the correct format.

Requirements:
- kaggle package installed (pip install kaggle)
- Kaggle API credentials set up
"""

import os
import sys
import subprocess
import zipfile
import shutil
import argparse
import random
from tqdm import tqdm
import json

def check_kaggle_api():
    """Check if kaggle API is installed and credentials are set up."""
    try:
        import kaggle
        # Check credentials path
        kaggle_dir = os.path.expanduser('~/.kaggle')
        credentials_path = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(credentials_path):
            print("\n⚠️ Kaggle API credentials not found!")
            print("You need to set up your Kaggle credentials to download the dataset.")
            print("1. Go to https://www.kaggle.com/settings")
            print("2. Click on 'Create New API Token' to download kaggle.json")
            print(f"3. Place it in {kaggle_dir}/")
            print(f"4. Run: chmod 600 {credentials_path}")
            
            create_credentials = input("\nDo you want to set up credentials now? (y/n): ")
            if create_credentials.lower() == 'y':
                # Create directory if it doesn't exist
                if not os.path.exists(kaggle_dir):
                    os.makedirs(kaggle_dir)
                
                # Get credentials from user
                username = input("Enter your Kaggle username: ")
                key = input("Enter your Kaggle API key: ")
                
                # Create credentials file
                with open(credentials_path, 'w') as f:
                    json.dump({"username": username, "key": key}, f)
                
                # Set proper permissions
                os.chmod(credentials_path, 0o600)
                print(f"Credentials saved to {credentials_path}")
            else:
                return False
        
        return True
    except ImportError:
        print("\n⚠️ Kaggle API not installed!")
        print("Please install it with: pip install kaggle")
        
        install = input("\nDo you want to install it now? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"])
            return check_kaggle_api()  # Check again after installation
        
        return False

def download_imagenet100(output_dir, force_download=False):
    """Download the ImageNet-100 dataset from Kaggle."""
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    # Check if dataset is already downloaded
    if not force_download and os.path.exists(train_dir) and os.path.exists(val_dir):
        if os.listdir(train_dir) and os.listdir(val_dir):
            print(f"Dataset already exists in {output_dir}")
            redownload = input("Do you want to re-download it? (y/n): ")
            if redownload.lower() != 'y':
                return True
    
    # Create temp directory for download
    download_dir = os.path.join(output_dir, "download_temp")
    os.makedirs(download_dir, exist_ok=True)
    
    # Download dataset using Kaggle API
    try:
        print(f"\n📥 Downloading ImageNet-100 dataset...")
        subprocess.run(
            ["kaggle", "datasets", "download", "ambityga/imagenet100", "--path", download_dir],
            check=True
        )
        print("✅ Download completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        return False
    
    # Extract dataset
    zip_path = os.path.join(download_dir, "imagenet100.zip")
    if not os.path.exists(zip_path):
        print(f"❌ Downloaded file not found: {zip_path}")
        return False
    
    # Create extraction directory
    extract_dir = os.path.join(output_dir, "extract_temp")
    os.makedirs(extract_dir, exist_ok=True)
    
    print("\n📂 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # First, check the zip structure
        all_entries = [name for name in zip_ref.namelist()]
        print(f"Zip structure preview (first 5 entries):")
        for entry in all_entries[:5]:
            print(f"  {entry}")
        if len(all_entries) > 5:
            print(f"  ... and {len(all_entries) - 5} more entries")
        
        # Extract all files
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(member, extract_dir)
    
    # Check the extracted structure
    print("\nAnalyzing extracted dataset structure...")
    root_items = os.listdir(extract_dir)
    print(f"Root items in the extracted directory: {root_items}")
    
    # Prepare the output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Process the extracted files based on their structure
    if "train" in root_items and "val" in root_items:
        # Standard ImageNet structure with train/val folders
        print("Found standard ImageNet structure with train/val folders")
        organize_standard_structure(extract_dir, train_dir, val_dir)
    else:
        # Try to detect if it's a single folder with class folders
        found_classes = False
        for item in root_items:
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path):
                subdir_items = os.listdir(item_path)
                if "train" in subdir_items and "val" in subdir_items:
                    # It's a container folder with train/val structure
                    print(f"Found train/val structure inside {item}")
                    organize_standard_structure(item_path, train_dir, val_dir)
                    found_classes = True
                    break
                
                # Check if it has image files (potential class folder)
                has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in subdir_items)
                if has_images:
                    found_classes = True
        
        if not found_classes:
            # If no clear structure, check for class directories at the root level
            class_dirs = [d for d in root_items if os.path.isdir(os.path.join(extract_dir, d))]
            if class_dirs:
                print(f"Found {len(class_dirs)} potential class directories, creating train/val split")
                create_train_val_split(extract_dir, train_dir, val_dir)
            else:
                print("❌ Couldn't determine dataset structure")
                return False
    
    # Clean up temporary files
    print("\n🧹 Cleaning up temporary files...")
    shutil.rmtree(download_dir)
    shutil.rmtree(extract_dir)
    
    # Verify the organized dataset
    train_classes = len(os.listdir(train_dir))
    val_classes = len(os.listdir(val_dir))
    
    print(f"\n✅ Dataset organized with {train_classes} training classes and {val_classes} validation classes")
    return True

def organize_standard_structure(source_dir, train_dir, val_dir):
    """Copy files from a standard train/val structure."""
    src_train_dir = os.path.join(source_dir, "train")
    src_val_dir = os.path.join(source_dir, "val")
    
    # Copy train classes
    print("Copying training classes...")
    for class_name in tqdm(os.listdir(src_train_dir)):
        src_class_dir = os.path.join(src_train_dir, class_name)
        if os.path.isdir(src_class_dir):
            dst_class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(dst_class_dir):
                shutil.rmtree(dst_class_dir)
            shutil.copytree(src_class_dir, dst_class_dir)
    
    # Copy validation classes
    print("Copying validation classes...")
    for class_name in tqdm(os.listdir(src_val_dir)):
        src_class_dir = os.path.join(src_val_dir, class_name)
        if os.path.isdir(src_class_dir):
            dst_class_dir = os.path.join(val_dir, class_name)
            if os.path.exists(dst_class_dir):
                shutil.rmtree(dst_class_dir)
            shutil.copytree(src_class_dir, dst_class_dir)

def create_train_val_split(source_dir, train_dir, val_dir, split=0.8):
    """Create train/val split from a directory of class folders."""
    # Find all class directories
    class_dirs = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if it has image files
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                           for f in os.listdir(item_path) 
                           if os.path.isfile(os.path.join(item_path, f)))
            if has_images:
                class_dirs.append(item)
    
    print(f"Creating train/val split for {len(class_dirs)} classes...")
    
    # Process each class
    for class_name in tqdm(class_dirs):
        src_class_dir = os.path.join(source_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        # Create class directories
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(src_class_dir)
                     if os.path.isfile(os.path.join(src_class_dir, f)) and
                     f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * split)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files
        for img in train_files:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(train_class_dir, img))
        
        for img in val_files:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(val_class_dir, img))

def count_images(directory):
    """Count the total number of images in a directory (including subdirectories)."""
    count = 0
    for root, _, files in os.walk(directory):
        count += sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    return count

def main():
    parser = argparse.ArgumentParser(description="Download and organize the ImageNet-100 dataset")
    parser.add_argument("--output", type=str, default="./datasets/imagenet100",
                       help="Output directory for the dataset")
    parser.add_argument("--force", action="store_true",
                       help="Force download even if dataset exists")
    
    args = parser.parse_args()
    
    print("\n🌐 ImageNet-100 Dataset Downloader 🌐")
    print("=====================================")
    
    # Check for Kaggle API
    if not check_kaggle_api():
        sys.exit(1)
    
    # Download and organize dataset
    if download_imagenet100(args.output, args.force):
        # Print dataset statistics
        train_dir = os.path.join(args.output, "train")
        val_dir = os.path.join(args.output, "val")
        
        train_images = count_images(train_dir)
        val_images = count_images(val_dir)
        train_classes = len(os.listdir(train_dir))
        val_classes = len(os.listdir(val_dir))
        
        print("\n📊 Dataset Statistics:")
        print(f"   - Training classes: {train_classes}")
        print(f"   - Training images: {train_images}")
        print(f"   - Validation classes: {val_classes}")
        print(f"   - Validation images: {val_images}")
        print(f"   - Total images: {train_images + val_images}")
        
        print("\n🎉 Success! The ImageNet-100 dataset is ready for StyleGAN3-HVAE training.")
        print("\nExample command for training:")
        print(f"python stylegan3_hvae_full.py --generator models/stylegan3-t-ffhq-1024x1024.pkl \\")
        print(f"  --output ./imagenet_output --resolution 256 --batch_size 4 \\")
        print(f"  --dataset {args.output}/train --val_dataset {args.output}/val --imagenet")
    else:
        print("\n❌ Failed to download or organize the dataset.")

if __name__ == "__main__":
    main()