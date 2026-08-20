"""
Create metadata files for custom dataset
Generate training.txt, validation.txt, test.txt
"""
import os
from pathlib import Path
import random

# Configuration
DATA_DIR = '../mydata'
IMG_DIR = os.path.join(DATA_DIR, 'img')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')

# Split ratios (train:val:test)
TRAIN_RATIO = 0.7    # 70%
VAL_RATIO = 0.15     # 15%
TEST_RATIO = 0.15    # 15%

# Random seed (ensure reproducibility)
RANDOM_SEED = 42

def create_metadata():
    """Create metadata folder and split lists"""
    
    # Create metadata folder
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    # Get all image filenames (excluding path)
    if not os.path.exists(IMG_DIR):
        print(f"Error: {IMG_DIR} does not exist")
        return
    
    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    print(f"Found {len(image_files)} image files")
    
    # Shuffle file list randomly
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    
    # Calculate split counts
    total = len(image_files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count  # Ensure total count is correct
    
    # Split data
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Write to file
    def write_list(filename, file_list):
        filepath = os.path.join(METADATA_DIR, filename)
        with open(filepath, 'w') as f:
            for fname in file_list:
                f.write(fname + '\n')
        print(f"✓ Created {filename}: {len(file_list)} files")
    
    write_list('training.txt', train_files)
    write_list('validation.txt', val_files)
    write_list('test.txt', test_files)
    
    print(f"\nSplit statistics:")
    print(f"  Training set (70%): {train_count} files")
    print(f"  Validation set (15%): {val_count} files")
    print(f"  Test set (15%): {test_count} files")
    print(f"  Total: {total} files")
    print(f"\nMetadata saved in: {METADATA_DIR}")

if __name__ == '__main__':
    create_metadata()
