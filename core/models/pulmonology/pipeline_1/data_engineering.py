import os
import json
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==============================================================================
# 1. MANIFEST GENERATION (DATA SPLITTING)
# ==============================================================================
def generate_manifest(raw_dir: str, metadata_dir: str):
    """
    Scans the raw dataset directory and creates a 'manifest.json'.
    
    Why do we do this?
    Instead of reading the folder structure every time we train, we generate 
    a static JSON file that holds the exact Train/Validation/Test split. 
    This ensures reproducibility (the split never changes randomly between runs).
    
    Args:
        raw_dir (str): Path to the raw dataset (e.g., 'dataset/raw/COVID19_XRay')
        metadata_dir (str): Path to save the manifest.
    """
    manifest_path = os.path.join(metadata_dir, 'manifest.json')
    raw_path = Path(raw_dir) / 'COVID-19_Radiography_Dataset'
    
    # Define our classes and their integer labels (0 to 3)
    classes = {'Normal': 0, 'COVID': 1, 'Lung_Opacity': 2, 'Viral Pneumonia': 3}
    samples = []
    
    # Set a random seed so the shuffle is exactly the same every time you run this
    random.seed(42) 
    
    for class_name, label in classes.items():
        class_dir = raw_path / class_name
        # Handle cases where images are inside an 'images' subfolder
        images_dir = class_dir / 'images' if (class_dir / 'images').exists() else class_dir
        
        # Grab all PNG and JPG files
        filepaths = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        
        # Shuffle the files randomly before we split them
        random.shuffle(filepaths)
        
        n_total = len(filepaths)
        # 80% for Training, 10% for Validation, 10% for Testing
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        # Assign each file to a split based on its index
        for i, filepath in enumerate(filepaths):
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
                
            samples.append({
                'filepath': str(filepath.resolve()),
                'label': label,
                'class_name': class_name,
                'split': split
            })
            
    manifest = {
        'dataset': 'COVID19_XRay',
        'total_samples': len(samples),
        'classes': classes,
        'samples': samples
    }
    
    # Save to JSON
    os.makedirs(metadata_dir, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"✅ Manifest generated at {manifest_path} with {len(samples)} total samples.")

# ==============================================================================
# 2. PYTORCH DATASET CLASS
# ==============================================================================
class MedicalDataset(Dataset):
    """
    A custom PyTorch Dataset class.
    
    How it works:
    PyTorch requires two main methods to be implemented:
    1. __len__: Returns the total number of items.
    2. __getitem__: Loads ONE item (image and label) at a specific index.
    
    This is extremely memory efficient! It only loads the image from the hard drive 
    into RAM when the model explicitly asks for it during training (batching).
    """
    def __init__(self, manifest_file: str, split: str = 'train', transform=None):
        self.manifest_file = manifest_file
        self.split = split
        self.transform = transform
        
        # Read the static JSON manifest
        with open(manifest_file, 'r') as f:
            self.data = json.load(f)
            
        # Filter the samples to only include the ones for this specific split
        self.samples = [s for s in self.data['samples'] if s.get('split', 'train') == split]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # 1. Get the file path and label for the requested index
        img_path = self.samples[idx]['filepath']
        label = self.samples[idx]['label']
        
        # 2. Open the image using PIL (Python Imaging Library). 
        # Convert to 'RGB' to ensure 3-channels, since DenseNet expects 3 channels.
        image = Image.open(img_path).convert('RGB')
        
        # 3. Apply transformations (Resizing, Normalization, Data Augmentation)
        if self.transform:
            image = self.transform(image)
            
        # 4. Return the image tensor and the label as a PyTorch long tensor
        return image, torch.tensor(label, dtype=torch.long)

# ==============================================================================
# 3. DATALOADER GENERATOR
# ==============================================================================
def get_dataloaders(manifest_file: str, batch_size: int = 32):
    """
    Creates the Train and Validation DataLoaders.
    
    DataLoaders act as the "Generator" mentioned in your architecture.
    They take the `MedicalDataset`, group the images into batches (e.g., 32 at a time),
    and send them to the GPU.
    """
    
    # --- Data Augmentation for Training ---
    # We alter the training images slightly to prevent the model from memorizing them (Overfitting).
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Standard size for DenseNet
        transforms.RandomHorizontalFlip(),   # Randomly flip the X-ray left-to-right
        transforms.RandomRotation(10),       # Randomly rotate by max 10 degrees
        transforms.ToTensor(),               # Convert PIL Image to PyTorch Tensor (scales pixels to 0.0 - 1.0)
        # Normalize uses ImageNet standard mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # --- Transformations for Validation ---
    # CRITICAL: We NEVER augment validation data. We only resize and normalize it.
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the datasets
    train_ds = MedicalDataset(manifest_file, split='train', transform=train_transform)
    val_ds = MedicalDataset(manifest_file, split='val', transform=val_transform)
    
    # Create the DataLoaders
    # shuffle=True for training ensures the model doesn't learn the order of the dataset
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
