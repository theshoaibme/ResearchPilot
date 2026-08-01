import os
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==============================================================================
# 1. PYTORCH MEDICAL DATASET CLASS
# ==============================================================================
class MedicalDataset(Dataset):
    """
    Production-Quality PyTorch Dataset for Version 2 Pipeline.
    """
    def __init__(self, split_csv_path: str, images_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        if not os.path.exists(split_csv_path):
            raise FileNotFoundError(f"Split CSV not found at: {split_csv_path}")
            
        self.data = pd.read_csv(split_csv_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        label = row['class_id']
        
        img_path = self.images_dir / filename
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# ==============================================================================
# 2. ENHANCED DATALOADER GENERATOR (VERSION 2 - MEDICALLY SAFE AUGMENTATION)
# ==============================================================================
def get_dataloaders(splits_dir: str, images_dir: str, batch_size: int = 32):
    """
    Creates Train and Validation DataLoaders with Medically-Safe Augmentation Pipeline.
    """
    
    # --- Medically Safe Data Augmentation for Training (Version 2) ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- Transformations for Validation ---
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv = os.path.join(splits_dir, 'validation.csv')
    
    train_ds = MedicalDataset(train_csv, images_dir, transform=train_transform)
    val_ds = MedicalDataset(val_csv, images_dir, transform=val_transform)
    
    # Calculate class weights for Focal Loss / Weighted Cross Entropy
    train_df = pd.read_csv(train_csv)
    class_counts = train_df['class_id'].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts], dtype=torch.float)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, class_weights
