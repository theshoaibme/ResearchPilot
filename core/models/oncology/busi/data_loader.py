import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BusiDataset(Dataset):
    """
    PyTorch Dataset reading from BUSI Breast Ultrasound CSV splits.
    3 classes: normal (0), benign (1), malignant (2).
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
        img_path = self.images_dir / row['filename']
        label = row['class_id']
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_busi_loaders(
    splits_dir: str = 'dataset/splits/oncology/busi',
    images_dir: str = 'dataset/processed/oncology/busi/images',
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 2
):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv = os.path.join(splits_dir, 'validation.csv')
    test_csv = os.path.join(splits_dir, 'test.csv')

    train_ds = BusiDataset(train_csv, images_dir, transform=train_transform)
    val_ds = BusiDataset(val_csv, images_dir, transform=eval_transform)
    test_ds = BusiDataset(test_csv, images_dir, transform=eval_transform)

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )

    return train_loader, val_loader, test_loader
