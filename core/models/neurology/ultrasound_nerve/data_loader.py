import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NerveUltrasoundDataset(Dataset):
    """
    Patient-Aware Dataset for Nerve Detection (Brachial Plexus Ultrasound).
    Classes: 0 (No Nerve / Negative), 1 (Nerve Present / Positive).
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

def get_nerve_loaders(
    splits_dir: str = 'dataset/splits/neurology/ultrasound_nerve',
    images_dir: str = 'dataset/processed/neurology/ultrasound_nerve/images',
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4
):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
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

    train_ds = NerveUltrasoundDataset(train_csv, images_dir, transform=train_transform)
    val_ds = NerveUltrasoundDataset(val_csv, images_dir, transform=eval_transform)
    test_ds = NerveUltrasoundDataset(test_csv, images_dir, transform=eval_transform)

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
