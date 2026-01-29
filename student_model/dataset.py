import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from collections import Counter

from config import Config


class DefectDataset(Dataset):
    """Industrial defect detection dataset"""
    
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.samples = []
        self.classes = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                self.classes.append(class_dir.name)
        
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = [cls_name for cls_name in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])]
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, class_idx))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class"""
        labels = [label for _, label in self.samples]
        label_counts = Counter(labels)
        
        distribution = {}
        for class_name, class_idx in self.class_to_idx.items():
            distribution[class_name] = label_counts.get(class_idx, 0)
        
        return distribution


def get_transforms(phase: str = 'train') -> transforms.Compose:
    """Get data transforms for train/val phases"""
    
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.RandomRotation(
                degrees=Config.AUG_ROTATION,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5 if Config.AUG_HORIZONTAL_FLIP else 0.0),
            transforms.RandomVerticalFlip(p=0.5 if Config.AUG_VERTICAL_FLIP else 0.0),
            transforms.ColorJitter(
                brightness=Config.AUG_BRIGHTNESS,
                contrast=Config.AUG_CONTRAST,
                saturation=Config.AUG_SATURATION,
                hue=Config.AUG_HUE
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.NORMALIZE_MEAN,
                std=Config.NORMALIZE_STD
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.NORMALIZE_MEAN,
                std=Config.NORMALIZE_STD
            ),
        ])
    
    return transform


def create_dataloaders(
    train_dir: Optional[Path] = None,
    val_dir: Optional[Path] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create train and validation dataloaders"""
    
    train_dir = train_dir or Config.TRAIN_DIR
    val_dir = val_dir or Config.VAL_DIR
    
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    if val_dir.exists():
        train_dataset = DefectDataset(root_dir=train_dir, transform=train_transform)
        val_dataset = DefectDataset(root_dir=val_dir, transform=val_transform, class_to_idx=train_dataset.class_to_idx)
    else:
        full_dataset = DefectDataset(root_dir=train_dir, transform=None)
        
        if Config.STRATIFY:
            from sklearn.model_selection import StratifiedShuffleSplit
            
            labels = [label for _, label in full_dataset.samples]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=Config.SEED)
            train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
            
            train_samples = [full_dataset.samples[i] for i in train_idx]
            val_samples = [full_dataset.samples[i] for i in val_idx]
            
            train_dataset = DefectDataset.__new__(DefectDataset)
            train_dataset.root_dir = full_dataset.root_dir
            train_dataset.transform = train_transform
            train_dataset.classes = full_dataset.classes
            train_dataset.class_to_idx = full_dataset.class_to_idx
            train_dataset.samples = train_samples
            
            val_dataset = DefectDataset.__new__(DefectDataset)
            val_dataset.root_dir = full_dataset.root_dir
            val_dataset.transform = val_transform
            val_dataset.classes = full_dataset.classes
            val_dataset.class_to_idx = full_dataset.class_to_idx
            val_dataset.samples = val_samples
        else:
            train_size = int((1 - val_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(Config.SEED)
            )
            
            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx