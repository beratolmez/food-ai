"""PyTorch Dataset for classification with data augmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FoodClassifierDataset(Dataset):
    """Dataset for Food Classification loading images from class-named folders."""

    def __init__(
        self,
        root_dir: Path,
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ):
        """
        Args:
            root_dir: Directory containing class subfolders (e.g. data/splits/train)
            class_to_idx: Dictionary mapping class name to integer index
            transform: Optional torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []
        
        # Gather all image paths
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists() or not class_dir.is_dir():
                continue
                
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        
        try:
            # OpenCV could be used, but PIL is standard for Torchvision transforms
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Could not read image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_train: bool, image_size: int = 224) -> transforms.Compose:
    """Get train/val transforms for EfficientNet/MobileNet.
    
    Standard ImageNet normalization is used:
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # e.g. 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
