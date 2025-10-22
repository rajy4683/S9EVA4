"""
ImageNet Dataset Module
Custom dataset class for ImageNet subset with flexible transforms.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from data.transforms import (
    get_train_transforms,
    get_val_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class ImageNetSubset(ImageFolder):
    """
    ImageNet subset dataset extending ImageFolder.

    Directory structure should be:
    root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            img2.jpg
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable] = None
    ):
        """
        Args:
            root: Root directory path
            transform: Image transforms
            target_transform: Target transforms
            is_valid_file: Function to check if file is valid
        """
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )

        self.root = Path(root)
        print(f"Loaded dataset from {root}")
        print(f"Found {len(self.classes)} classes with {len(self.samples)} images")

    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.classes

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            index: Index

        Returns:
            Tuple of (image, label)
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    auto_augment: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        image_size: Image size (default: 224)
        auto_augment: Use AutoAugment (default: False)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = ImageNetSubset(
        root=train_dir,
        transform=get_train_transforms(image_size, auto_augment)
    )

    val_dataset = ImageNetSubset(
        root=val_dir,
        transform=get_val_transforms(image_size)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    print(f"\nData Loaders Created:")
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")

    return train_loader, val_loader


def verify_dataset(data_dir: str) -> dict:
    """
    Verify dataset structure and return statistics.

    Args:
        data_dir: Root directory containing train/val splits

    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"

    stats = {
        'train_exists': train_path.exists(),
        'val_exists': val_path.exists(),
        'train_classes': 0,
        'val_classes': 0,
        'train_images': 0,
        'val_images': 0
    }

    if train_path.exists():
        train_classes = [d for d in train_path.iterdir() if d.is_dir()]
        stats['train_classes'] = len(train_classes)
        stats['train_images'] = sum(
            len(list(c.glob('*.jpg'))) + len(list(c.glob('*.JPEG'))) + len(list(c.glob('*.png')))
            for c in train_classes
        )

    if val_path.exists():
        val_classes = [d for d in val_path.iterdir() if d.is_dir()]
        stats['val_classes'] = len(val_classes)
        stats['val_images'] = sum(
            len(list(c.glob('*.jpg'))) + len(list(c.glob('*.JPEG'))) + len(list(c.glob('*.png')))
            for c in val_classes
        )

    return stats


if __name__ == "__main__":
    # Test dataset loading
    import argparse

    parser = argparse.ArgumentParser(description="Test ImageNet dataset loading")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Verify dataset
    print("Verifying dataset...")
    stats = verify_dataset(args.data_dir)
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if stats['train_exists'] and stats['val_exists']:
        # Create data loaders
        train_loader, val_loader = get_data_loaders(
            train_dir=os.path.join(args.data_dir, "train"),
            val_dir=os.path.join(args.data_dir, "val"),
            batch_size=args.batch_size,
            num_workers=2
        )

        # Test loading a batch
        print("\nTesting batch loading...")
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print("\nDataset verification successful!")
    else:
        print("\nError: Train or val directory not found!")
