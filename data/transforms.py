"""
Data Transforms for ImageNet Training
Includes ImageNet normalization and medium augmentation.
"""

import torch
from torchvision import transforms
from typing import Tuple


# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    auto_augment: bool = False
) -> transforms.Compose:
    """
    Get training transforms with medium augmentation.

    Augmentation strategy:
    - RandomResizedCrop (224x224)
    - Random Horizontal Flip
    - ColorJitter (brightness, contrast, saturation)
    - RandomErasing
    - ImageNet normalization

    Args:
        image_size: Target image size (default: 224)
        auto_augment: Use AutoAugment policy (optional)

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    # Add AutoAugment if requested
    if auto_augment:
        transform_list.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        )
    else:
        # Medium augmentation with ColorJitter
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )
        )

    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation transforms (no augmentation).

    Args:
        image_size: Target image size (default: 224)

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms (same as validation).

    Args:
        image_size: Target image size (default: 224)

    Returns:
        Composed transforms
    """
    return get_val_transforms(image_size)


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Denormalize a tensor image.

    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


class Cutout:
    """
    Cutout augmentation - randomly mask out one or more patches.
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Args:
            n_holes: Number of patches to cut out
            length: Length (in pixels) of each square patch
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image of size (C, H, W)

        Returns:
            Image with n_holes of dimension length x length cut out
        """
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = torch.randint(h, (1,)).item()
            x = torch.randint(w, (1,)).item()

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.

        mask = mask.expand_as(img)
        return img * mask


def get_train_transforms_with_cutout(
    image_size: int = 224,
    cutout_holes: int = 1,
    cutout_length: int = 16
) -> transforms.Compose:
    """
    Get training transforms with Cutout augmentation.

    Args:
        image_size: Target image size
        cutout_holes: Number of cutout patches
        cutout_length: Size of each cutout patch

    Returns:
        Composed transforms with Cutout
    """
    base_transforms = get_train_transforms(image_size).transforms[:-1]  # Remove RandomErasing
    cutout = Cutout(n_holes=cutout_holes, length=cutout_length)

    return transforms.Compose(base_transforms + [cutout])
