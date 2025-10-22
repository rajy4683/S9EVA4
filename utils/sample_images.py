"""
Image Sampling Utility
Select random samples from the saved dataset given a list of classes.
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Union
import argparse
from PIL import Image
import matplotlib.pyplot as plt


class ImageSampler:
    """Sample random images from dataset."""

    def __init__(self, data_dir: str, split: str = "train"):
        """
        Initialize sampler.

        Args:
            data_dir: Root data directory
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split

        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        # Get available classes
        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        print(f"Found {len(self.classes)} classes in {split} split")

    def get_class_images(self, class_name: str) -> List[Path]:
        """
        Get all images for a class.

        Args:
            class_name: Name of the class

        Returns:
            List of image paths
        """
        class_dir = self.split_dir / class_name
        if not class_dir.exists():
            return []

        # Get all image files
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
            images.extend(class_dir.glob(ext))

        return sorted(images)

    def sample_from_class(
        self,
        class_name: str,
        num_samples: int = 5,
        seed: Optional[int] = None
    ) -> List[Path]:
        """
        Sample random images from a class.

        Args:
            class_name: Name of the class
            num_samples: Number of samples to return
            seed: Random seed for reproducibility

        Returns:
            List of sampled image paths
        """
        if seed is not None:
            random.seed(seed)

        images = self.get_class_images(class_name)

        if not images:
            print(f"Warning: No images found for class '{class_name}'")
            return []

        # Sample
        num_samples = min(num_samples, len(images))
        sampled = random.sample(images, num_samples)

        return sampled

    def sample_from_classes(
        self,
        class_names: List[str],
        num_samples_per_class: int = 5,
        seed: Optional[int] = None
    ) -> dict:
        """
        Sample random images from multiple classes.

        Args:
            class_names: List of class names
            num_samples_per_class: Number of samples per class
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping class name to list of image paths
        """
        if seed is not None:
            random.seed(seed)

        results = {}
        for class_name in class_names:
            results[class_name] = self.sample_from_class(
                class_name, num_samples_per_class, seed=None  # Use global seed
            )

        return results

    def visualize_samples(
        self,
        class_names: Union[str, List[str]],
        num_samples: int = 5,
        figsize: tuple = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize random samples from classes.

        Args:
            class_names: Single class name or list of class names
            num_samples: Number of samples per class
            figsize: Figure size
            save_path: Path to save visualization (optional)
        """
        if isinstance(class_names, str):
            class_names = [class_names]

        # Sample images
        samples = self.sample_from_classes(class_names, num_samples)

        # Create visualization
        num_classes = len(class_names)
        fig, axes = plt.subplots(num_classes, num_samples, figsize=figsize)

        if num_classes == 1:
            axes = axes.reshape(1, -1)

        for i, class_name in enumerate(class_names):
            images = samples[class_name]

            for j in range(num_samples):
                ax = axes[i, j] if num_classes > 1 else axes[0, j]

                if j < len(images):
                    # Load and display image
                    img = Image.open(images[j])
                    ax.imshow(img)
                    ax.axis('off')

                    # Add title to first column
                    if j == 0:
                        ax.set_title(f'{class_name}\n{len(self.get_class_images(class_name))} images',
                                   fontsize=10, loc='left')
                else:
                    ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def copy_samples_to_directory(
        self,
        class_names: List[str],
        num_samples_per_class: int,
        output_dir: str,
        seed: Optional[int] = None
    ):
        """
        Copy sampled images to a new directory.

        Args:
            class_names: List of class names
            num_samples_per_class: Number of samples per class
            output_dir: Output directory
            seed: Random seed
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = self.sample_from_classes(class_names, num_samples_per_class, seed)

        total_copied = 0
        for class_name, images in samples.items():
            class_output = output_path / class_name
            class_output.mkdir(exist_ok=True)

            for img_path in images:
                dest = class_output / img_path.name
                shutil.copy2(img_path, dest)
                total_copied += 1

        print(f"Copied {total_copied} images to {output_dir}")

    def get_dataset_statistics(self) -> dict:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_classes': len(self.classes),
            'total_images': 0,
            'images_per_class': {},
            'min_images': float('inf'),
            'max_images': 0
        }

        for class_name in self.classes:
            images = self.get_class_images(class_name)
            num_images = len(images)

            stats['images_per_class'][class_name] = num_images
            stats['total_images'] += num_images
            stats['min_images'] = min(stats['min_images'], num_images)
            stats['max_images'] = max(stats['max_images'], num_images)

        stats['avg_images'] = stats['total_images'] / stats['num_classes']

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Sample random images from dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root data directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to sample from"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs='+',
        help="List of class names to sample from (default: all)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples per class"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize sampled images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Copy sampled images to this directory"
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        help="Path to save visualization plot"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Create sampler
    sampler = ImageSampler(args.data_dir, args.split)

    # Show statistics
    if args.stats:
        stats = sampler.get_dataset_statistics()
        print(f"\nDataset Statistics ({args.split}):")
        print(f"  Total classes: {stats['num_classes']}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Images per class: min={stats['min_images']}, "
              f"max={stats['max_images']}, avg={stats['avg_images']:.1f}")

    # Select classes
    if args.classes:
        class_names = args.classes
    else:
        # Sample random classes
        class_names = random.sample(sampler.classes, min(5, len(sampler.classes)))

    print(f"\nSampling {args.num_samples} images from classes: {class_names}")

    # Visualize
    if args.visualize or args.save_plot:
        sampler.visualize_samples(
            class_names,
            num_samples=args.num_samples,
            save_path=args.save_plot
        )

    # Copy to output directory
    if args.output_dir:
        sampler.copy_samples_to_directory(
            class_names,
            args.num_samples,
            args.output_dir,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
