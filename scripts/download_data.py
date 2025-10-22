"""
Data Extraction Script for ImageNet-1k Subset
Downloads and organizes 100 classes from HuggingFace ILSVRC/imagenet-1k dataset.
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import random


# Default curated list of 100 class labels covering diverse categories
DEFAULT_CLASS_LABELS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,           # Animals (mammals)
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55, # More animals
    100, 105, 110, 115, 120, 125, 130, 135, # Birds
    200, 205, 210, 215, 220, 225, 230, 235, # Aquatic animals
    300, 305, 310, 315, 320, 325, 330, 335, # Reptiles/amphibians
    400, 405, 410, 415, 420, 425, 430, 435, # Insects
    500, 505, 510, 515, 520, 525, 530, 535, # Vehicles
    600, 605, 610, 615, 620, 625, 630, 635, # Furniture
    700, 705, 710, 715, 720, 725, 730, 735, # Instruments
    800, 805, 810, 815, 820, 825, 830, 835, # Tools/equipment
    900, 905, 910, 915, 920, 925, 930, 935, # Food
    940, 945, 950, 955, 960, 965, 970, 975  # Misc objects
]


class ImageNetExtractor:
    """Extract and save a subset of ImageNet classes from HuggingFace."""

    def __init__(
        self,
        output_dir: str = "./imagenet_subset",
        num_classes: int = 100,
        seed: int = 42,
        class_list: Optional[List[int]] = None
    ):
        """
        Initialize the extractor.

        Args:
            output_dir: Directory to save extracted images
            num_classes: Number of classes to extract (default: 100)
            seed: Random seed for reproducibility
            class_list: Optional list of specific class IDs to extract.
                       If provided, overrides num_classes and random selection.
        """
        self.output_dir = Path(output_dir)
        self.num_classes = num_classes
        self.seed = seed
        self.class_list = class_list
        random.seed(seed)

        # Create directory structure
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

    def get_class_mapping(self, dataset) -> dict:
        """
        Get mapping of label IDs to class names.

        Args:
            dataset: HuggingFace dataset

        Returns:
            Dictionary mapping label ID to class name
        """
        # Get features to extract label information
        features = dataset.features
        class_names = features['label'].names
        return {i: name for i, name in enumerate(class_names)}

    def select_classes(self, total_classes: int = 1000) -> List[int]:
        """
        Select classes to extract. Uses predefined class_list if provided,
        otherwise randomly selects classes.

        Args:
            total_classes: Total number of classes in ImageNet (1000)

        Returns:
            List of selected class IDs
        """
        if self.class_list is not None:
            # Use provided class list
            selected = sorted(self.class_list)
            print(f"Using predefined class list with {len(selected)} classes")
            print(f"Classes: {selected[:10]}... (showing first 10)")
            return selected
        else:
            # Random selection
            selected = sorted(random.sample(range(total_classes), self.num_classes))
            print(f"Randomly selected {self.num_classes} classes: {selected[:10]}... (showing first 10)")
            return selected

    def save_image(self, image: Image.Image, save_path: Path) -> bool:
        """
        Save image to disk.

        Args:
            image: PIL Image
            save_path: Path to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(save_path, 'JPEG', quality=95)
            return True
        except Exception as e:
            print(f"Error saving image to {save_path}: {e}")
            return False

    def extract_split(
        self,
        dataset,
        split_name: str,
        selected_classes: List[int],
        class_mapping: dict,
        max_images_per_class: Optional[int] = None
    ):
        """
        Extract images for a specific split (train/val).

        Args:
            dataset: HuggingFace dataset split
            split_name: Name of the split ('train' or 'validation')
            selected_classes: List of class IDs to extract
            class_mapping: Mapping from label ID to class name
            max_images_per_class: Maximum images to save per class (None = all)
        """
        output_dir = self.train_dir if split_name == 'train' else self.val_dir

        # Track counts per class
        class_counts = {cls_id: 0 for cls_id in selected_classes}
        selected_set = set(selected_classes)

        print(f"\nExtracting {split_name} split...")

        # Iterate through dataset
        total_saved = 0
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            label = sample['label']

            # Skip if not in selected classes
            if label not in selected_set:
                continue

            # Check if we've reached max images for this class
            if max_images_per_class and class_counts[label] >= max_images_per_class:
                # Check if all classes have reached max
                if all(cnt >= max_images_per_class for cnt in class_counts.values()):
                    break
                continue

            # Get image and class name
            image = sample['image']
            class_name = class_mapping[label]

            # Create class directory
            class_dir = output_dir / class_name

            # Save image with unique filename
            img_filename = f"{class_name}_{class_counts[label]:05d}.jpg"
            img_path = class_dir / img_filename

            if self.save_image(image, img_path):
                class_counts[label] += 1
                total_saved += 1

        print(f"Saved {total_saved} images for {split_name} split")
        print(f"Images per class: min={min(class_counts.values())}, "
              f"max={max(class_counts.values())}, "
              f"avg={sum(class_counts.values())/len(class_counts):.1f}")

    def extract(
        self,
        max_train_per_class: Optional[int] = None,
        max_val_per_class: Optional[int] = None
    ):
        """
        Extract the complete dataset.

        Args:
            max_train_per_class: Max images per class for training (None = all)
            max_val_per_class: Max images per class for validation (None = all)
        """
        print("Loading ImageNet-1k dataset from HuggingFace (streaming mode)...")
        print("Note: This may take some time for the first download.\n")

        # Load dataset in streaming mode
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            streaming=True,
            trust_remote_code=True
        )

        # Get non-streaming version of a small sample to extract class info
        print("Loading class information...")
        dataset_sample = load_dataset(
            "ILSVRC/imagenet-1k",
            split="train",
            streaming=False,
            trust_remote_code=True
        )

        # Get class mapping
        class_mapping = self.get_class_mapping(dataset_sample)

        # Select random classes
        selected_classes = self.select_classes(len(class_mapping))

        # Save class mapping
        mapping_file = self.output_dir / "class_mapping.txt"
        with open(mapping_file, 'w') as f:
            f.write("# Selected Classes (Label ID -> Class Name)\n")
            for cls_id in selected_classes:
                f.write(f"{cls_id}: {class_mapping[cls_id]}\n")
        print(f"Saved class mapping to {mapping_file}")

        # Extract train split
        self.extract_split(
            dataset['train'],
            'train',
            selected_classes,
            class_mapping,
            max_train_per_class
        )

        # Extract validation split
        self.extract_split(
            dataset['validation'],
            'validation',
            selected_classes,
            class_mapping,
            max_val_per_class
        )

        print(f"\n{'='*60}")
        print(f"Dataset extraction complete!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Train images: {self.train_dir}")
        print(f"Val images: {self.val_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract subset of ImageNet-1k from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./imagenet_subset",
        help="Output directory for extracted images"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=100,
        help="Number of classes to extract (default: 100, ignored if --use_default_classes is set)"
    )
    parser.add_argument(
        "--max_train_per_class",
        type=int,
        default=None,
        help="Maximum training images per class (default: all)"
    )
    parser.add_argument(
        "--max_val_per_class",
        type=int,
        default=None,
        help="Maximum validation images per class (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for class selection (default: 42, ignored if --use_default_classes is set)"
    )
    parser.add_argument(
        "--use_default_classes",
        action="store_true",
        help="Use the predefined curated list of 100 classes covering diverse categories "
             "(animals, birds, vehicles, furniture, etc.) instead of random selection"
    )
    parser.add_argument(
        "--class_list",
        type=int,
        nargs='+',
        default=None,
        help="Custom list of class IDs to extract (space-separated, e.g., --class_list 0 1 2 3)"
    )

    args = parser.parse_args()

    # Determine which class list to use
    class_list = None
    if args.use_default_classes:
        class_list = DEFAULT_CLASS_LABELS
        print("Using default curated class list (100 classes covering diverse categories)")
    elif args.class_list is not None:
        class_list = args.class_list
        print(f"Using custom class list with {len(class_list)} classes")

    # Create extractor and run
    extractor = ImageNetExtractor(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        seed=args.seed,
        class_list=class_list
    )

    extractor.extract(
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class
    )


if __name__ == "__main__":
    main()
