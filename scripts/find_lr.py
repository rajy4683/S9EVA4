"""
LR Finder Script
Find optimal learning rate for training.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import SGD

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet50 import resnet50
from data.dataset import get_data_loaders
from utils.lr_finder import LRFinder
from training.config import LRFinderConfig


def main():
    parser = argparse.ArgumentParser(description="Find optimal learning rate")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root directory containing train/val splits")
    parser.add_argument("--num_classes", type=int, default=100,
                       help="Number of classes")

    # LR finder arguments
    parser.add_argument("--start_lr", type=float, default=1e-7,
                       help="Starting learning rate")
    parser.add_argument("--end_lr", type=float, default=10.0,
                       help="Ending learning rate")
    parser.add_argument("--num_iter", type=int, default=100,
                       help="Number of iterations")
    parser.add_argument("--step_mode", type=str, default="exp",
                       choices=["exp", "linear"],
                       help="LR step mode")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")

    # Optimizer settings
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")

    # Output settings
    parser.add_argument("--plot_path", type=str, default="./lr_finder_plot.png",
                       help="Path to save plot")
    parser.add_argument("--suggested_lr_path", type=str, default="./suggested_lr.txt",
                       help="Path to save suggested LR")

    # Device
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("Learning Rate Finder")
    print("="*70)
    print(f"Device: {device}")
    print(f"LR Range: {args.start_lr:.2e} -> {args.end_lr:.2e}")
    print(f"Iterations: {args.num_iter}")
    print(f"Step Mode: {args.step_mode}")
    print("="*70)

    # Load data
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    print("\nLoading data...")
    train_loader, _ = get_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    print("Creating model...")
    model = resnet50(num_classes=args.num_classes)
    model.to(device)

    print(f"Model created with {model.get_num_params():,} parameters")

    # Create optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.start_lr,  # Initial LR (will be changed by LR finder)
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create LR finder
    print("\nInitializing LR Finder...")
    lr_finder = LRFinder(model, optimizer, criterion, device)

    # Run range test
    print("\n" + "="*70)
    print("Running LR Range Test")
    print("="*70)

    lr_finder.range_test(
        train_loader=train_loader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        step_mode=args.step_mode
    )

    # Plot results
    print("\nGenerating plot...")
    lr_finder.plot(save_path=args.plot_path)

    # Get suggested LR
    initial_lr, max_lr = lr_finder.get_lr_range()

    # Save suggested LR to file
    with open(args.suggested_lr_path, 'w') as f:
        f.write(f"Suggested Learning Rate Range:\n")
        f.write(f"initial_lr: {initial_lr:.6f}\n")
        f.write(f"max_lr: {max_lr:.6f}\n")
        f.write(f"\nUsage in training:\n")
        f.write(f"python scripts/train.py --data_dir {args.data_dir} \\\n")
        f.write(f"    --initial_lr {initial_lr:.6f} \\\n")
        f.write(f"    --max_lr {max_lr:.6f}\n")

    print(f"\nSuggested LR saved to: {args.suggested_lr_path}")

    print("\n" + "="*70)
    print("LR Finder Complete!")
    print("="*70)
    print(f"Plot saved to: {args.plot_path}")
    print(f"\nRecommended LR for training:")
    print(f"  --initial_lr {initial_lr:.6f}")
    print(f"  --max_lr {max_lr:.6f}")
    print("="*70)


if __name__ == "__main__":
    main()
