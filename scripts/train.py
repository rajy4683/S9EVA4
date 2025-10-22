"""
Main Training Script
Train ResNet-50 on ImageNet subset.
"""

import os
import sys
import argparse
import torch
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet50 import resnet50
from data.dataset import get_data_loaders
from training.config import TrainingConfig
from training.trainer import ResNetTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ImageNet subset")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root directory containing train/val splits")
    parser.add_argument("--num_classes", type=int, default=100,
                       help="Number of classes")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=90,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")

    # Optimizer arguments
    parser.add_argument("--initial_lr", type=float, default=0.07047,
                       help="Initial learning rate")
    parser.add_argument("--max_lr", type=float, default=0.7047,
                       help="Maximum learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--no_nesterov", action="store_true",
                       help="Disable Nesterov momentum")

    # Scheduler arguments
    parser.add_argument("--div_factor", type=float, default=25.0,
                       help="OneCycleLR div_factor")
    parser.add_argument("--pct_start", type=float, default=0.3,
                       help="OneCycleLR pct_start")

    # Loss arguments
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing factor")

    # Model arguments
    parser.add_argument("--no_zero_init", action="store_true",
                       help="Disable zero-init residual")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Log directory")
    parser.add_argument("--save_freq", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no_amp", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--auto_augment", action="store_true",
                       help="Use AutoAugment")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create configuration
    config = TrainingConfig(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        initial_lr=args.initial_lr,
        max_lr=args.max_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=not args.no_nesterov,
        div_factor=args.div_factor,
        pct_start=args.pct_start,
        label_smoothing=args.label_smoothing,
        zero_init_residual=not args.no_zero_init,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_freq=args.save_freq,
        use_amp=not args.no_amp,
        auto_augment=args.auto_augment,
        seed=args.seed,
        resume_from=args.resume
    )

    if args.device:
        config.device = args.device

    print("="*70)
    print("ResNet-50 Training on ImageNet Subset")
    print("="*70)
    print(config)

    # Create data loaders
    train_dir = os.path.join(config.data_dir, config.train_dir)
    val_dir = os.path.join(config.data_dir, config.val_dir)

    print("\nLoading data...")
    train_loader, val_loader = get_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        auto_augment=config.auto_augment
    )

    # Create model
    print("\nCreating model...")
    model = resnet50(
        num_classes=config.num_classes,
        zero_init_residual=config.zero_init_residual
    )

    print(f"Model created with {model.get_num_params():,} parameters")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Resume from checkpoint if specified
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        trainer.resume_from_checkpoint(config.resume_from)

    # Start training
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    try:
        best_info = trainer.train()

        print("\n" + "="*70)
        print("Training completed successfully!")
        print(f"Best validation accuracy: {best_info['val_metrics']['acc']:.2f}%")
        print(f"Best epoch: {best_info['epoch']}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Latest checkpoint saved. You can resume training with --resume flag")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
