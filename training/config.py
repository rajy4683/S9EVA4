"""
Training Configuration
Contains all hyperparameters and settings for training.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class TrainingConfig:
    """Configuration for ResNet-50 training."""

    # Data settings
    data_dir: str = "./imagenet_subset"
    train_dir: str = "train"
    val_dir: str = "val"
    num_classes: int = 100
    image_size: int = 224

    # Training hyperparameters
    epochs: int = 90
    batch_size: int = 256
    num_workers: int = 4

    # Optimizer settings (SGD + Nesterov)
    optimizer: str = "sgd"
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 1e-4
    exclude_bias_and_bn_from_wd: bool = True  # Exclude biases/BN from weight decay

    # Learning rate settings (OneCycleLR)
    initial_lr: float = 0.07047  # From LR finder
    max_lr: float = 0.7047       # From LR finder
    div_factor: float = 25.0     # initial_lr = max_lr / div_factor
    final_div_factor: float = 1e4  # final_lr = initial_lr / final_div_factor
    pct_start: float = 0.3       # Percentage of cycle spent increasing LR
    anneal_strategy: str = "cos"  # 'cos' or 'linear'

    # Loss settings
    label_smoothing: float = 0.1

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Model settings
    zero_init_residual: bool = True

    # Checkpoint and logging
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 5  # Save checkpoint every N epochs
    print_freq: int = 100  # Print progress every N batches

    # Best model tracking
    save_best_only: bool = True
    metric_for_best: str = "val_acc"  # Metric to track for best model

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision

    # Reproducibility
    seed: int = 42

    # Augmentation
    auto_augment: bool = False

    # Resume training
    resume_from: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 <= self.label_smoothing < 1, "label_smoothing must be in [0, 1)"
        assert 0 < self.pct_start < 1, "pct_start must be in (0, 1)"
        assert self.max_lr > self.initial_lr, "max_lr must be greater than initial_lr"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation."""
        config_str = "TrainingConfig:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str


@dataclass
class LRFinderConfig:
    """Configuration for LR Finder."""

    # Data settings
    data_dir: str = "./imagenet_subset"
    train_dir: str = "train"
    num_classes: int = 100
    image_size: int = 224

    # LR Finder settings
    start_lr: float = 1e-7
    end_lr: float = 10.0
    num_iter: int = 100
    step_mode: str = "exp"  # 'exp' or 'linear'
    smooth_f: float = 0.05  # Smoothing factor for loss
    diverge_th: float = 5.0  # Divergence threshold

    # Training settings for LR finder
    batch_size: int = 256
    num_workers: int = 4

    # Optimizer settings (match training)
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model settings
    zero_init_residual: bool = True

    # Output
    plot_path: str = "./lr_finder_plot.png"
    suggested_lr_path: str = "./suggested_lr.txt"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
        }


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_LR_FINDER_CONFIG = LRFinderConfig()


if __name__ == "__main__":
    # Test configuration
    print("Default Training Configuration:")
    print(DEFAULT_TRAINING_CONFIG)

    print("\n" + "="*60)
    print("Default LR Finder Configuration:")
    config_dict = DEFAULT_LR_FINDER_CONFIG.to_dict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
