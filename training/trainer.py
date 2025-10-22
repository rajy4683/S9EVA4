"""
ResNet-50 Trainer
Comprehensive training loop with all features.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict
from pathlib import Path

from training.config import TrainingConfig
from training.metrics import MetricsTracker, compute_metrics
from utils.logger import MetricsLogger, ProgressPrinter
from utils.checkpointing import CheckpointManager


class ResNetTrainer:
    """Trainer for ResNet-50 on ImageNet subset."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: TrainingConfig
    ):
        """
        Initialize trainer.

        Args:
            model: ResNet-50 model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Initialize optimizer with weight decay handling
        self.optimizer = self._create_optimizer()

        # Initialize scheduler (OneCycleLR)
        self.scheduler = self._create_scheduler()

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        ).to(self.device)

        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None

        # Logging and checkpointing
        self.logger = MetricsLogger(
            config.log_dir,
            experiment_name=self._get_experiment_name()
        )
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            metric_name=config.metric_for_best,
            mode='max'
        )

        # Track training state
        self.current_epoch = 0
        self.global_step = 0

        # Log configuration
        self.logger.log_config(config.to_dict())

        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model params: {self._count_parameters():,}")
        print(f"  Optimizer: {config.optimizer}")
        print(f"  Scheduler: OneCycleLR")
        print(f"  Mixed precision: {config.use_amp}")

    def _get_experiment_name(self) -> str:
        """Generate experiment name."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"resnet50_imagenet_{timestamp}"

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _create_optimizer(self) -> SGD:
        """
        Create SGD optimizer with Nesterov momentum.
        Optionally exclude biases and BatchNorm from weight decay.
        """
        if self.config.exclude_bias_and_bn_from_wd:
            # Separate parameters: those with WD and those without
            decay_params = []
            no_decay_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                # No weight decay for biases and BN parameters
                if 'bias' in name or 'bn' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            param_groups = [
                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]

            print(f"  Parameters with WD: {sum(p.numel() for p in decay_params):,}")
            print(f"  Parameters without WD: {sum(p.numel() for p in no_decay_params):,}")
        else:
            param_groups = self.model.parameters()

        optimizer = SGD(
            param_groups,
            lr=self.config.initial_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay if not self.config.exclude_bias_and_bn_from_wd else 0,
            nesterov=self.config.nesterov
        )

        return optimizer

    def _create_scheduler(self) -> OneCycleLR:
        """Create OneCycleLR scheduler."""
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.max_lr,
            epochs=self.config.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=self.config.pct_start,
            div_factor=self.config.div_factor,
            final_div_factor=self.config.final_div_factor,
            anneal_strategy=self.config.anneal_strategy
        )

        return scheduler

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        tracker = MetricsTracker()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                self.optimizer.step()

            # Update scheduler (per batch for OneCycleLR)
            self.scheduler.step()

            # Compute metrics
            metrics = compute_metrics(outputs, targets, loss.item())
            for key, value in metrics.items():
                tracker.update(key, value, images.size(0))

            # Print progress
            if (batch_idx + 1) % self.config.print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                ProgressPrinter.print_batch_progress(
                    "Train",
                    batch_idx + 1,
                    len(self.train_loader),
                    tracker.get('loss'),
                    tracker.get('acc'),
                    current_lr
                )

            self.global_step += 1

        return tracker.get_dict()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        tracker = MetricsTracker()

        for batch_idx, (images, targets) in enumerate(self.val_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Compute metrics
            metrics = compute_metrics(outputs, targets, loss.item())
            for key, value in metrics.items():
                tracker.update(key, value, images.size(0))

            # Print progress occasionally
            if (batch_idx + 1) % (self.config.print_freq * 2) == 0:
                ProgressPrinter.print_batch_progress(
                    "Val  ",
                    batch_idx + 1,
                    len(self.val_loader),
                    tracker.get('loss'),
                    tracker.get('acc')
                )

        return tracker.get_dict()

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs...")
        total_start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch

            # Print epoch header
            ProgressPrinter.print_epoch_start(epoch, self.config.epochs)

            # Train
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Compute epoch time
            epoch_time = time.time() - epoch_start_time

            # Print summary
            ProgressPrinter.print_epoch_summary(
                epoch, train_metrics, val_metrics, current_lr, epoch_time
            )

            # Log metrics
            self.logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Check if best model
            metric_value = val_metrics[self.config.metric_for_best]
            is_best = self.checkpoint_manager.is_best_model(metric_value)
            if is_best:
                self.checkpoint_manager.update_best(epoch, metric_value)

            # Save checkpoint
            if epoch % self.config.save_freq == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics={'train': train_metrics, 'val': val_metrics},
                    is_best=is_best
                )

        # Training complete
        total_time = time.time() - total_start_time
        best_info = self.logger.get_best_epoch(self.config.metric_for_best)

        ProgressPrinter.print_training_complete(
            total_time,
            best_info['val_metrics'][self.config.metric_for_best],
            best_info['epoch']
        )

        # Close logger
        self.logger.close()

        return best_info

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.device
        )

        self.current_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {self.current_epoch}")


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use scripts/train.py to start training.")
