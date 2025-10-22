"""
Checkpointing Utilities
Robust model checkpointing and best-model snapshotting.
"""

import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manage model checkpoints and best model saving."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        metric_name: str = "val_acc",
        mode: str = "max"
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            metric_name: Metric to track for best model
            mode: 'max' or 'min' for best metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode

        # Track best metric
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0

        # Keep track of saved checkpoints
        self.saved_checkpoints = []

        print(f"Checkpoint manager initialized: {self.checkpoint_dir}")
        print(f"Tracking best {metric_name} ({mode})")

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_state: Optional[Dict] = None
    ) -> str:
        """
        Save checkpoint.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            metrics: Metrics dictionary
            is_best: Whether this is the best model
            extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state is not None:
            checkpoint.update(extra_state)

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Track saved checkpoint
        self.saved_checkpoints.append(checkpoint_path)

        # Remove old checkpoints if exceeding max
        self._cleanup_old_checkpoints()

        # Save best model if this is the best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy2(checkpoint_path, best_path)
            print(f"Saved best model: {best_path}")

        # Always save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        shutil.copy2(checkpoint_path, latest_path)

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints recent ones."""
        if len(self.saved_checkpoints) > self.max_checkpoints:
            # Sort by epoch number (extracted from filename)
            self.saved_checkpoints.sort(
                key=lambda p: int(p.stem.split('_')[-1])
            )

            # Remove oldest checkpoints
            to_remove = self.saved_checkpoints[:-self.max_checkpoints]
            for checkpoint_path in to_remove:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path}")

            # Update list
            self.saved_checkpoints = self.saved_checkpoints[-self.max_checkpoints:]

    def is_best_model(self, current_metric: float) -> bool:
        """
        Check if current metric is the best.

        Args:
            current_metric: Current metric value

        Returns:
            True if this is the best model
        """
        if self.mode == 'max':
            is_best = current_metric > self.best_metric
        else:
            is_best = current_metric < self.best_metric

        if is_best:
            self.best_metric = current_metric

        return is_best

    def update_best(self, epoch: int, metric: float):
        """
        Update best metric and epoch.

        Args:
            epoch: Current epoch
            metric: Current metric value
        """
        if self.is_best_model(metric):
            self.best_epoch = epoch
            print(f"New best {self.metric_name}: {metric:.4f} (Epoch {epoch})")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to map tensors to

        Returns:
            Checkpoint dictionary
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Update best metric
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    def load_best_model(
        self,
        model: torch.nn.Module,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load best model.

        Args:
            model: Model to load state into
            device: Device to map tensors to

        Returns:
            Checkpoint dictionary
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        return self.load_checkpoint(str(best_path), model, device=device)

    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load latest checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to map tensors to

        Returns:
            Checkpoint dictionary
        """
        latest_path = self.checkpoint_dir / "latest.pth"
        return self.load_checkpoint(
            str(latest_path), model, optimizer, scheduler, device
        )

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about saved checkpoints."""
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'num_checkpoints': len(self.saved_checkpoints),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'metric_name': self.metric_name,
            'mode': self.mode
        }


if __name__ == "__main__":
    # Test checkpoint manager
    import tempfile
    import torch.nn as nn

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simple model
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Create checkpoint manager
        manager = CheckpointManager(tmpdir, max_checkpoints=3)

        # Save some checkpoints
        for epoch in range(1, 6):
            metrics = {'val_acc': 50 + epoch * 5, 'val_loss': 2.0 - epoch * 0.2}
            is_best = manager.is_best_model(metrics['val_acc'])
            manager.update_best(epoch, metrics['val_acc'])

            manager.save_checkpoint(
                epoch, model, optimizer, None, metrics, is_best
            )

        # Check info
        info = manager.get_checkpoint_info()
        print(f"\nCheckpoint info: {info}")

        # Load best model
        manager.load_best_model(model)

        print("\nCheckpoint manager test passed!")
