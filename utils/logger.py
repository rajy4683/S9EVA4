"""
Logging Utilities
Lightweight logging for training metrics and progress.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import csv


class MetricsLogger:
    """
    Simple metrics logger for training/validation.
    Saves metrics to JSON and CSV files.
    """

    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # Create experiment directory
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics = {
            'train': [],
            'val': [],
            'lr': []
        }

        # File paths
        self.json_path = self.exp_dir / "metrics.json"
        self.train_csv_path = self.exp_dir / "train_metrics.csv"
        self.val_csv_path = self.exp_dir / "val_metrics.csv"

        # CSV writers
        self.train_csv_file = None
        self.val_csv_file = None
        self.train_writer = None
        self.val_writer = None

        print(f"Logger initialized: {self.exp_dir}")

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float], lr: float):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            lr: Current learning rate
        """
        # Add epoch and timestamp
        train_metrics['epoch'] = epoch
        val_metrics['epoch'] = epoch
        train_metrics['timestamp'] = time.time()
        val_metrics['timestamp'] = time.time()

        # Store metrics
        self.metrics['train'].append(train_metrics)
        self.metrics['val'].append(val_metrics)
        self.metrics['lr'].append({'epoch': epoch, 'lr': lr})

        # Write to CSV
        self._write_csv(train_metrics, val_metrics)

        # Save to JSON
        self._save_json()

    def _write_csv(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Write metrics to CSV files."""
        # Initialize CSV files if needed
        if self.train_csv_file is None:
            self.train_csv_file = open(self.train_csv_path, 'w', newline='')
            self.train_writer = csv.DictWriter(self.train_csv_file, fieldnames=train_metrics.keys())
            self.train_writer.writeheader()

        if self.val_csv_file is None:
            self.val_csv_file = open(self.val_csv_path, 'w', newline='')
            self.val_writer = csv.DictWriter(self.val_csv_file, fieldnames=val_metrics.keys())
            self.val_writer.writeheader()

        # Write rows
        self.train_writer.writerow(train_metrics)
        self.val_writer.writerow(val_metrics)

        # Flush to ensure data is written
        self.train_csv_file.flush()
        self.val_csv_file.flush()

    def _save_json(self):
        """Save all metrics to JSON."""
        with open(self.json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def log_config(self, config: Dict[str, Any]):
        """
        Save training configuration.

        Args:
            config: Configuration dictionary
        """
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def get_best_epoch(self, metric: str = 'val_acc', mode: str = 'max') -> Dict[str, Any]:
        """
        Get best epoch based on a metric.

        Args:
            metric: Metric name to compare
            mode: 'max' or 'min'

        Returns:
            Dictionary with best epoch info
        """
        if not self.metrics['val']:
            return {}

        if mode == 'max':
            best_idx = max(range(len(self.metrics['val'])),
                          key=lambda i: self.metrics['val'][i].get(metric, float('-inf')))
        else:
            best_idx = min(range(len(self.metrics['val'])),
                          key=lambda i: self.metrics['val'][i].get(metric, float('inf')))

        return {
            'epoch': self.metrics['val'][best_idx]['epoch'],
            'train_metrics': self.metrics['train'][best_idx],
            'val_metrics': self.metrics['val'][best_idx]
        }

    def close(self):
        """Close CSV files."""
        if self.train_csv_file is not None:
            self.train_csv_file.close()
        if self.val_csv_file is not None:
            self.val_csv_file.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class ProgressPrinter:
    """Simple progress printer for training."""

    @staticmethod
    def print_epoch_start(epoch: int, total_epochs: int):
        """Print epoch start."""
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"{'='*70}")

    @staticmethod
    def print_batch_progress(
        phase: str,
        batch_idx: int,
        total_batches: int,
        loss: float,
        acc: float,
        lr: Optional[float] = None
    ):
        """Print batch progress."""
        progress_str = f"{phase} [{batch_idx}/{total_batches}] "
        progress_str += f"Loss: {loss:.4f} | Acc: {acc:.2f}%"
        if lr is not None:
            progress_str += f" | LR: {lr:.6f}"
        print(progress_str)

    @staticmethod
    def print_epoch_summary(
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time: float
    ):
        """Print epoch summary."""
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary ({epoch_time:.2f}s)")
        print(f"{'-'*70}")
        print(f"Train - Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['acc']:.2f}% | "
              f"Top-5: {train_metrics.get('top5_acc', 0):.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['acc']:.2f}% | "
              f"Top-5: {val_metrics.get('top5_acc', 0):.2f}%")
        print(f"Learning Rate: {lr:.6f}")
        print(f"{'='*70}")

    @staticmethod
    def print_training_complete(total_time: float, best_acc: float, best_epoch: int):
        """Print training completion message."""
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'-'*70}")
        print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch})")
        print(f"{'='*70}")


if __name__ == "__main__":
    # Test logger
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(tmpdir, "test_experiment")

        # Log config
        logger.log_config({
            'batch_size': 256,
            'lr': 0.1,
            'epochs': 10
        })

        # Log some epochs
        for epoch in range(1, 4):
            train_metrics = {
                'loss': 2.5 - epoch * 0.3,
                'acc': 30 + epoch * 10,
                'top5_acc': 50 + epoch * 5
            }
            val_metrics = {
                'loss': 2.3 - epoch * 0.25,
                'acc': 35 + epoch * 8,
                'top5_acc': 55 + epoch * 4
            }
            logger.log_epoch(epoch, train_metrics, val_metrics, lr=0.1)

        # Get best epoch
        best = logger.get_best_epoch('acc', mode='max')
        print(f"\nBest epoch: {best}")

        print("\nLogger test passed!")
