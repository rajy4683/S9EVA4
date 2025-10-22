"""
Training Metrics
Utilities for computing and tracking training metrics.
"""

import torch
from typing import Tuple, Dict


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ''):
        """
        Initialize meter.

        Args:
            name: Name of the metric
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.

        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.avg:.4f}"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list:
    """
    Compute top-k accuracy.

    Args:
        output: Model predictions (B, num_classes)
        target: Ground truth labels (B,)
        topk: Tuple of k values to compute accuracy for

    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

        return res


class MetricsTracker:
    """Track multiple metrics during training."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}

    def add_metric(self, name: str):
        """
        Add a new metric to track.

        Args:
            name: Metric name
        """
        self.metrics[name] = AverageMeter(name)

    def update(self, name: str, value: float, n: int = 1):
        """
        Update a metric.

        Args:
            name: Metric name
            value: New value
            n: Number of samples
        """
        if name not in self.metrics:
            self.add_metric(name)
        self.metrics[name].update(value, n)

    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def get(self, name: str) -> float:
        """
        Get current average value of a metric.

        Args:
            name: Metric name

        Returns:
            Average value
        """
        return self.metrics[name].avg if name in self.metrics else 0.0

    def get_dict(self) -> Dict[str, float]:
        """
        Get all metrics as dictionary.

        Returns:
            Dictionary of metric names to average values
        """
        return {name: meter.avg for name, meter in self.metrics.items()}

    def __str__(self) -> str:
        """String representation."""
        return " | ".join([str(meter) for meter in self.metrics.values()])


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss: float
) -> Dict[str, float]:
    """
    Compute standard metrics (loss, top-1, top-5 accuracy).

    Args:
        outputs: Model predictions (B, num_classes)
        targets: Ground truth labels (B,)
        loss: Computed loss value

    Returns:
        Dictionary of metrics
    """
    # Compute accuracies
    if outputs.size(1) >= 5:
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        return {
            'loss': loss,
            'acc': acc1,
            'top5_acc': acc5
        }
    else:
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        return {
            'loss': loss,
            'acc': acc1
        }


class ConfusionMatrix:
    """Compute and store confusion matrix."""

    def __init__(self, num_classes: int):
        """
        Initialize confusion matrix.

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix.

        Args:
            predictions: Predicted class indices (B,)
            targets: Ground truth class indices (B,)
        """
        for t, p in zip(targets.view(-1), predictions.view(-1)):
            self.matrix[t.long(), p.long()] += 1

    def reset(self):
        """Reset confusion matrix."""
        self.matrix.zero_()

    def get_matrix(self) -> torch.Tensor:
        """Get confusion matrix."""
        return self.matrix

    def get_per_class_accuracy(self) -> torch.Tensor:
        """
        Compute per-class accuracy.

        Returns:
            Tensor of per-class accuracies
        """
        class_correct = self.matrix.diag()
        class_total = self.matrix.sum(dim=1)
        return class_correct.float() / (class_total.float() + 1e-8)

    def get_overall_accuracy(self) -> float:
        """
        Compute overall accuracy.

        Returns:
            Overall accuracy
        """
        return (self.matrix.diag().sum().float() / (self.matrix.sum().float() + 1e-8)).item()


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics utilities...")

    # Test AverageMeter
    meter = AverageMeter('loss')
    meter.update(1.5, 32)
    meter.update(1.3, 32)
    meter.update(1.2, 32)
    print(f"\nAverageMeter: {meter}")
    assert abs(meter.avg - 1.333) < 0.01, "AverageMeter test failed"

    # Test accuracy
    outputs = torch.randn(10, 100)
    targets = torch.randint(0, 100, (10,))
    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    print(f"Top-1 Acc: {acc1:.2f}%, Top-5 Acc: {acc5:.2f}%")

    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.update('loss', 1.5, 32)
    tracker.update('acc', 85.3, 32)
    print(f"\nMetricsTracker: {tracker}")
    print(f"Metrics dict: {tracker.get_dict()}")

    # Test ConfusionMatrix
    cm = ConfusionMatrix(num_classes=5)
    preds = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3])
    cm.update(preds, targets)
    print(f"\nConfusion Matrix:\n{cm.get_matrix()}")
    print(f"Overall Accuracy: {cm.get_overall_accuracy():.2f}")
    print(f"Per-class Accuracy: {cm.get_per_class_accuracy()}")

    print("\nAll metrics tests passed!")
