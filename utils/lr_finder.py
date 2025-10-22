"""
Learning Rate Finder
Based on the cyclical learning rate paper and fastai implementation.

Reference: https://github.com/rajy4683/mini-Rekog/blob/master/miniRekog/utils/lrfinderutils.py
"""

import copy
import torch
import torch.nn as nn
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


class LRFinder:
    """
    Learning Rate Finder for finding optimal learning rate range.

    The LR finder gradually increases the learning rate and tracks the loss.
    The optimal learning rate is typically found where:
    1. Loss is still decreasing rapidly
    2. Just before loss starts to diverge
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize LR Finder.

        Args:
            model: Neural network model
            optimizer: Optimizer (will be reset during LR finding)
            criterion: Loss function
            device: Device to run on
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Store original model and optimizer states
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        # Results storage
        self.lrs = []
        self.losses = []
        self.best_loss = float('inf')

    def range_test(
        self,
        train_loader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        step_mode: str = 'exp',
        smooth_f: float = 0.05,
        diverge_th: float = 5.0
    ):
        """
        Perform LR range test.

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            step_mode: 'exp' for exponential or 'linear' for linear
            smooth_f: Smoothing factor for loss (0-1)
            diverge_th: Threshold for divergence (multiplier of best loss)
        """
        # Reset model and optimizer
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Set model to training mode
        self.model.train()

        # Initialize
        self.lrs = []
        self.losses = []
        self.best_loss = float('inf')

        # Calculate LR schedule
        if step_mode == 'exp':
            lr_schedule = np.geomspace(start_lr, end_lr, num_iter)
        else:
            lr_schedule = np.linspace(start_lr, end_lr, num_iter)

        # Create iterator
        iterator = iter(train_loader)

        print(f"Running LR Finder: {start_lr:.2e} -> {end_lr:.2e} ({num_iter} iterations)")

        for iteration in tqdm(range(num_iter), desc="LR Finder"):
            # Get batch
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Set learning rate
            lr = lr_schedule[iteration]
            self._set_learning_rate(lr)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            current_loss = loss.item()

            # Smooth loss
            if iteration == 0:
                smoothed_loss = current_loss
            else:
                smoothed_loss = smooth_f * current_loss + (1 - smooth_f) * self.losses[-1]

            # Store
            self.lrs.append(lr)
            self.losses.append(smoothed_loss)

            # Update best loss
            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > diverge_th * self.best_loss:
                print(f"Stopping early: loss diverged at iteration {iteration}")
                break

        # Restore original model and optimizer
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        print("LR Finder complete!")

    def _set_learning_rate(self, lr: float):
        """Set learning rate for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        log_lr: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot LR finder results.

        Args:
            skip_start: Number of batches to skip at start
            skip_end: Number of batches to skip at end
            log_lr: Use log scale for learning rate
            save_path: Path to save plot (optional)
        """
        if len(self.lrs) == 0:
            print("No data to plot. Run range_test() first.")
            return

        # Prepare data
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, linewidth=2)

        if log_lr:
            plt.xscale('log')

        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('LR Finder Results', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Mark suggested LR
        suggested_lr = self.suggest_lr(skip_start, skip_end)
        if suggested_lr is not None:
            plt.axvline(x=suggested_lr, color='r', linestyle='--', linewidth=2,
                       label=f'Suggested LR: {suggested_lr:.2e}')
            plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")

        plt.show()

    def suggest_lr(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        method: str = 'steepest'
    ) -> Optional[float]:
        """
        Suggest optimal learning rate.

        Args:
            skip_start: Number of batches to skip at start
            skip_end: Number of batches to skip at end
            method: 'steepest' for steepest gradient or 'minimum' for minimum loss

        Returns:
            Suggested learning rate
        """
        if len(self.lrs) == 0:
            return None

        # Prepare data
        lrs = np.array(self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:])
        losses = np.array(self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:])

        if len(losses) < 2:
            return None

        if method == 'steepest':
            # Find steepest gradient
            gradients = np.gradient(losses)
            min_gradient_idx = np.argmin(gradients)
            suggested_lr = lrs[min_gradient_idx]
        else:
            # Find minimum loss
            min_loss_idx = np.argmin(losses)
            # Suggest LR one order of magnitude lower
            suggested_lr = lrs[min_loss_idx] / 10.0

        return suggested_lr

    def get_lr_range(
        self,
        skip_start: int = 10,
        skip_end: int = 5
    ) -> Tuple[float, float]:
        """
        Get suggested LR range for OneCycleLR.

        Returns:
            Tuple of (initial_lr, max_lr)
        """
        suggested_lr = self.suggest_lr(skip_start, skip_end, method='steepest')

        if suggested_lr is None:
            # Default values
            initial_lr = 0.07047
            max_lr = 0.7047
        else:
            # max_lr is the suggested LR
            max_lr = suggested_lr
            # initial_lr = max_lr / div_factor (div_factor=25 by default)
            initial_lr = max_lr / 25.0

        print(f"\nSuggested LR Range:")
        print(f"  initial_lr: {initial_lr:.6f}")
        print(f"  max_lr: {max_lr:.6f}")
        print(f"  (div_factor: 25.0)")

        return initial_lr, max_lr


def find_lr(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str = 'cuda',
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    plot_path: Optional[str] = None
) -> Tuple[float, float]:
    """
    Convenience function to find learning rate.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations
        plot_path: Path to save plot (optional)

    Returns:
        Tuple of (initial_lr, max_lr)
    """
    lr_finder = LRFinder(model, optimizer, criterion, device)

    # Run range test
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter
    )

    # Plot results
    lr_finder.plot(save_path=plot_path)

    # Get suggested LR range
    initial_lr, max_lr = lr_finder.get_lr_range()

    return initial_lr, max_lr


if __name__ == "__main__":
    print("LR Finder module loaded successfully!")
    print("Use scripts/find_lr.py to run LR finder.")
