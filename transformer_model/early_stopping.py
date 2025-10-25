#!/usr/bin/env python
# coding: utf-8

"""
Early stopping monitor for training stability and convergence detection.

Provides instability detection, plateau detection, and overfitting detection.
"""

from typing import Optional, Dict
import numpy as np

# --- Early Stopping Monitor ---
class EarlyStoppingMonitor:
    """Monitor training for early stopping conditions.

    Detects:
        1. Early instability (wild loss swings in first N epochs)
        2. Plateau (no improvement for N epochs)
        3. Overfitting (train loss decreasing but val metric degrading)
        4. NaN/Inf in losses or gradients
    """

    def __init__(
        self,
        patience: int = 10,
        overfit_patience: int = 5,
        warmup_epochs: int = 5,
        instability_threshold: float = 7.0,
        maximize_metric: bool = False,
        min_delta: float = 1e-4
    ):
        """Initialize early stopping monitor.

        Args:
            patience: Stop if no improvement for N epochs (0=disable)
            overfit_patience: Stop if overfitting for N epochs (0=disable)
            warmup_epochs: Number of epochs to check for instability
            instability_threshold: Stop if loss changes > N× stddev in warmup
            maximize_metric: Whether higher metric is better (True) or lower is better (False)
            min_delta: Minimum change to consider as improvement
        """
        self.patience = patience
        self.overfit_patience = overfit_patience
        self.warmup_epochs = warmup_epochs
        self.instability_threshold = instability_threshold
        self.maximize_metric = maximize_metric
        self.min_delta = min_delta

        # Tracking
        self.best_metric = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.train_loss_history = []
        self.val_metric_history = []
        self.overfit_counter = 0

        # Stopping state
        self.should_stop = False
        self.stop_reason = None

    def check_instability(self, epoch: int, train_loss: float) -> bool:
        """Check for wild loss swings in early epochs.

        Args:
            epoch: Current epoch number (1-indexed)
            train_loss: Current training loss

        Returns:
            True if instability detected (should stop), False otherwise
        """
        if epoch > self.warmup_epochs or self.warmup_epochs == 0:
            return False

        # Check for NaN/Inf first
        if not np.isfinite(train_loss):
            self.should_stop = True
            self.stop_reason = (
                f"Training instability: NaN/Inf loss detected at epoch {epoch}\n"
                f"  Likely causes: Learning rate too high, gradient explosion, or numerical instability\n"
                f"  Recommendation: Reduce learning rate by 5-10× or enable gradient clipping"
            )
            return True

        # Check for wild swings (BEFORE adding current loss to history)
        # Need at least 3 epochs of history to detect instability
        if len(self.train_loss_history) >= 3:
            losses = np.array(self.train_loss_history)
            mean_loss = losses.mean()
            std_loss = losses.std()

            # Check if current loss is wildly different from historical mean
            if std_loss > 0:
                z_score = abs(train_loss - mean_loss) / std_loss
                if z_score > self.instability_threshold:
                    self.should_stop = True
                    self.stop_reason = (
                        f"Training instability: Loss variance too high in first {self.warmup_epochs} epochs\n"
                        f"  Historical losses: {[f'{l:.4f}' for l in losses.tolist()]}\n"
                        f"  Current loss: {train_loss:.4f}\n"
                        f"  Loss deviation: {z_score:.2f}× standard deviation (threshold: {self.instability_threshold}×)\n"
                        f"  Likely causes: Learning rate too high, batch size too small, or data issues\n"
                        f"  Recommendation: Reduce learning rate by 2-5× or increase batch size"
                    )
                    return True

        # Add current loss to history for next iteration
        self.train_loss_history.append(train_loss)

        return False

    def check_plateau(self, epoch: int, val_metric: float) -> bool:
        """Check if validation metric has plateaued.

        Args:
            epoch: Current epoch number (1-indexed)
            val_metric: Current validation metric value

        Returns:
            True if plateau detected (should stop), False otherwise
        """
        if self.patience == 0:
            return False

        # Check for NaN/Inf
        if not np.isfinite(val_metric):
            self.should_stop = True
            self.stop_reason = (
                f"Validation metric is NaN/Inf at epoch {epoch}\n"
                f"  This indicates severe training instability\n"
                f"  Recommendation: Check data quality, reduce learning rate, or adjust model architecture"
            )
            return True

        # Initialize best metric on first call
        if self.best_metric is None:
            self.best_metric = val_metric
            self.best_epoch = epoch
            return False

        # Check if current metric is better than best
        if self.maximize_metric:
            is_improvement = val_metric > (self.best_metric + self.min_delta)
        else:
            is_improvement = val_metric < (self.best_metric - self.min_delta)

        if is_improvement:
            self.best_metric = val_metric
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Check if we've exceeded patience
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            metric_direction = "higher" if self.maximize_metric else "lower"
            self.stop_reason = (
                f"Plateau detected: No improvement in validation metric for {self.patience} epochs\n"
                f"  Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}\n"
                f"  Current metric: {val_metric:.4f} at epoch {epoch}\n"
                f"  Training stopped early - best model already saved\n"
                f"  Note: Model may benefit from different hyperparameters or more regularization"
            )
            return True

        return False

    def check_overfitting(self, epoch: int, train_loss: float, val_metric: float) -> bool:
        """Check if model is overfitting (train improving but val degrading).

        Args:
            epoch: Current epoch number (1-indexed)
            train_loss: Current training loss
            val_metric: Current validation metric value

        Returns:
            True if overfitting detected (should stop), False otherwise
        """
        if self.overfit_patience == 0 or epoch <= self.warmup_epochs:
            return False

        self.val_metric_history.append(val_metric)

        # Need enough history to detect trend
        if len(self.train_loss_history) < 2 or len(self.val_metric_history) < 2:
            return False

        # Check if train loss is decreasing
        recent_train_losses = self.train_loss_history[-2:]
        train_improving = recent_train_losses[-1] < recent_train_losses[-2]

        # Check if val metric is degrading
        recent_val_metrics = self.val_metric_history[-2:]
        if self.maximize_metric:
            val_degrading = recent_val_metrics[-1] < recent_val_metrics[-2]
        else:
            val_degrading = recent_val_metrics[-1] > recent_val_metrics[-2]

        # Increment counter if both conditions met
        if train_improving and val_degrading:
            self.overfit_counter += 1
        else:
            self.overfit_counter = 0

        # Check if we've exceeded patience
        if self.overfit_counter >= self.overfit_patience:
            self.should_stop = True
            self.stop_reason = (
                f"Overfitting detected: Train loss decreasing but validation metric degrading for {self.overfit_patience} epochs\n"
                f"  Best validation metric: {self.best_metric:.4f} at epoch {self.best_epoch}\n"
                f"  Current validation metric: {val_metric:.4f} at epoch {epoch}\n"
                f"  Train loss trend: {[f'{l:.4f}' for l in self.train_loss_history[-(self.overfit_patience+1):]]}\n"
                f"  Best model saved at epoch {self.best_epoch}\n"
                f"  Recommendation: Increase regularization (dropout, weight decay) or reduce model capacity"
            )
            return True

        return False

    def update(self, epoch: int, train_loss: float, val_metric: float) -> bool:
        """Update monitor with latest metrics and check all stopping conditions.

        Args:
            epoch: Current epoch number (1-indexed)
            train_loss: Current training loss
            val_metric: Current validation metric value

        Returns:
            True if training should stop, False otherwise
        """
        # Add to history (for overfitting detection)
        if epoch > self.warmup_epochs:
            # Only track stable training history after warmup
            self.train_loss_history.append(train_loss)

        # Check instability (only in warmup period)
        if self.check_instability(epoch, train_loss):
            return True

        # Check plateau
        if self.check_plateau(epoch, val_metric):
            return True

        # Check overfitting
        if self.check_overfitting(epoch, train_loss, val_metric):
            return True

        return False


