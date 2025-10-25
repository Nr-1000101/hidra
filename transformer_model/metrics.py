#!/usr/bin/env python
# coding: utf-8

"""
Metrics and loss functions for model evaluation and training.

Includes classification metrics, regression metrics, focal loss, and circular losses for angles.
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# --- Helper Functions for Metrics ---
def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        logits: (batch, num_classes) predicted logits
        targets: (batch,) true class indices

    Returns:
        Accuracy as percentage (0-100)
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        MAE as scalar
    """
    return torch.abs(predictions - targets).mean().item()


def circular_mse_loss(pred_angles: torch.Tensor, true_angles: torch.Tensor) -> torch.Tensor:
    """MSE loss for angles with 2π periodicity (future use for actual angle predictions).

    Args:
        pred_angles: Predicted angles in degrees
        true_angles: True angles in degrees

    Returns:
        Circular MSE loss (scalar tensor)
    """
    # Convert to radians
    pred_rad = pred_angles * (torch.pi / 180.0)
    true_rad = true_angles * (torch.pi / 180.0)

    # Compute wrapped angular difference using atan2
    diff = torch.atan2(torch.sin(pred_rad - true_rad), torch.cos(pred_rad - true_rad))

    # MSE on the wrapped difference
    return torch.mean(diff ** 2)


def circular_mae_loss(pred_angles: torch.Tensor, true_angles: torch.Tensor) -> torch.Tensor:
    """MAE for angles with circular distance (future use for actual angle predictions).

    Args:
        pred_angles: Predicted angles in degrees
        true_angles: True angles in degrees

    Returns:
        Circular MAE in radians (scalar tensor)
    """
    pred_rad = pred_angles * (torch.pi / 180.0)
    true_rad = true_angles * (torch.pi / 180.0)

    # Wrapped difference
    diff = torch.atan2(torch.sin(pred_rad - true_rad), torch.cos(pred_rad - true_rad))

    # MAE on absolute wrapped difference
    return torch.abs(diff).mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t: probability of correct class
        alpha_t: class weight for balancing class frequencies
        gamma: focusing parameter (higher = more focus on hard examples)

    Args:
        alpha: Class weights tensor (shape: num_classes) or None
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # Class weights (will be moved to correct device in forward)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits (shape: batch_size, num_classes)
            targets: Class indices (shape: batch_size)

        Returns:
            Focal loss (scalar)
        """
        # Compute cross entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # Get probabilities for correct class
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # Probability of true class

        # Compute focal loss: (1 - p_t)^gamma * CE_loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray = None,
    num_classes: int = None
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        predictions: Predicted class indices (shape: N)
        targets: True class indices (shape: N)
        probabilities: Class probabilities for AUPRC (shape: N, num_classes), optional
        num_classes: Number of classes (auto-detected if None)

    Returns:
        Dictionary with metrics: accuracy, macro_f1, weighted_f1, macro_recall, auprc
    """
    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    metrics = {}

    # Basic accuracy
    metrics["accuracy"] = 100.0 * accuracy_score(targets, predictions)

    # F1 scores
    metrics["macro_f1"] = f1_score(targets, predictions, average='macro', zero_division=0.0)
    metrics["weighted_f1"] = f1_score(targets, predictions, average='weighted', zero_division=0.0)

    # Recall scores
    metrics["macro_recall"] = recall_score(targets, predictions, average='macro', zero_division=0.0)
    metrics["weighted_recall"] = recall_score(targets, predictions, average='weighted', zero_division=0.0)

    # AUPRC (requires probabilities)
    if probabilities is not None:
        try:
            # For binary classification (num_classes=2), compute directly
            # For multi-class, use macro average over one-hot encoded targets
            if num_classes == 2:
                # Binary: use positive class probabilities (no 'average' parameter needed)
                metrics["auprc"] = average_precision_score(
                    targets, probabilities[:, 1]
                )
            else:
                # Multi-class: one-hot encode targets and compute macro AUPRC
                targets_onehot = label_binarize(targets, classes=range(num_classes))
                metrics["auprc"] = average_precision_score(
                    targets_onehot, probabilities, average='macro'
                )
        except Exception as e:
            # If AUPRC computation fails (e.g., missing classes), set to 0
            metrics["auprc"] = 0.0

    return metrics


def compute_class_weights(dataloader: DataLoader, property_name: str, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced classification.

    Args:
        dataloader: DataLoader to compute weights from
        property_name: Name of the classification property
        num_classes: Number of classes
        device: Device to put weights on

    Returns:
        Tensor of class weights (shape: num_classes)
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)

    # Count samples per class
    for _, _, batch_targets in dataloader:
        targets = batch_targets[property_name]
        for target in targets:
            class_counts[target.item()] += 1

    # Compute inverse frequency weights
    # weight = total_samples / (num_classes * class_count)
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)

    # Cap maximum weight ratio to prevent extreme gradient imbalance
    max_ratio = 20.0
    min_weight = weights[weights > 0].min()  # Ignore zero weights
    weights = torch.clamp(weights, max=min_weight * max_ratio)

    # Handle zero counts (shouldn't happen but safety check)
    weights[class_counts == 0] = 0.0

    return weights.to(device)


