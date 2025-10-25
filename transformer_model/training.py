#!/usr/bin/env python
# coding: utf-8

"""
Training and evaluation loops for the transformer model.

Handles forward/backward passes, metric computation, and logging.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics import (
    classification_accuracy,
    compute_mae,
    circular_mse_loss,
    circular_mae_loss,
    compute_classification_metrics,
    FocalLoss
)

# --- Training & Evaluation ---
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    property_configs: List[Dict],
    bf16: bool = True,
    scheduler = None,
    class_weights: Dict[str, torch.Tensor] = None,
    task_loss_weights: Dict[str, float] = None
) -> Tuple[float, Dict[str, float], bool]:
    """Train for one epoch with per-property loss tracking and health checks.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        property_configs: List of property configurations with 'name' and 'task' keys
        bf16: Whether to use bfloat16 autocast
        scheduler: Optional learning rate scheduler (OneCycleLR steps per batch)
        class_weights: Optional dict of class weights for imbalanced classification {prop_name: weights_tensor}
        task_loss_weights: Optional dict of per-task loss weights for multi-task balancing {prop_name: weight}

    Returns:
        epoch_loss: Average total training loss
        property_losses: Dictionary of per-property losses {prop_name: loss_value}
        is_healthy: False if NaN/Inf detected in losses or gradients, True otherwise
    """
    model.train()
    epoch_loss = 0.0
    is_healthy = True  # Track if NaN/Inf detected

    # Track per-property losses
    property_losses = {prop_cfg["name"]: 0.0 for prop_cfg in property_configs}
    property_counts = {prop_cfg["name"]: 0 for prop_cfg in property_configs}

    # Build loss functions for each property
    loss_functions = {}
    for prop_cfg in property_configs:
        if prop_cfg["task"] == "classification":
            # Choose loss function based on configuration
            if hasattr(model, 'loss_fn_type') and model.loss_fn_type == "focal":
                # Use Focal Loss with optional class weights
                alpha = class_weights.get(prop_cfg["name"]) if class_weights else None
                loss_functions[prop_cfg["name"]] = FocalLoss(
                    alpha=alpha,
                    gamma=model.focal_gamma
                )
            else:
                # Use CrossEntropyLoss with optional class weights
                if class_weights is not None and prop_cfg["name"] in class_weights:
                    loss_functions[prop_cfg["name"]] = nn.CrossEntropyLoss(
                        weight=class_weights[prop_cfg["name"]]
                    )
                else:
                    loss_functions[prop_cfg["name"]] = nn.CrossEntropyLoss()
        elif prop_cfg["task"] == "sequence_regression":
            # No pre-built loss function for masked sequence regression
            # Will compute manually with masking in the loop below
            loss_functions[prop_cfg["name"]] = None
        else:
            # Scalar regression
            loss_functions[prop_cfg["name"]] = nn.MSELoss()

    for batch_inputs, batch_masks, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)
        batch_targets = {k: v.to(device) for k, v in batch_targets.items()}

        optimizer.zero_grad()

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if bf16 else torch.autocast("cuda", enabled=False)
        )

        with autocast_ctx:
            batch_outputs = model(batch_inputs, batch_masks)
            batch_loss = 0.0

            # Accumulate loss for each property
            for prop_cfg in property_configs:
                prop_name = prop_cfg["name"]
                if prop_name not in batch_outputs:
                    continue

                prop_target = batch_targets[prop_name]
                prop_pred = batch_outputs[prop_name]

                if prop_cfg["task"] == "classification":
                    prop_loss = loss_functions[prop_name](prop_pred, prop_target.long())
                elif prop_cfg["task"] == "sequence_regression":
                    # Masked loss for variable-length sequences
                    # prop_pred: (batch_size, max_seq_len) e.g., (256, 105)
                    # prop_target: (batch_size, max_len_in_batch) e.g., (256, 6) if max in batch is 6
                    # Need to slice predictions to match target size
                    mask = batch_targets[prop_name + "_mask"]  # (batch_size, max_len_in_batch)
                    batch_max_len = prop_target.size(1)

                    # Slice predictions to match batch's max length
                    prop_pred_sliced = prop_pred[:, :batch_max_len]  # (batch_size, max_len_in_batch)

                    # Use circular MSE for angles (periodic values)
                    if prop_name == "ring_plane_angles":
                        # Convert to radians
                        pred_rad = prop_pred_sliced * (torch.pi / 180.0)
                        true_rad = prop_target * (torch.pi / 180.0)

                        # Compute wrapped angular difference using atan2
                        diff = torch.atan2(torch.sin(pred_rad - true_rad), torch.cos(pred_rad - true_rad))
                        squared_error = diff ** 2  # Squared circular distance
                    else:
                        # Standard squared error for non-angular sequences
                        squared_error = (prop_pred_sliced - prop_target) ** 2

                    masked_error = squared_error * mask

                    # Compute mean only over valid (non-masked) positions
                    n_valid = mask.sum()
                    if n_valid > 0:
                        prop_loss = masked_error.sum() / n_valid
                    else:
                        # No valid angles in this batch (all molecules have 0 rings)
                        prop_loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    # Scalar regression
                    prop_loss = loss_functions[prop_name](prop_pred.squeeze(-1), prop_target.float())

                # Apply task-level loss weight for multi-task balancing
                task_weight = 1.0
                if task_loss_weights is not None and prop_name in task_loss_weights:
                    task_weight = task_loss_weights[prop_name]

                batch_loss += task_weight * prop_loss

                # Track per-property loss (unweighted for monitoring)
                property_losses[prop_name] += prop_loss.item() * batch_inputs.size(0)
                property_counts[prop_name] += batch_inputs.size(0)

        batch_loss.backward()

        # Check for NaN/Inf in gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    print(f"  WARNING: NaN/Inf detected in gradients for parameter: {name}")
                    is_healthy = False
                    break

        # Check if loss is finite
        if not torch.isfinite(batch_loss):
            print(f"  WARNING: NaN/Inf detected in batch loss: {batch_loss.item()}")
            is_healthy = False

        optimizer.step()

        # Step scheduler if OneCycleLR (steps per batch)
        if scheduler is not None:
            scheduler.step()

        epoch_loss += batch_loss.item() * batch_inputs.size(0)

    # Compute averages
    epoch_loss /= len(dataloader.dataset)
    for prop_name in property_losses:
        if property_counts[prop_name] > 0:
            property_losses[prop_name] /= property_counts[prop_name]

    # Final check for NaN/Inf in epoch loss
    if not np.isfinite(epoch_loss):
        is_healthy = False

    return epoch_loss, property_losses, is_healthy

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    property_configs: List[Dict],
    bf16: bool = False,
    class_weights: Dict[str, torch.Tensor] = None,
    compute_confusion: bool = False,
    task_loss_weights: Dict[str, float] = None
) -> Tuple[float, float, Dict[str, Dict[str, any]]]:
    """Evaluate model with comprehensive per-property metrics.

    Args:
        model: Model to evaluate
        dataloader: Validation/test data loader
        device: Device to evaluate on
        property_configs: List of property configurations with 'name' and 'task' keys
        bf16: Whether to use bfloat16 autocast
        class_weights: Optional dict of class weights for imbalanced classification
        compute_confusion: Whether to compute confusion matrices for classification tasks
        task_loss_weights: Optional dict of per-task loss weights for multi-task balancing {prop_name: weight}

    Returns:
        mse: Overall MSE (for backward compatibility)
        rmse: Overall RMSE (for backward compatibility)
        property_metrics: Dictionary of per-property metrics:
            For classification: {"loss": float, "accuracy": float, "confusion_matrix": array (if requested)}
            For regression: {"loss": float, "rmse": float, "mae": float}
    """
    model.eval()
    eval_loss = 0.0

    # Initialize metric accumulators per property
    property_metrics = {}
    for prop_cfg in property_configs:
        prop_name = prop_cfg["name"]
        task = prop_cfg["task"]

        if task == "classification":
            property_metrics[prop_name] = {
                "loss": 0.0,
                "correct": 0,
                "total": 0,
                "all_predictions": [],
                "all_targets": [],
                "all_probabilities": [],  # For AUPRC computation
                "num_classes": prop_cfg.get("num_classes", 0)
            }
            # Initialize confusion matrix if requested
            if compute_confusion:
                num_classes = prop_cfg.get("num_classes", 0)
                property_metrics[prop_name]["confusion_matrix"] = np.zeros((num_classes, num_classes), dtype=np.int64)
        elif task == "sequence_regression":
            # Variable-length sequence regression
            property_metrics[prop_name] = {
                "loss": 0.0,
                "mae": 0.0,
                "n_valid_total": 0  # Total number of valid (non-masked) predictions
            }
        else:  # scalar regression
            property_metrics[prop_name] = {
                "loss": 0.0,
                "mae": 0.0,
                "count": 0
            }

    # Build loss functions for each property
    loss_functions = {}
    for prop_cfg in property_configs:
        if prop_cfg["task"] == "classification":
            # Choose loss function based on configuration
            if hasattr(model, 'loss_fn_type') and model.loss_fn_type == "focal":
                # Use Focal Loss with optional class weights
                alpha = class_weights.get(prop_cfg["name"]) if class_weights else None
                loss_functions[prop_cfg["name"]] = FocalLoss(
                    alpha=alpha,
                    gamma=model.focal_gamma
                )
            else:
                # Use CrossEntropyLoss with optional class weights
                if class_weights is not None and prop_cfg["name"] in class_weights:
                    loss_functions[prop_cfg["name"]] = nn.CrossEntropyLoss(
                        weight=class_weights[prop_cfg["name"]]
                    )
                else:
                    loss_functions[prop_cfg["name"]] = nn.CrossEntropyLoss()
        elif prop_cfg["task"] == "sequence_regression":
            # No pre-built loss function for masked sequence regression
            loss_functions[prop_cfg["name"]] = None
        else:
            # Scalar regression
            loss_functions[prop_cfg["name"]] = nn.MSELoss()

    with torch.no_grad():
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if bf16 else torch.autocast("cuda", enabled=False)
        )

        with autocast_ctx:
            for batch_inputs, batch_masks, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_masks = batch_masks.to(device)
                batch_targets = {k: v.to(device) for k, v in batch_targets.items()}

                batch_outputs = model(batch_inputs, batch_masks)
                batch_loss = 0.0

                # Accumulate metrics for each property
                for prop_cfg in property_configs:
                    prop_name = prop_cfg["name"]
                    if prop_name not in batch_outputs:
                        continue

                    prop_target = batch_targets[prop_name]
                    prop_pred = batch_outputs[prop_name]
                    task = prop_cfg["task"]

                    if task == "classification":
                        # Compute cross-entropy loss
                        prop_loss = loss_functions[prop_name](prop_pred, prop_target.long())
                        property_metrics[prop_name]["loss"] += prop_loss.item() * batch_inputs.size(0)

                        # Compute accuracy
                        predictions = prop_pred.argmax(dim=-1)
                        correct = (predictions == prop_target).sum().item()
                        property_metrics[prop_name]["correct"] += correct
                        property_metrics[prop_name]["total"] += batch_inputs.size(0)

                        # Track predictions, targets, and probabilities for comprehensive metrics
                        property_metrics[prop_name]["all_predictions"].extend(predictions.cpu().numpy().tolist())
                        property_metrics[prop_name]["all_targets"].extend(prop_target.cpu().numpy().tolist())

                        # Softmax probabilities for AUPRC
                        probabilities = F.softmax(prop_pred, dim=-1)
                        property_metrics[prop_name]["all_probabilities"].extend(probabilities.cpu().numpy().tolist())

                        # Apply task-level loss weight for multi-task balancing
                        task_weight = 1.0
                        if task_loss_weights is not None and prop_name in task_loss_weights:
                            task_weight = task_loss_weights[prop_name]
                        batch_loss += task_weight * prop_loss

                    elif task == "sequence_regression":
                        # Masked loss and MAE for variable-length sequences
                        # prop_pred: (batch_size, max_seq_len) e.g., (256, 105)
                        # prop_target: (batch_size, max_len_in_batch) e.g., (256, 6)
                        # Need to slice predictions to match target size
                        mask = batch_targets[prop_name + "_mask"]  # (batch_size, max_len_in_batch)
                        batch_max_len = prop_target.size(1)

                        # Slice predictions to match batch's max length
                        prop_pred_sliced = prop_pred[:, :batch_max_len]  # (batch_size, max_len_in_batch)

                        # Use circular MSE for angles (periodic values)
                        if prop_name == "ring_plane_angles":
                            # Convert to radians
                            pred_rad = prop_pred_sliced * (torch.pi / 180.0)
                            true_rad = prop_target * (torch.pi / 180.0)

                            # Compute wrapped angular difference using atan2
                            diff = torch.atan2(torch.sin(pred_rad - true_rad), torch.cos(pred_rad - true_rad))
                            squared_error = diff ** 2  # Squared circular distance

                            # Circular MAE (absolute angular difference)
                            abs_error = torch.abs(diff)
                        else:
                            # Standard squared error for non-angular sequences
                            squared_error = (prop_pred_sliced - prop_target) ** 2
                            abs_error = torch.abs(prop_pred_sliced - prop_target)

                        masked_error = squared_error * mask

                        # Compute MSE loss only over valid (non-masked) positions
                        n_valid = mask.sum()
                        if n_valid > 0:
                            prop_loss = masked_error.sum() / n_valid
                            property_metrics[prop_name]["loss"] += prop_loss.item() * n_valid.item()

                            # Compute MAE (circular for angles, standard otherwise)
                            masked_abs_error = abs_error * mask
                            mae = masked_abs_error.sum().item()
                            property_metrics[prop_name]["mae"] += mae
                            property_metrics[prop_name]["n_valid_total"] += n_valid.item()

                            # Apply task-level loss weight for multi-task balancing
                            task_weight = 1.0
                            if task_loss_weights is not None and prop_name in task_loss_weights:
                                task_weight = task_loss_weights[prop_name]
                            batch_loss += task_weight * prop_loss
                        else:
                            # No valid angles in this batch
                            prop_loss = torch.tensor(0.0, device=device)
                            # Apply task-level loss weight for multi-task balancing
                            task_weight = 1.0
                            if task_loss_weights is not None and prop_name in task_loss_weights:
                                task_weight = task_loss_weights[prop_name]
                            batch_loss += task_weight * prop_loss

                    else:  # scalar regression
                        prop_pred_squeezed = prop_pred.squeeze(-1)
                        prop_target_float = prop_target.float()

                        # Compute MSE loss
                        prop_loss = loss_functions[prop_name](prop_pred_squeezed, prop_target_float)
                        property_metrics[prop_name]["loss"] += prop_loss.item() * batch_inputs.size(0)

                        # Compute MAE
                        mae = torch.abs(prop_pred_squeezed - prop_target_float).sum().item()
                        property_metrics[prop_name]["mae"] += mae
                        property_metrics[prop_name]["count"] += batch_inputs.size(0)

                        # Apply task-level loss weight for multi-task balancing
                        task_weight = 1.0
                        if task_loss_weights is not None and prop_name in task_loss_weights:
                            task_weight = task_loss_weights[prop_name]
                        batch_loss += task_weight * prop_loss

                eval_loss += batch_loss.item() * batch_inputs.size(0)

    # Compute final metrics
    final_property_metrics = {}
    for prop_name, metrics in property_metrics.items():
        final_property_metrics[prop_name] = {}

        # Find task type for this property
        task = None
        for prop_cfg in property_configs:
            if prop_cfg["name"] == prop_name:
                task = prop_cfg["task"]
                break

        if task == "classification":
            # Average loss
            final_property_metrics[prop_name]["loss"] = metrics["loss"] / metrics["total"]

            # Compute comprehensive classification metrics
            preds = np.array(metrics["all_predictions"])
            targets = np.array(metrics["all_targets"])
            probs = np.array(metrics["all_probabilities"])
            num_classes = metrics["num_classes"]

            # Use our comprehensive metrics function
            comp_metrics = compute_classification_metrics(
                predictions=preds,
                targets=targets,
                probabilities=probs,
                num_classes=num_classes
            )

            # Add all metrics to final output
            final_property_metrics[prop_name].update(comp_metrics)

            # Determine primary metric based on number of classes
            if num_classes == 2:
                # Binary classification: AUPRC as primary
                final_property_metrics[prop_name]["primary_metric"] = "auprc"
                final_property_metrics[prop_name]["secondary_metric"] = "macro_f1"
            else:
                # Multi-class: macro_f1 as primary
                final_property_metrics[prop_name]["primary_metric"] = "macro_f1"
                final_property_metrics[prop_name]["secondary_metric"] = "weighted_f1"

            # Compute confusion matrix if requested
            if compute_confusion and "confusion_matrix" in metrics:
                preds = np.array(metrics["all_predictions"])
                targets = np.array(metrics["all_targets"])
                num_classes = metrics["confusion_matrix"].shape[0]

                # Build confusion matrix
                for pred, target in zip(preds, targets):
                    metrics["confusion_matrix"][target, pred] += 1

                final_property_metrics[prop_name]["confusion_matrix"] = metrics["confusion_matrix"]

                # Compute per-class precision, recall, F1
                final_property_metrics[prop_name]["per_class_metrics"] = {}
                for class_idx in range(num_classes):
                    tp = metrics["confusion_matrix"][class_idx, class_idx]
                    fp = metrics["confusion_matrix"][:, class_idx].sum() - tp
                    fn = metrics["confusion_matrix"][class_idx, :].sum() - tp

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                    final_property_metrics[prop_name]["per_class_metrics"][class_idx] = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": metrics["confusion_matrix"][class_idx, :].sum()
                    }

        elif task == "sequence_regression":
            # Variable-length sequence regression
            n_valid = metrics["n_valid_total"]
            if n_valid > 0:
                mse_val = metrics["loss"] / n_valid
                final_property_metrics[prop_name]["loss"] = mse_val
                final_property_metrics[prop_name]["rmse"] = mse_val ** 0.5
                final_property_metrics[prop_name]["mae"] = metrics["mae"] / n_valid
            else:
                # No valid angles in entire dataset
                final_property_metrics[prop_name]["loss"] = 0.0
                final_property_metrics[prop_name]["rmse"] = 0.0
                final_property_metrics[prop_name]["mae"] = 0.0

        else:  # scalar regression
            # Average loss (MSE), compute RMSE and MAE
            mse_val = metrics["loss"] / metrics["count"]
            final_property_metrics[prop_name]["loss"] = mse_val
            final_property_metrics[prop_name]["rmse"] = mse_val ** 0.5
            final_property_metrics[prop_name]["mae"] = metrics["mae"] / metrics["count"]

    # Overall metrics (for backward compatibility)
    mse = eval_loss / len(dataloader.dataset)
    rmse = mse ** 0.5

    return mse, rmse, final_property_metrics


