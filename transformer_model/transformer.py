#!/usr/bin/env python
# coding: utf-8

"""
HIDRA: HIerarchical DFT accuracy transformer model to Reconstruct Atomic geometry

Main training script for molecular property prediction from SMILES/SELFIES sequences.

Architecture:
    1. Token + positional embeddings
    2. Stack of N encoder blocks (shared across all predictions)
    3. Prediction heads attach to specific encoder blocks:
       - Multiple heads can attach to the same block (parallel multi-task)
       - Heads can attach to different blocks (hierarchical prediction)
       - Each head is a lightweight linear layer (classification or regression)
    4. Optional cross-attention feedback:
       - Heads can embed their predictions
       - Subsequent encoder blocks cross-attend to prediction embeddings
       - Enables information flow from earlier predictions to later encoders

Configuration modes:
    - Default: N blocks → all heads at last block (standard multi-task)
    - Parallel: N blocks → all heads at same block (shared representation)
    - Hierarchical: Heads at different blocks with cross-attention feedback
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import all components from the package
from tokenizers import LabelEncoder, SelfiesTokenizer, SmilesTokenizer
from dataset import H5SequenceDataset, collate_fn
from model import HierarchicalTransformer
from metrics import FocalLoss, compute_class_weights
from early_stopping import EarlyStoppingMonitor
from training import train_one_epoch, evaluate

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Main Training Script ---
def main():
    """Main training loop with flexible attachment point architecture."""
    parser = argparse.ArgumentParser(description="Train Flexible Transformer for molecular property prediction")
    parser.add_argument("--mol_files", nargs='+', required=True, help="HDF5 files with molecules")
    parser.add_argument("--feat_files", nargs='+', required=True, help="HDF5 files with features")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="Learning rate (base: 1e-4 for batch_size=64; scale with: lr * (batch_size/64) / sqrt(n_encoder_blocks))")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_encoder_blocks", type=int, default=4, help="Number of encoder blocks in the stack")
    parser.add_argument("--mode", choices=["smiles", "selfies"], default="smiles", help="Input format")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--properties",
        nargs='+',
        default=["dimension", "ring_count", "chirality", "n_symmetry_planes",
                 "point_group", "planar_fit_error", "ring_plane_angles"],
        help="Properties to predict (subset of: dimension, ring_count, chirality, n_symmetry_planes, point_group, planar_fit_error, ring_plane_angles)"
    )
    parser.add_argument(
        "--attach_at_block",
        type=int,
        default=None,
        help="Default block index to attach prediction heads (default: last block). Can be overridden per property."
    )
    parser.add_argument(
        "--property_attach_blocks",
        nargs='*',
        default=[],
        help="Per-property attachment blocks as 'property:block' pairs (e.g., dimension:0 ring_count:1). Overrides --attach_at_block for specified properties."
    )
    parser.add_argument(
        "--enable_cross_attention",
        action="store_true",
        help="Enable cross-attention feedback from prediction embeddings (default: False)"
    )
    parser.add_argument(
        "--max_molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to load (default: None = all molecules)"
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "onecycle", "plateau"],
        default="none",
        help="Learning rate scheduler: none (constant), onecycle (warmup+cosine), plateau (adaptive)"
    )
    parser.add_argument(
        "--warmup_pct",
        type=float,
        default=0.1,
        help="Warmup percentage for onecycle scheduler (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--loss_fn",
        choices=["crossentropy", "focal"],
        default="crossentropy",
        help="Loss function for classification: crossentropy (standard) or focal (for imbalanced data)"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (focusing on hard examples, default: 2.0)"
    )
    parser.add_argument(
        "--focal_alpha",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Focal loss alpha (class weights): 'auto' uses computed class weights, 'none' uses uniform weights"
    )
    parser.add_argument(
        "--task_loss_weights",
        type=str,
        default="",
        help="Per-task loss weights for multi-task balancing (space-separated property:weight pairs, e.g., 'dimension:2.0 chirality:1.5'). Empty = uniform weights (1.0 for all)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory to save best_model.pt and test outputs (default: current directory)"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop if no improvement in validation metric for N epochs (0=disable, default: 10)"
    )
    parser.add_argument(
        "--early_stop_overfit_patience",
        type=int,
        default=5,
        help="Stop if overfitting detected for N consecutive epochs (0=disable, default: 5)"
    )
    parser.add_argument(
        "--early_stop_warmup",
        type=int,
        default=5,
        help="Number of initial epochs to check for training instability (default: 5)"
    )
    parser.add_argument(
        "--early_stop_instability_threshold",
        type=float,
        default=7.0,
        help="Stop if loss change exceeds N× standard deviation in warmup period (default: 7.0)"
    )
    parser.add_argument(
        "--onecycle_div_factor",
        type=float,
        default=10.0,
        help="OneCycleLR initial LR divisor: initial_lr = max_lr / div_factor (default: 10.0)"
    )
    parser.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=100.0,
        help="OneCycleLR final LR divisor: final_lr = max_lr / final_div_factor (default: 100.0)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file (best_model.pt) to resume training from (default: None = train from scratch)"
    )
    parser.add_argument(
        "--additional_epochs",
        type=int,
        default=None,
        help="Number of additional epochs to train when resuming (default: None = use --epochs as total)"
    )
    parser.add_argument(
        "--onecycle_resume_mode",
        type=str,
        default="restart",
        choices=["continue", "restart"],
        help="OneCycleLR resume behavior: 'continue' = preserve exact schedule state, 'restart' = create new schedule for remaining epochs (default: restart)"
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device is not None 
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    # Initialize tokenizer
    vocab_file = f"mol3d_data/{args.mode}_vocab.json"
    if args.mode == "smiles":
        tokenizer = SmilesTokenizer(vocab_file)
    else:
        tokenizer = SelfiesTokenizer(vocab_file)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Initialize label encoder and fit on all data
    label_encoder = LabelEncoder()
    print("Fitting label encoder...")
    
    # Collect all labels from all files
    all_dimensions = []
    all_ring_counts = []
    all_symmetry_planes = []
    all_point_groups = []
    
    with open("mol3d_data/underrepresented_data.json", "r") as f:
        underrep = json.load(f)
    
    for feat_file in args.feat_files:
        with h5py.File(feat_file, "r") as f:
            all_dimensions.extend([d.decode("utf-8") for d in f["dimensions"][:]])
            
            # Point groups with "Other" mapping
            for pg in f["point_groups"][:]:
                pg_str = pg.decode("utf-8")
                if pg_str in underrep["point_groups"]:
                    all_point_groups.append("Other")
                else:
                    all_point_groups.append(pg_str)
            
            # Symmetry planes with binning
            for sp in f["symmetry_planes"][:]:
                sp_int = int(sp)
                thresh = underrep["symmetry_planes"]
                all_symmetry_planes.append(str(sp_int) if sp_int < thresh else f"{thresh}+")
            
            # Ring counts with binning
            for rc in f["nrings"][:]:
                thresh = underrep["nrings"]
                all_ring_counts.append(str(rc) if rc < thresh else f"{thresh}+")
    
    label_encoder.fit("dimension", all_dimensions)
    label_encoder.fit("ring_count", all_ring_counts)
    label_encoder.fit("n_symmetry_planes", all_symmetry_planes)
    label_encoder.fit("point_group", all_point_groups)
    
    print(f"  Dimension classes: {label_encoder.get_num_classes('dimension')}")
    print(f"  Ring count classes: {label_encoder.get_num_classes('ring_count')}")
    print(f"  Symmetry plane classes: {label_encoder.get_num_classes('n_symmetry_planes')}")
    print(f"  Point group classes: {label_encoder.get_num_classes('point_group')}")

    # Create dataset
    dataset = H5SequenceDataset(
        args.mol_files,
        args.feat_files,
        tokenizer,
        label_encoder,
        mode=args.mode,
        max_len=args.max_len,
        max_molecules=args.max_molecules
    )

    # Split dataset
    n = len(dataset)
    ntrain = int(0.8 * n)
    nval = int(0.1 * n)
    train_ds, val_ds, test_ds = random_split(dataset, [ntrain, nval, n - ntrain - nval])

    # Optimized DataLoader with pin_memory and multiple workers for faster data transfer to GPU
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                           num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True)

    print(f"Dataset: {ntrain} train, {nval} val, {n - ntrain - nval} test")

    # Define property configurations dynamically based on --properties argument
    all_property_info = {
        "dimension": {
            "task": "classification",
            "num_classes": label_encoder.get_num_classes("dimension")
        },
        "ring_count": {
            "task": "classification",
            "num_classes": label_encoder.get_num_classes("ring_count")
        },
        "chirality": {
            "task": "classification",
            "num_classes": 2
        },
        "n_symmetry_planes": {
            "task": "classification",
            "num_classes": label_encoder.get_num_classes("n_symmetry_planes")
        },
        "point_group": {
            "task": "classification",
            "num_classes": label_encoder.get_num_classes("point_group")
        },
        "planar_fit_error": {
            "task": "sequence_regression",
            "max_seq_len": 15  # Max for 15 rings: one error per ring
        },
        "ring_plane_angles": {
            "task": "sequence_regression",
            "max_seq_len": 105  # Max for 15 rings: 15*14/2 = 105 angles
        }
    }

    # Validate requested properties
    for prop in args.properties:
        if prop not in all_property_info:
            raise ValueError(f"Unknown property: {prop}. Valid properties: {list(all_property_info.keys())}")

    # Determine default attachment block (last block if not specified)
    # Block indices are 0-indexed, so last block is n_encoder_blocks - 1
    default_attach_block = args.attach_at_block if args.attach_at_block is not None else args.n_encoder_blocks - 1

    # Parse per-property attachment blocks
    property_attach_map = {}
    for mapping in args.property_attach_blocks:
        if ':' not in mapping:
            raise ValueError(f"Invalid property_attach_blocks format: '{mapping}'. Expected 'property:block' (e.g., dimension:0)")
        prop_name, block_str = mapping.split(':', 1)
        try:
            block_idx = int(block_str)
        except ValueError:
            raise ValueError(f"Invalid block index in '{mapping}'. Expected integer after colon.")

        # Validate block index is within valid range
        if block_idx < 0 or block_idx >= args.n_encoder_blocks:
            raise ValueError(
                f"Property '{prop_name}' has attach block {block_idx}, "
                f"but must be in range [0, {args.n_encoder_blocks - 1}]"
            )

        property_attach_map[prop_name] = block_idx

    # Build property configs for selected properties
    # Each config specifies: task type, attachment point, cross-attention flag, num_classes (if classification)
    property_configs = []
    for i, prop_name in enumerate(args.properties):
        prop_info = all_property_info[prop_name]

        # Use per-property attachment if specified, otherwise use default
        attach_block = property_attach_map.get(prop_name, default_attach_block)

        config = {
            "name": prop_name,
            "task": prop_info["task"],  # "classification" or "regression"
            "attach_at_block": attach_block,  # Which encoder block output to attach head to
            "use_cross_attention": args.enable_cross_attention  # Whether to embed prediction for feedback
        }
        if "num_classes" in prop_info:
            config["num_classes"] = prop_info["num_classes"]  # Only for classification tasks
        property_configs.append(config)

    print(f"\nTraining properties: {args.properties}")
    print(f"Encoder blocks: {args.n_encoder_blocks}")
    print(f"Default attachment block: {default_attach_block}")
    if property_attach_map:
        print(f"Per-property attachments: {property_attach_map}")
    print(f"Cross-attention enabled: {args.enable_cross_attention}")

    # Create model
    model = HierarchicalTransformer(
        vocab_size=tokenizer.vocab_size,
        property_configs=property_configs,
        n_encoder_blocks=args.n_encoder_blocks,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=8,
        pad_idx=tokenizer.token_to_id["<pad>"]
    )
    model.to(device)

    # Store loss function configuration in model for later use
    model.loss_fn_type = args.loss_fn
    model.focal_gamma = args.focal_gamma if args.loss_fn == "focal" else None

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Apply learning rate scaling formula: LR = (base_lr * batch_size / 64) / sqrt(n_encoder_blocks)
    # This scales LR based on batch size and model depth for optimal training
    import math
    batch_scale = args.batch_size / 64
    depth_scale = 1.0 / math.sqrt(args.n_encoder_blocks)
    scaled_lr = args.lr * batch_scale * depth_scale

    print(f"\nLearning rate configuration:")
    print(f"  Base LR (--lr): {args.lr:.6e}")
    print(f"  Batch size: {args.batch_size} (scale factor: {batch_scale:.2f}×)")
    print(f"  Encoder blocks: {args.n_encoder_blocks} (depth scale: {depth_scale:.4f}×)")
    print(f"  Final scaled LR: {scaled_lr:.6e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_metric_init = float('inf') if len(property_configs) == 1 and property_configs[0]["task"] == "regression" else 0.0

    if args.resume_from is not None:
        import os
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from}")

        print(f"\nLoading checkpoint from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)

        # Validate checkpoint compatibility
        if checkpoint["vocab_size"] != tokenizer.vocab_size:
            raise ValueError(f"Vocab size mismatch: checkpoint={checkpoint['vocab_size']}, current={tokenizer.vocab_size}")
        if checkpoint["n_encoder_blocks"] != args.n_encoder_blocks:
            raise ValueError(f"Encoder blocks mismatch: checkpoint={checkpoint['n_encoder_blocks']}, current={args.n_encoder_blocks}")
        if checkpoint["d_model"] != args.d_model:
            raise ValueError(f"Model dimension mismatch: checkpoint={checkpoint['d_model']}, current={args.d_model}")

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if available (backward compatibility)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"  Loaded optimizer state")
        else:
            print(f"  Warning: Old checkpoint format detected (no optimizer state), starting optimizer from scratch")

        # Get epoch information (backward compatibility)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"  Checkpoint was saved at epoch {checkpoint['epoch']}")
        else:
            print(f"  Warning: Old checkpoint format detected (no epoch info), assuming epoch 0")
            start_epoch = 1

        best_val_metric_init = checkpoint.get("best_val_metric", best_val_metric_init)

        # Update learning rate if user provided a different one
        if args.lr != checkpoint["training_args"]["base_lr"]:
            print(f"  Warning: Overriding checkpoint LR {checkpoint['training_args']['scaled_lr']:.6e} with new scaled LR {scaled_lr:.6e}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = scaled_lr

        # Determine total epochs
        if args.additional_epochs is not None:
            total_epochs = start_epoch - 1 + args.additional_epochs
            print(f"  Resuming from epoch {start_epoch} (completed {start_epoch - 1} epochs)")
            print(f"  Training for {args.additional_epochs} additional epochs (total: {total_epochs} epochs)")
        else:
            total_epochs = args.epochs
            remaining_epochs = total_epochs - (start_epoch - 1)
            print(f"  Resuming from epoch {start_epoch} (completed {start_epoch - 1} epochs)")
            print(f"  Training until epoch {total_epochs} ({remaining_epochs} remaining epochs)")

        # Update args.epochs to reflect total training epochs
        args.epochs = total_epochs

        print(f"  Best validation metric from checkpoint: {best_val_metric_init:.4f}")
        print(f"  Loaded optimizer state and model weights")
    else:
        print("\nTraining from scratch")

    # Determine best model selection criterion
    # For single property training, use task-appropriate metric
    # For multiple properties, use weighted average
    if len(property_configs) == 1:
        prop_cfg = property_configs[0]
        if prop_cfg["task"] == "classification":
            num_classes = prop_cfg["num_classes"]
            if num_classes == 2:
                # Binary: maximize AUPRC
                best_val_metric = best_val_metric_init if args.resume_from else 0.0
                best_metric_name = "val_AUPRC"
                maximize_metric = True
            else:
                # Multi-class: maximize macro_F1
                best_val_metric = best_val_metric_init if args.resume_from else 0.0
                best_metric_name = "val_macro_F1"
                maximize_metric = True
        else:
            # Regression: minimize RMSE
            best_val_metric = best_val_metric_init if args.resume_from else float("inf")
            best_metric_name = "val_RMSE"
            maximize_metric = False
    else:
        # Multiple properties: use RMSE for now (backward compatible)
        best_val_metric = best_val_metric_init if args.resume_from else float("inf")
        best_metric_name = "val_RMSE"
        maximize_metric = False

    # Compute class weights for imbalanced classification tasks
    print(f"\nComputing class weights for imbalanced classification (loss_fn: {args.loss_fn})...")
    class_weights = {}
    for prop_cfg in property_configs:
        if prop_cfg["task"] == "classification":
            prop_name = prop_cfg["name"]
            num_classes = prop_cfg["num_classes"]

            # Compute weights if using with loss function
            if args.loss_fn == "focal" and args.focal_alpha == "none":
                # Focal loss without alpha (uniform weights)
                print(f"  {prop_name}: Using focal loss with gamma={args.focal_gamma}, no class weights")
            else:
                # Compute class weights for CrossEntropyLoss or Focal Loss with alpha
                weights = compute_class_weights(train_loader, prop_name, num_classes, device)
                class_weights[prop_name] = weights
                if args.loss_fn == "focal":
                    print(f"  {prop_name}: Using focal loss with gamma={args.focal_gamma}, alpha weights: min={weights.min():.4f}, max={weights.max():.4f}, ratio={weights.max()/weights.min():.1f}:1")
                else:
                    print(f"  {prop_name}: Using CrossEntropyLoss with weights: min={weights.min():.4f}, max={weights.max():.4f}, ratio={weights.max()/weights.min():.1f}:1")

    # Parse task-level loss weights for multi-task balancing
    task_loss_weights = {}
    if args.task_loss_weights:
        print(f"\nParsing task-level loss weights for multi-task balancing...")
        for pair in args.task_loss_weights.split():
            prop_name, weight_str = pair.split(":")
            task_loss_weights[prop_name] = float(weight_str)
            print(f"  {prop_name}: weight={float(weight_str):.2f}")
        print(f"  Total expected weighted loss (estimated): ~{sum(task_loss_weights.values()) * 0.14:.2f}")
    else:
        task_loss_weights = None

    # Create learning rate scheduler
    scheduler = None
    if args.scheduler == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR

        # Determine epochs for OneCycleLR scheduler
        # If resuming with restart mode, use remaining epochs; otherwise use total epochs
        if args.resume_from is not None and args.onecycle_resume_mode == "restart":
            # Restart mode: create new schedule for remaining epochs only
            scheduler_epochs = args.epochs - start_epoch
            print(f"\nOneCycleLR restart mode: Creating new schedule for {scheduler_epochs} remaining epochs")
        else:
            # Continue mode or training from scratch: use total epochs
            scheduler_epochs = args.epochs

        scheduler = OneCycleLR(
            optimizer,
            max_lr=scaled_lr,
            epochs=scheduler_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.warmup_pct,
            anneal_strategy='cos',
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor
        )
        print(f"\nUsing OneCycleLR scheduler:")
        print(f"  Warmup: {args.warmup_pct*100:.0f}% of training")
        print(f"  Max LR: {scaled_lr:.2e}")
        print(f"  Initial LR: {scaled_lr/args.onecycle_div_factor:.2e}")
        print(f"  Final LR: {scaled_lr/args.onecycle_final_div_factor:.2e}")
        print(f"  Div factors: initial={args.onecycle_div_factor}, final={args.onecycle_final_div_factor}")
    elif args.scheduler == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        # Use correct mode based on metric direction
        plateau_mode = 'max' if maximize_metric else 'min'

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=True
        )
        print(f"\nUsing ReduceLROnPlateau scheduler:")
        print(f"  Mode: '{plateau_mode}' ({'higher is better' if maximize_metric else 'lower is better'})")
        print(f"  Factor: 0.5 (halve LR on plateau)")
        print(f"  Patience: 3 epochs")
        print(f"  Min LR: 1e-6")

        # Warn if early stopping patience is too low
        if args.early_stop_patience > 0 and args.early_stop_patience < 15:
            print(f"  ⚠️  WARNING: early_stop_patience={args.early_stop_patience} may be too low for plateau scheduler")
            print(f"      Plateau scheduler needs time to reduce LR and see improvement")
            print(f"      Recommended: --early_stop_patience 15 or higher")
    else:
        print(f"\nUsing constant learning rate: {scaled_lr:.2e}")

    # Load scheduler state if resuming and scheduler was used in checkpoint
    if args.resume_from is not None and scheduler is not None:
        # Only load scheduler state in "continue" mode for OneCycleLR
        if args.scheduler == "onecycle" and args.onecycle_resume_mode == "restart":
            print(f"  OneCycleLR restart mode: Starting fresh scheduler (not loading checkpoint state)")
        elif "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"  Loaded scheduler state from checkpoint")
        else:
            print(f"  Warning: Checkpoint has no scheduler state, starting scheduler from scratch")

    # Initialize early stopping monitor
    early_stopping = EarlyStoppingMonitor(
        patience=args.early_stop_patience,
        overfit_patience=args.early_stop_overfit_patience,
        warmup_epochs=args.early_stop_warmup,
        instability_threshold=args.early_stop_instability_threshold,
        maximize_metric=maximize_metric,
        min_delta=1e-4
    )

    print(f"\nEarly stopping configuration:")
    print(f"  Plateau patience: {args.early_stop_patience} epochs (0=disabled)")
    print(f"  Overfit patience: {args.early_stop_overfit_patience} epochs (0=disabled)")
    print(f"  Warmup period: {args.early_stop_warmup} epochs (instability checks)")
    print(f"  Instability threshold: {args.early_stop_instability_threshold}× std deviation")

    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        # Training with per-property loss tracking
        # Pass scheduler for OneCycleLR (steps per batch), class weights, and task loss weights
        train_loss, train_property_losses, is_healthy = train_one_epoch(
            model, train_loader, optimizer, device, property_configs, bf16=True,
            scheduler=scheduler if args.scheduler == "onecycle" else None,
            class_weights=class_weights,
            task_loss_weights=task_loss_weights
        )

        # Check training health (NaN/Inf detection)
        if not is_healthy:
            print("\n" + "="*70)
            print("TRAINING STOPPED: NaN/Inf detected in losses or gradients")
            print("="*70)
            print("This indicates severe numerical instability.")
            print("Recommendations:")
            print("  1. Reduce learning rate by 5-10×")
            print("  2. Enable gradient clipping (torch.nn.utils.clip_grad_norm_)")
            print("  3. Check for data quality issues (corrupted samples)")
            print("  4. Consider using mixed precision training with loss scaling")
            break

        # Evaluation with comprehensive metrics, class weights, and task loss weights
        val_mse, val_rmse, val_property_metrics = evaluate(
            model, val_loader, device, property_configs, bf16=True,
            class_weights=class_weights,
            compute_confusion=False,  # Don't compute confusion during training (saves time)
            task_loss_weights=task_loss_weights
        )

        # Print epoch summary with current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Check if any property is regression to determine if RMSE should be shown
        has_regression = any(prop_cfg["task"] == "regression" for prop_cfg in property_configs)

        if has_regression:
            print(f"\nEpoch {epoch:3d} | Train loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | LR: {current_lr:.2e}")
        else:
            print(f"\nEpoch {epoch:3d} | Train loss: {train_loss:.4f} | LR: {current_lr:.2e}")

        # Print per-property training losses
        print("  Training losses per property:")
        for prop_name, loss_val in train_property_losses.items():
            print(f"    {prop_name:20s}: {loss_val:.4f}")

        # Print per-property validation metrics
        print("  Validation metrics per property:")
        for prop_name, metrics in val_property_metrics.items():
            # Find task type
            task = None
            num_classes = None
            for prop_cfg in property_configs:
                if prop_cfg["name"] == prop_name:
                    task = prop_cfg["task"]
                    num_classes = prop_cfg.get("num_classes", None)
                    break

            if task == "classification":
                # Show comprehensive classification metrics
                if num_classes == 2:
                    # Binary: show AUPRC, macro_F1
                    print(f"    {prop_name:20s}: AUPRC={metrics.get('auprc', 0):.4f}, macro_F1={metrics.get('macro_f1', 0):.4f}, acc={metrics['accuracy']:.2f}%")
                else:
                    # Multi-class: show macro_F1, weighted_F1, macro_recall
                    print(f"    {prop_name:20s}: macro_F1={metrics.get('macro_f1', 0):.4f}, w_F1={metrics.get('weighted_f1', 0):.4f}, m_recall={metrics.get('macro_recall', 0):.4f}, acc={metrics['accuracy']:.2f}%")
            else:  # regression
                print(f"    {prop_name:20s}: rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}")

        # Get current validation metric for model selection
        if len(property_configs) == 1:
            prop_name = property_configs[0]["name"]
            if property_configs[0]["task"] == "classification":
                num_classes = property_configs[0]["num_classes"]
                if num_classes == 2:
                    current_val_metric = val_property_metrics[prop_name].get("auprc", 0.0)
                else:
                    current_val_metric = val_property_metrics[prop_name].get("macro_f1", 0.0)
            else:
                current_val_metric = val_rmse
        else:
            current_val_metric = val_rmse

        # Step ReduceLROnPlateau scheduler (steps per epoch based on validation metric)
        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(current_val_metric)

        # Check early stopping conditions
        should_stop = early_stopping.update(epoch, train_loss, current_val_metric)
        if should_stop:
            print("\n" + "="*70)
            print("EARLY STOPPING TRIGGERED")
            print("="*70)
            print(early_stopping.stop_reason)
            print("="*70)
            break

        # Save best model
        is_best = (current_val_metric > best_val_metric) if maximize_metric else (current_val_metric < best_val_metric)
        if is_best:
            best_val_metric = current_val_metric
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_metric": best_val_metric,
                "property_configs": property_configs,
                "vocab_size": tokenizer.vocab_size,
                "max_len": args.max_len,
                "d_model": args.d_model,
                "n_encoder_blocks": args.n_encoder_blocks,
                "label_encoder": label_encoder,
                "class_weights": class_weights,  # Save class weights for reproducibility
                "val_property_metrics": val_property_metrics,
                "scheduler_type": args.scheduler,
                "training_args": {
                    "base_lr": args.lr,
                    "scaled_lr": scaled_lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "warmup_pct": args.warmup_pct,
                    "n_encoder_blocks": args.n_encoder_blocks,
                }
            }
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            # Save to specified directory (creates directory if needed)
            import os
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(checkpoint, model_path)
            print(f"  -> Saved best model to {model_path} ({best_metric_name}: {current_val_metric:.4f})")

        # Save periodic checkpoint (last_checkpoint.pt) after every epoch
        # Allows resuming from exact point if job terminates early (e.g., 24h SLURM limit)
        last_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_metric": best_val_metric,
            "property_configs": property_configs,
            "vocab_size": tokenizer.vocab_size,
            "max_len": args.max_len,
            "d_model": args.d_model,
            "n_encoder_blocks": args.n_encoder_blocks,
            "label_encoder": label_encoder,
            "class_weights": class_weights,
            "val_property_metrics": val_property_metrics,
            "scheduler_type": args.scheduler,
            "training_args": {
                "base_lr": args.lr,
                "scaled_lr": scaled_lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "warmup_pct": args.warmup_pct,
                "n_encoder_blocks": args.n_encoder_blocks,
            }
        }
        if scheduler is not None:
            last_checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        os.makedirs(args.save_dir, exist_ok=True)
        last_checkpoint_path = os.path.join(args.save_dir, "last_checkpoint.pt")
        torch.save(last_checkpoint, last_checkpoint_path)

    # Final test evaluation with confusion matrices
    test_mse, test_rmse, test_property_metrics = evaluate(
        model, test_loader, device, property_configs, bf16=True,
        class_weights=class_weights,
        compute_confusion=True,  # Compute confusion matrices for test set
        task_loss_weights=task_loss_weights
    )

    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*70}")

    # Only show overall RMSE if there are regression tasks
    has_regression = any(prop_cfg["task"] == "regression" for prop_cfg in property_configs)
    if has_regression:
        print(f"Overall Test RMSE: {test_rmse:.4f} (MSE {test_mse:.6f})")

    print(f"\nPer-property test metrics:")
    for prop_name, metrics in test_property_metrics.items():
        # Find task type
        task = None
        num_classes = None
        for prop_cfg in property_configs:
            if prop_cfg["name"] == prop_name:
                task = prop_cfg["task"]
                num_classes = prop_cfg.get("num_classes", None)
                break

        if task == "classification":
            # Show comprehensive test metrics
            if num_classes == 2:
                # Binary classification
                print(f"  {prop_name:20s}: AUPRC={metrics.get('auprc', 0):.4f}, macro_F1={metrics.get('macro_f1', 0):.4f}, acc={metrics['accuracy']:.2f}%")
            else:
                # Multi-class
                print(f"  {prop_name:20s}: macro_F1={metrics.get('macro_f1', 0):.4f}, w_F1={metrics.get('weighted_f1', 0):.4f}, m_recall={metrics.get('macro_recall', 0):.4f}, acc={metrics['accuracy']:.2f}%")

            # Print confusion matrix and per-class metrics if available
            if "confusion_matrix" in metrics:
                print(f"\n  Confusion Matrix for {prop_name}:")
                cm = metrics["confusion_matrix"]
                print(f"    Shape: {cm.shape} (true × pred)")
                print(f"    Confusion matrix saved in test_property_metrics")

                if "per_class_metrics" in metrics:
                    print(f"\n  Per-class metrics for {prop_name}:")
                    print(f"    {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
                    print(f"    {'-'*60}")
                    for class_idx, class_metrics in metrics["per_class_metrics"].items():
                        print(f"    {class_idx:<8} {class_metrics['precision']:<12.4f} {class_metrics['recall']:<12.4f} {class_metrics['f1']:<12.4f} {int(class_metrics['support']):<10}")
                print()  # Extra newline for readability
        else:  # regression
            print(f"  {prop_name:20s}: rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}")

    # Save comprehensive test metrics to JSON (without confusion matrix - too large)
    test_metrics_summary = {}
    for prop_name, metrics in test_property_metrics.items():
        # Remove numpy arrays and non-JSON-serializable objects
        test_metrics_summary[prop_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
            if k not in ["confusion_matrix", "per_class_metrics", "all_predictions", "all_targets", "all_probabilities"]
        }

    import os
    test_metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics_summary, f, indent=2)
    print(f"\nTest metrics summary saved to: {test_metrics_path}")

    # Save confusion matrices and detailed per-class metrics to pickle
    if any("confusion_matrix" in m for m in test_property_metrics.values()):
        import pickle
        confusion_path = os.path.join(args.save_dir, "test_confusion_matrices.pkl")
        with open(confusion_path, "wb") as f:
            pickle.dump(test_property_metrics, f)
        print(f"Confusion matrices and per-class metrics saved to: {confusion_path}")
        print(f"Load with: import pickle; metrics = pickle.load(open('{confusion_path}', 'rb'))")

    print(f"\nBest validation {best_metric_name}: {best_val_metric:.4f}")
    print("Training complete!")

    # Clean up periodic checkpoint after successful completion
    # last_checkpoint.pt is only needed for resuming interrupted training
    last_checkpoint_path = os.path.join(args.save_dir, "last_checkpoint.pt")
    if os.path.exists(last_checkpoint_path):
        os.remove(last_checkpoint_path)
        print(f"Removed temporary checkpoint: {last_checkpoint_path}")


if __name__ == "__main__":
    main()
