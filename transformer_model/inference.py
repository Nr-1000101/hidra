#!/usr/bin/env python
# coding: utf-8

"""
General-purpose inference script for HIDRA hierarchical transformer models.

Automatically adapts to any model configuration, property set, and label types.
Supports both SMILES and SELFIES input formats.

Usage:
    python inference.py --model best_model.pt --smiles "c1ccccc1"
    python inference.py --model best_model.pt --smiles "C[C@H](N)C(=O)O" --verbose
    python inference.py --model best_model.pt --selfies "[C][C][C]"
"""

import argparse
import torch
from typing import Dict, Any, List, Tuple
from transformer import HierarchicalTransformer, SmilesTokenizer, SelfiesTokenizer, LabelEncoder


class MolecularPropertyPredictor:
    """Flexible predictor for molecular properties using trained HIDRA models."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize predictor from checkpoint.

        Args:
            model_path: Path to saved model checkpoint (.pt file)
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = torch.device(device)

        # Load checkpoint (need weights_only=False for LabelEncoder)
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract model configuration
        self.vocab_size = self.checkpoint['vocab_size']
        self.max_len = self.checkpoint['max_len']
        self.d_model = self.checkpoint['d_model']
        self.n_initial_blocks = self.checkpoint['n_initial_blocks']
        self.property_configs = self.checkpoint['property_configs']
        self.label_encoder = self.checkpoint['label_encoder']

        # Detect mode from checkpoint (default to smiles if not stored)
        self.mode = self.checkpoint.get('mode', 'smiles')

        # Initialize tokenizer
        vocab_file = f"mol3d_data/{self.mode}_vocab.json"
        if self.mode == "smiles":
            self.tokenizer = SmilesTokenizer(vocab_file)
        else:
            self.tokenizer = SelfiesTokenizer(vocab_file)

        # Initialize model
        self.model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=self.n_initial_blocks,
            pad_idx=self.tokenizer.token_to_id["<pad>"]
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def _detect_label_type(self, prop_name: str, num_classes: int) -> str:
        """Detect if property uses boolean or encoded labels.

        Args:
            prop_name: Property name
            num_classes: Number of classes

        Returns:
            "boolean" if binary without encoding, "encoded" otherwise
        """
        # Chirality is always boolean (0/1) without label encoding
        if prop_name == "chirality" and num_classes == 2:
            return "boolean"

        # Check if label encoder has mappings for this property
        if prop_name in self.label_encoder.label_to_idx:
            return "encoded"

        # Default to boolean for binary classification without encoding
        if num_classes == 2:
            return "boolean"

        return "encoded"

    def _format_boolean_output(self, prop_name: str, pred_idx: int, probs: torch.Tensor) -> Dict[str, Any]:
        """Format output for boolean properties.

        Args:
            prop_name: Property name
            pred_idx: Predicted class index
            probs: Class probabilities

        Returns:
            Formatted prediction dictionary
        """
        # Map common boolean properties
        boolean_labels = {
            "chirality": {0: "Non-chiral", 1: "Chiral"},
        }

        if prop_name in boolean_labels:
            labels = boolean_labels[prop_name]
            pred_label = labels[pred_idx]
            prob_dict = {labels[i]: probs[0, i].item() for i in range(len(labels))}
        else:
            # Generic boolean handling
            pred_label = f"Class {pred_idx}"
            prob_dict = {f"Class {i}": probs[0, i].item() for i in range(probs.size(1))}

        return {
            "predicted_class": pred_idx,
            "predicted_label": pred_label,
            "confidence": probs.max().item(),
            "probabilities": prob_dict
        }

    def _format_encoded_output(self, prop_name: str, pred_idx: int, probs: torch.Tensor) -> Dict[str, Any]:
        """Format output for label-encoded properties.

        Args:
            prop_name: Property name
            pred_idx: Predicted class index
            probs: Class probabilities

        Returns:
            Formatted prediction dictionary
        """
        pred_label = self.label_encoder.inverse_transform(prop_name, pred_idx)

        # Build probability dictionary with labels
        prob_dict = {}
        if prop_name in self.label_encoder.idx_to_label:
            for idx in range(probs.size(1)):
                label = self.label_encoder.inverse_transform(prop_name, idx)
                prob_dict[label] = probs[0, idx].item()
        else:
            # Fallback to indices if no labels available
            for idx in range(probs.size(1)):
                prob_dict[f"Class {idx}"] = probs[0, idx].item()

        return {
            "predicted_class": pred_idx,
            "predicted_label": pred_label,
            "confidence": probs.max().item(),
            "probabilities": prob_dict
        }

    def _format_regression_output(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Format output for regression properties.

        Args:
            logits: Model output logits

        Returns:
            Formatted prediction dictionary
        """
        pred_value = logits.squeeze().item()
        return {
            "predicted_value": pred_value
        }

    def predict(self, molecule: str, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """Predict properties for a molecule.

        Args:
            molecule: SMILES or SELFIES string
            verbose: Whether to print detailed output

        Returns:
            Dictionary mapping property names to prediction results
        """
        # Tokenize
        tokens = self.tokenizer.encode(molecule, add_special=True)

        if verbose:
            print(f"Input molecule: {molecule}")
            print(f"Tokenized length: {len(tokens)}")

        # Pad to max_len
        pad_id = self.tokenizer.token_to_id["<pad>"]
        if len(tokens) < self.max_len:
            tokens = tokens + [pad_id] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        # Create tensors
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = (input_ids != pad_id).long()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        # Format predictions for each property
        predictions = {}
        for prop_name, logits in outputs.items():
            prop_cfg = next(cfg for cfg in self.property_configs if cfg['name'] == prop_name)

            if prop_cfg['task'] == 'classification':
                pred_idx = logits.argmax(dim=-1).item()
                probs = torch.softmax(logits, dim=-1)

                # Detect label type and format accordingly
                num_classes = prop_cfg.get('num_classes', 2)
                label_type = self._detect_label_type(prop_name, num_classes)

                if label_type == "boolean":
                    predictions[prop_name] = self._format_boolean_output(prop_name, pred_idx, probs)
                else:
                    predictions[prop_name] = self._format_encoded_output(prop_name, pred_idx, probs)

                # Add raw logits if verbose
                if verbose:
                    predictions[prop_name]["raw_logits"] = logits.squeeze().tolist()

            else:  # regression
                predictions[prop_name] = self._format_regression_output(logits)

                if verbose:
                    predictions[prop_name]["raw_output"] = logits.squeeze().item()

        return predictions

    def print_model_info(self):
        """Print model configuration information."""
        print("=" * 60)
        print("MODEL CONFIGURATION")
        print("=" * 60)
        print(f"Mode: {self.mode.upper()}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max sequence length: {self.max_len}")
        print(f"Model dimension: {self.d_model}")
        print(f"Initial encoder blocks: {self.n_initial_blocks}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"\nPredicted properties ({len(self.property_configs)}):")
        for i, cfg in enumerate(self.property_configs, 1):
            task = cfg['task']
            n_blocks = cfg.get('n_blocks', 'N/A')
            cross_attn = cfg.get('use_cross_attention', False)

            if task == 'classification':
                num_classes = cfg.get('num_classes', 'N/A')
                print(f"  {i}. {cfg['name']}: {task} ({num_classes} classes), "
                      f"{n_blocks} blocks, cross-attn={cross_attn}")
            else:
                print(f"  {i}. {cfg['name']}: {task}, "
                      f"{n_blocks} blocks, cross-attn={cross_attn}")
        print("=" * 60)

    def print_predictions(self, predictions: Dict[str, Dict[str, Any]], show_all_probs: bool = False):
        """Print formatted predictions.

        Args:
            predictions: Prediction dictionary from predict()
            show_all_probs: Whether to show all class probabilities
        """
        print("\n" + "=" * 60)
        print("PREDICTIONS")
        print("=" * 60)

        for prop_name, result in predictions.items():
            print(f"\n{prop_name.upper().replace('_', ' ')}:")

            if "predicted_label" in result:  # Classification
                print(f"  Prediction: {result['predicted_label']}")
                print(f"  Confidence: {result['confidence']:.2%}")

                if show_all_probs and "probabilities" in result:
                    print(f"  All probabilities:")
                    for label, prob in sorted(result['probabilities'].items(),
                                             key=lambda x: x[1], reverse=True):
                        print(f"    {label}: {prob:.4f} ({prob:.2%})")

            elif "predicted_value" in result:  # Regression
                print(f"  Predicted value: {result['predicted_value']:.6f}")

        print("=" * 60)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Predict molecular properties using HIDRA transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict properties for benzene
  python inference.py --model best_model.pt --smiles "c1ccccc1"

  # Predict for chiral molecule with verbose output
  python inference.py --model best_model.pt --smiles "C[C@H](N)C(=O)O" --verbose

  # Show all class probabilities
  python inference.py --model best_model.pt --smiles "CCO" --show-all-probs

  # Use SELFIES input
  python inference.py --model best_model.pt --selfies "[C][C][O]"
        """
    )

    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--smiles", help="SMILES string to predict")
    parser.add_argument("--selfies", help="SELFIES string to predict")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--show-all-probs", action="store_true",
                       help="Show all class probabilities for classification tasks")
    parser.add_argument("--quiet", action="store_true", help="Suppress model info output")

    args = parser.parse_args()

    # Validate input
    if not args.smiles and not args.selfies:
        parser.error("Either --smiles or --selfies must be provided")
    if args.smiles and args.selfies:
        parser.error("Cannot specify both --smiles and --selfies")

    molecule = args.smiles if args.smiles else args.selfies

    # Initialize predictor
    print(f"Loading model from {args.model}...")
    predictor = MolecularPropertyPredictor(args.model, device=args.device)

    if not args.quiet:
        predictor.print_model_info()

    # Run prediction
    print(f"\nRunning inference on: {molecule}")
    predictions = predictor.predict(molecule, verbose=args.verbose)

    # Print results
    predictor.print_predictions(predictions, show_all_probs=args.show_all_probs)


if __name__ == "__main__":
    main()
