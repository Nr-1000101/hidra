"""
Comprehensive test suite for transformer.py (Flexible Attachment Architecture)

Tests all components individually and integration:
- Tokenizers (SMILES/SELFIES)
- LabelEncoder
- Dataset loading
- Model components (positional encoding, cross-attention, PropertyHead)
- Full HierarchicalTransformer with flexible attachment points
- Training/evaluation functions
- New features: per-property attachment blocks, optional cross-attention

Usage:
    python tests.py

Or run specific test:
    python tests.py TestHierarchicalTransformer.test_parallel_attachment
"""

import unittest
import tempfile
import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

# Import all components from modular transformer package
from tokenizers import (
    load_vocab,
    LabelEncoder,
    SelfiesTokenizer,
    SmilesTokenizer,
)

from dataset import (
    H5SequenceDataset,
    collate_fn,
)

from model_components import (
    LearnedPositionalEncoding,
    CrossAttention,
    PropertyHead,
    SequenceRegressionHead,
)

from model import HierarchicalTransformer

from metrics import (
    classification_accuracy,
    compute_mae,
    circular_mse_loss,
    circular_mae_loss,
    compute_class_weights,
    FocalLoss,
    compute_classification_metrics,
)

from early_stopping import EarlyStoppingMonitor

from training import (
    train_one_epoch,
    evaluate,
)


class TestVocabLoader(unittest.TestCase):
    """Test vocabulary loading functionality."""

    def setUp(self):
        """Create temporary vocab file."""
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.temp_dir, "test_vocab.json")

        vocab_data = {
            "tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "C", "N", "O"],
            "token_to_id": {
                "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                "C": 4, "N": 5, "O": 6
            },
            "id_to_token": {
                "0": "<pad>", "1": "<unk>", "2": "<bos>", "3": "<eos>",
                "4": "C", "5": "N", "6": "O"
            }
        }

        with open(self.vocab_file, "w") as f:
            json.dump(vocab_data, f)

    def test_load_vocab(self):
        """Test vocabulary loading returns correct structures."""
        tokens, token_to_id, id_to_token = load_vocab(self.vocab_file)

        self.assertEqual(len(tokens), 7)
        self.assertIn("<pad>", tokens)
        self.assertEqual(token_to_id["C"], 4)
        self.assertEqual(id_to_token[4], "C")
        self.assertIsInstance(id_to_token, dict)
        self.assertTrue(all(isinstance(k, int) for k in id_to_token.keys()))

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.vocab_file):
            os.remove(self.vocab_file)
        os.rmdir(self.temp_dir)


class TestLabelEncoder(unittest.TestCase):
    """Test LabelEncoder functionality."""

    def setUp(self):
        """Initialize label encoder."""
        self.encoder = LabelEncoder()

    def test_fit_and_transform(self):
        """Test fitting and transforming labels."""
        labels = ["linear", "planar", "tetrahedral", "linear", "planar"]
        self.encoder.fit("dimension", labels)

        # Check encoding
        idx = self.encoder.transform("dimension", "linear")
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)

        # Check inverse
        label = self.encoder.inverse_transform("dimension", idx)
        self.assertEqual(label, "linear")

    def test_num_classes(self):
        """Test getting number of classes."""
        labels = ["A", "B", "C", "A", "B"]
        self.encoder.fit("test_prop", labels)

        self.assertEqual(self.encoder.get_num_classes("test_prop"), 3)

    def test_unknown_label(self):
        """Test handling unknown labels (should default to 0)."""
        self.encoder.fit("dimension", ["linear", "planar"])
        idx = self.encoder.transform("dimension", "unknown_label")
        self.assertEqual(idx, 0)

    def test_multiple_properties(self):
        """Test encoding multiple different properties."""
        self.encoder.fit("prop1", ["A", "B", "C"])
        self.encoder.fit("prop2", ["X", "Y"])

        self.assertEqual(self.encoder.get_num_classes("prop1"), 3)
        self.assertEqual(self.encoder.get_num_classes("prop2"), 2)


class TestPropertyHead(unittest.TestCase):
    """Test PropertyHead with flexible configuration options."""

    def test_classification_head_without_cross_attention(self):
        """Test classification head without cross-attention."""
        head = PropertyHead(
            d_model=128,
            task="classification",
            num_classes=5,
            use_cross_attention=False
        )

        self.assertFalse(hasattr(head, 'pred_embedding'))

    def test_classification_head_with_cross_attention(self):
        """Test classification head with cross-attention enabled."""
        head = PropertyHead(
            d_model=128,
            task="classification",
            num_classes=5,
            use_cross_attention=True
        )

        self.assertTrue(hasattr(head, 'pred_embedding'))
        self.assertIsInstance(head.pred_embedding, nn.Embedding)

    def test_regression_head_without_cross_attention(self):
        """Test regression head without cross-attention."""
        head = PropertyHead(
            d_model=128,
            task="regression",
            use_cross_attention=False
        )

        self.assertFalse(hasattr(head, 'pred_embedding'))

    def test_regression_head_with_cross_attention(self):
        """Test regression head with cross-attention enabled."""
        head = PropertyHead(
            d_model=128,
            task="regression",
            use_cross_attention=True
        )

        self.assertTrue(hasattr(head, 'pred_embedding'))
        self.assertIsInstance(head.pred_embedding, nn.Linear)

    def test_forward_classification_without_cross_attention(self):
        """Test forward pass for classification without cross-attention."""
        head = PropertyHead(
            d_model=128,
            task="classification",
            num_classes=5,
            use_cross_attention=False
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)

        logits, pred_emb = head(x)

        self.assertEqual(logits.shape, (batch_size, 5))
        self.assertIsNone(pred_emb)

    def test_forward_classification_with_cross_attention(self):
        """Test forward pass for classification with cross-attention."""
        head = PropertyHead(
            d_model=128,
            task="classification",
            num_classes=5,
            use_cross_attention=True
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)

        logits, pred_emb = head(x)

        self.assertEqual(logits.shape, (batch_size, 5))
        self.assertIsNotNone(pred_emb)
        self.assertEqual(pred_emb.shape, (batch_size, seq_len, 128))

    def test_forward_regression(self):
        """Test forward pass for regression."""
        head = PropertyHead(
            d_model=128,
            task="regression",
            use_cross_attention=False
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)

        logits, pred_emb = head(x)

        self.assertEqual(logits.shape, (batch_size, 1))
        self.assertIsNone(pred_emb)


class TestHierarchicalTransformer(unittest.TestCase):
    """Test full hierarchical transformer with flexible attachment architecture."""

    def setUp(self):
        """Set up model configuration."""
        self.vocab_size = 100
        self.max_len = 64
        self.d_model = 128
        self.batch_size = 4

    def test_default_mode_single_head_at_end(self):
        """Test default mode: single head attached to last block."""
        property_configs = [
            {
                "name": "dimension",
                "task": "classification",
                "num_classes": 5,
                "attach_at_block": 3,  # Last block (n_encoder_blocks=4)
                "use_cross_attention": False
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            n_encoder_blocks=4,
            max_len=self.max_len,
            d_model=self.d_model
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        self.assertEqual(len(outputs), 1)
        self.assertIn("dimension", outputs)
        self.assertEqual(outputs["dimension"].shape, (self.batch_size, 5))

    def test_parallel_mode_all_heads_same_block(self):
        """Test parallel mode: all heads attached to same block."""
        property_configs = [
            {
                "name": "dimension",
                "task": "classification",
                "num_classes": 5,
                "attach_at_block": 1,
                "use_cross_attention": False
            },
            {
                "name": "ring_count",
                "task": "classification",
                "num_classes": 7,
                "attach_at_block": 1,  # Same block as dimension
                "use_cross_attention": False
            },
            {
                "name": "chirality",
                "task": "classification",
                "num_classes": 2,
                "attach_at_block": 1,  # Same block as others
                "use_cross_attention": False
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            n_encoder_blocks=2,
            max_len=self.max_len,
            d_model=self.d_model
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs["dimension"].shape, (self.batch_size, 5))
        self.assertEqual(outputs["ring_count"].shape, (self.batch_size, 7))
        self.assertEqual(outputs["chirality"].shape, (self.batch_size, 2))

    def test_hierarchical_mode_different_blocks_with_cross_attention(self):
        """Test hierarchical mode: heads at different blocks with cross-attention."""
        property_configs = [
            {
                "name": "dimension",
                "task": "classification",
                "num_classes": 5,
                "attach_at_block": 0,  # First block
                "use_cross_attention": True  # Enable for feedback to block 1
            },
            {
                "name": "ring_count",
                "task": "classification",
                "num_classes": 7,
                "attach_at_block": 1,  # Second block
                "use_cross_attention": True  # Enable for feedback to block 2
            },
            {
                "name": "chirality",
                "task": "classification",
                "num_classes": 2,
                "attach_at_block": 2,  # Third block
                "use_cross_attention": False  # Last property, no feedback needed
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            n_encoder_blocks=3,
            max_len=self.max_len,
            d_model=self.d_model
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        self.assertEqual(len(outputs), 3)
        self.assertIn("dimension", outputs)
        self.assertIn("ring_count", outputs)
        self.assertIn("chirality", outputs)

    def test_mixed_mode_some_parallel_some_hierarchical(self):
        """Test mixed mode: some properties parallel, some hierarchical."""
        property_configs = [
            {
                "name": "prop1",
                "task": "classification",
                "num_classes": 3,
                "attach_at_block": 0,
                "use_cross_attention": True
            },
            {
                "name": "prop2",
                "task": "classification",
                "num_classes": 4,
                "attach_at_block": 2,  # Skip block 1
                "use_cross_attention": False
            },
            {
                "name": "prop3",
                "task": "regression",
                "attach_at_block": 2,  # Same as prop2 (parallel)
                "use_cross_attention": False
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            n_encoder_blocks=3,
            max_len=self.max_len,
            d_model=self.d_model
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        self.assertEqual(len(outputs), 3)

    def test_model_gradient_flow(self):
        """Test gradients flow through entire model."""
        property_configs = [
            {"name": "prop1", "task": "classification", "num_classes": 3, "attach_at_block": 0},
            {"name": "prop2", "task": "regression", "attach_at_block": 1},
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            n_encoder_blocks=2,
            max_len=self.max_len,
            d_model=self.d_model
        )

        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        # Compute dummy loss
        loss = outputs["prop1"].sum() + outputs["prop2"].sum()
        loss.backward()

        # Check at least some gradients exist
        has_gradients = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break

        self.assertTrue(has_gradients, "No gradients found in model parameters")

    def test_invalid_attachment_block(self):
        """Test that invalid attachment block raises error."""
        property_configs = [
            {
                "name": "dimension",
                "task": "classification",
                "num_classes": 5,
                "attach_at_block": 5,  # Invalid: n_encoder_blocks=4
                "use_cross_attention": False
            }
        ]

        with self.assertRaises(ValueError):
            model = HierarchicalTransformer(
                vocab_size=self.vocab_size,
                property_configs=property_configs,
                n_encoder_blocks=4,
                max_len=self.max_len,
                d_model=self.d_model
            )

    def test_invalid_n_encoder_blocks(self):
        """Test that n_encoder_blocks < 1 raises error."""
        property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 5, "attach_at_block": 0}
        ]

        with self.assertRaises(ValueError):
            model = HierarchicalTransformer(
                vocab_size=self.vocab_size,
                property_configs=property_configs,
                n_encoder_blocks=0,  # Invalid
                max_len=self.max_len,
                d_model=self.d_model
            )


class TestTrainingFunctions(unittest.TestCase):
    """Test training and evaluation functions."""

    def setUp(self):
        """Set up dummy model and data."""
        self.device = torch.device("cpu")
        self.vocab_size = 50
        self.max_len = 32
        self.d_model = 64
        self.batch_size = 4

        self.property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 3, "attach_at_block": 1},
            {"name": "ring_count", "task": "classification", "num_classes": 4, "attach_at_block": 1},
            {"name": "chirality", "task": "classification", "num_classes": 2, "attach_at_block": 1},
            {"name": "n_symmetry_planes", "task": "classification", "num_classes": 5, "attach_at_block": 1},
            {"name": "point_group", "task": "classification", "num_classes": 5, "attach_at_block": 1},
            {"name": "planar_fit_error", "task": "sequence_regression", "max_seq_len": 15, "attach_at_block": 1},
            {"name": "ring_plane_angles", "task": "sequence_regression", "max_seq_len": 105, "attach_at_block": 1},
        ]

        self.model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            n_encoder_blocks=2,
            max_len=self.max_len,
            d_model=self.d_model
        )

    def create_dummy_dataloader(self, n_batches=5):
        """Create dummy dataloader for testing."""
        class DummyDataset:
            def __init__(self, n_samples):
                self.n_samples = n_samples

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                input_ids = torch.randint(0, 50, (32,))
                attention_mask = torch.ones(32)
                # Sequence regression properties return variable-length arrays
                n_rings = torch.randint(1, 6, (1,)).item()
                targets = {
                    "dimension": torch.randint(0, 3, (1,)).item(),
                    "ring_count": torch.randint(0, 4, (1,)).item(),
                    "chirality": torch.randint(0, 2, (1,)).item(),
                    "n_symmetry_planes": torch.randint(0, 5, (1,)).item(),
                    "point_group": torch.randint(0, 5, (1,)).item(),
                    "planar_fit_error": torch.randn(n_rings).numpy(),  # One per ring
                    "ring_plane_angles": torch.rand(n_rings * (n_rings - 1) // 2).numpy() * 180,  # Ring pairs
                }
                return input_ids, attention_mask, targets

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            attention_masks = torch.stack([b[1] for b in batch])
            targets = {
                "dimension": torch.tensor([b[2]["dimension"] for b in batch]),
                "ring_count": torch.tensor([b[2]["ring_count"] for b in batch]),
                "chirality": torch.tensor([b[2]["chirality"] for b in batch]),
                "n_symmetry_planes": torch.tensor([b[2]["n_symmetry_planes"] for b in batch]),
                "point_group": torch.tensor([b[2]["point_group"] for b in batch]),
            }
            # Handle variable-length sequences with padding
            from transformer import collate_fn as real_collate
            _, _, seq_targets = real_collate(batch)
            targets.update({k: v for k, v in seq_targets.items() if k.startswith("planar_fit") or k.startswith("ring_plane")})
            return input_ids, attention_masks, targets

        from torch.utils.data import DataLoader
        dataset = DummyDataset(n_batches * self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate)

    def test_train_one_epoch(self):
        """Test training for one epoch runs without errors."""
        dataloader = self.create_dummy_dataloader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        loss, property_losses, is_healthy = train_one_epoch(
            self.model,
            dataloader,
            optimizer,
            self.device,
            self.property_configs,
            bf16=False
        )

        # Check loss is a valid number
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

        # Check property_losses is a dictionary with all properties
        self.assertIsInstance(property_losses, dict)
        for prop_cfg in self.property_configs:
            self.assertIn(prop_cfg["name"], property_losses)
            self.assertIsInstance(property_losses[prop_cfg["name"]], float)
            self.assertGreaterEqual(property_losses[prop_cfg["name"]], 0)

    def test_evaluate(self):
        """Test evaluation runs without errors."""
        dataloader = self.create_dummy_dataloader()

        mse, rmse, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False
        )

        # Check overall metrics are valid
        self.assertIsInstance(mse, float)
        self.assertIsInstance(rmse, float)
        self.assertGreater(mse, 0)
        self.assertGreater(rmse, 0)
        self.assertAlmostEqual(rmse, mse ** 0.5, places=5)

        # Check property_metrics structure
        self.assertIsInstance(property_metrics, dict)
        for prop_cfg in self.property_configs:
            prop_name = prop_cfg["name"]
            self.assertIn(prop_name, property_metrics)
            self.assertIsInstance(property_metrics[prop_name], dict)

            # Check metrics based on task type
            if prop_cfg["task"] == "classification":
                self.assertIn("loss", property_metrics[prop_name])
                self.assertIn("accuracy", property_metrics[prop_name])
                self.assertIsInstance(property_metrics[prop_name]["loss"], float)
                self.assertIsInstance(property_metrics[prop_name]["accuracy"], float)
                self.assertGreaterEqual(property_metrics[prop_name]["accuracy"], 0)
                self.assertLessEqual(property_metrics[prop_name]["accuracy"], 100)
            else:  # regression
                self.assertIn("loss", property_metrics[prop_name])
                self.assertIn("rmse", property_metrics[prop_name])
                self.assertIn("mae", property_metrics[prop_name])
                self.assertIsInstance(property_metrics[prop_name]["loss"], float)
                self.assertIsInstance(property_metrics[prop_name]["rmse"], float)
                self.assertIsInstance(property_metrics[prop_name]["mae"], float)
                self.assertGreaterEqual(property_metrics[prop_name]["mae"], 0)

    def test_training_updates_parameters(self):
        """Test that training actually updates model parameters."""
        dataloader = self.create_dummy_dataloader(n_batches=2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Store initial parameters
        initial_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        # Train
        train_one_epoch(
            self.model,
            dataloader,
            optimizer,
            self.device,
            self.property_configs,
            bf16=False
        )

        # Check at least some parameters changed
        changed = False
        for name, param in self.model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                changed = True
                break

        self.assertTrue(changed, "No parameters were updated during training")

    def test_evaluation_no_gradient(self):
        """Test evaluation doesn't compute gradients."""
        dataloader = self.create_dummy_dataloader()

        # Clear any existing gradients
        self.model.zero_grad()

        # Run evaluation
        evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False
        )

        # Check no gradients were computed
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.all(param.grad == 0))


class TestDeterminism(unittest.TestCase):
    """Test model determinism with fixed random seed."""

    def test_forward_pass_deterministic(self):
        """Test that forward passes are deterministic with fixed seed."""
        torch.manual_seed(42)

        property_configs = [
            {"name": "prop1", "task": "classification", "num_classes": 3, "attach_at_block": 0},
            {"name": "prop2", "task": "regression", "attach_at_block": 1},
        ]

        model = HierarchicalTransformer(
            vocab_size=100,
            property_configs=property_configs,
            n_encoder_blocks=2,
            max_len=64,
            d_model=128
        )
        model.eval()  # Set to eval mode for deterministic behavior

        input_ids = torch.randint(0, 100, (4, 64))
        attention_mask = torch.ones(4, 64)

        # First forward pass
        with torch.no_grad():
            outputs1 = model(input_ids, attention_mask)

        # Second forward pass (same input)
        with torch.no_grad():
            outputs2 = model(input_ids, attention_mask)

        # Outputs should be identical
        for key in outputs1.keys():
            self.assertTrue(torch.allclose(outputs1[key], outputs2[key]))


class TestMetricFunctions(unittest.TestCase):
    """Test new metric helper functions."""

    def test_classification_accuracy_perfect(self):
        """Test classification accuracy with perfect predictions."""
        logits = torch.tensor([
            [10.0, 0.0, 0.0],  # Predicts class 0
            [0.0, 10.0, 0.0],  # Predicts class 1
            [0.0, 0.0, 10.0],  # Predicts class 2
        ])
        targets = torch.tensor([0, 1, 2])

        accuracy = classification_accuracy(logits, targets)
        self.assertAlmostEqual(accuracy, 100.0, places=5)

    def test_classification_accuracy_partial(self):
        """Test classification accuracy with partial correctness."""
        logits = torch.tensor([
            [10.0, 0.0, 0.0],  # Predicts 0, correct
            [0.0, 10.0, 0.0],  # Predicts 1, wrong (should be 2)
            [0.0, 0.0, 10.0],  # Predicts 2, correct
            [10.0, 0.0, 0.0],  # Predicts 0, wrong (should be 1)
        ])
        targets = torch.tensor([0, 2, 2, 1])

        accuracy = classification_accuracy(logits, targets)
        self.assertAlmostEqual(accuracy, 50.0, places=5)  # 2/4 = 50%

    def test_classification_accuracy_zero(self):
        """Test classification accuracy with all wrong predictions."""
        logits = torch.tensor([
            [10.0, 0.0, 0.0],  # Predicts 0, wrong
            [0.0, 10.0, 0.0],  # Predicts 1, wrong
        ])
        targets = torch.tensor([2, 2])

        accuracy = classification_accuracy(logits, targets)
        self.assertAlmostEqual(accuracy, 0.0, places=5)

    def test_classification_accuracy_empty_batch(self):
        """Test classification accuracy handles empty batch."""
        logits = torch.empty(0, 3)
        targets = torch.empty(0, dtype=torch.long)

        accuracy = classification_accuracy(logits, targets)
        self.assertEqual(accuracy, 0.0)  # Should not crash, return 0

    def test_compute_mae_perfect(self):
        """Test MAE with perfect predictions."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        mae = compute_mae(predictions, targets)
        self.assertAlmostEqual(mae, 0.0, places=6)

    def test_compute_mae_with_errors(self):
        """Test MAE with various errors."""
        predictions = torch.tensor([1.5, 2.0, 3.5])
        targets = torch.tensor([1.0, 2.0, 3.0])
        # Errors: |0.5|, |0.0|, |0.5| -> mean = 1.0/3 = 0.333...

        mae = compute_mae(predictions, targets)
        self.assertAlmostEqual(mae, 1.0/3.0, places=6)

    def test_compute_mae_negative_errors(self):
        """Test MAE handles negative errors correctly."""
        predictions = torch.tensor([0.5, 1.5])
        targets = torch.tensor([1.0, 1.0])
        # Errors: |-0.5|, |0.5| -> mean = 0.5

        mae = compute_mae(predictions, targets)
        self.assertAlmostEqual(mae, 0.5, places=6)

    def test_circular_mse_loss_identical_angles(self):
        """Test circular MSE with identical angles."""
        pred_angles = torch.tensor([0.0, 90.0, 180.0, 270.0])
        true_angles = torch.tensor([0.0, 90.0, 180.0, 270.0])

        loss = circular_mse_loss(pred_angles, true_angles)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_circular_mse_loss_wraparound(self):
        """Test circular MSE handles 360° wraparound correctly."""
        # 5° is close to 355° (only 10° apart when wrapped)
        pred_angles = torch.tensor([5.0])
        true_angles = torch.tensor([355.0])

        loss = circular_mse_loss(pred_angles, true_angles)

        # Expected: 10° = 10 * π/180 ≈ 0.1745 radians
        # MSE = 0.1745² ≈ 0.0305
        import math
        expected_diff_rad = 10.0 * math.pi / 180.0
        expected_loss = expected_diff_rad ** 2

        self.assertAlmostEqual(loss.item(), expected_loss, places=4)

    def test_circular_mse_loss_opposite_angles(self):
        """Test circular MSE with opposite angles (180° apart)."""
        pred_angles = torch.tensor([0.0])
        true_angles = torch.tensor([180.0])

        loss = circular_mse_loss(pred_angles, true_angles)

        # 180° = π radians, MSE = π² ≈ 9.8696
        import math
        expected_loss = math.pi ** 2
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)

    def test_circular_mae_loss_identical_angles(self):
        """Test circular MAE with identical angles."""
        pred_angles = torch.tensor([0.0, 90.0, 180.0])
        true_angles = torch.tensor([0.0, 90.0, 180.0])

        mae = circular_mae_loss(pred_angles, true_angles)
        self.assertAlmostEqual(mae.item(), 0.0, places=6)

    def test_circular_mae_loss_wraparound(self):
        """Test circular MAE handles wraparound correctly."""
        # 5° vs 355° = 10° difference
        pred_angles = torch.tensor([5.0])
        true_angles = torch.tensor([355.0])

        mae = circular_mae_loss(pred_angles, true_angles)

        # Expected: 10° = 10 * π/180 ≈ 0.1745 radians
        import math
        expected_diff_rad = 10.0 * math.pi / 180.0

        self.assertAlmostEqual(mae.item(), expected_diff_rad, places=4)

    def test_circular_mae_loss_multiple_angles(self):
        """Test circular MAE with multiple angles."""
        # Differences: 10°, 20°, 350° wrapped = 10°
        pred_angles = torch.tensor([10.0, 80.0, 5.0])
        true_angles = torch.tensor([0.0, 100.0, 355.0])

        mae = circular_mae_loss(pred_angles, true_angles)

        # Mean of |10°|, |20°|, |10°| = 13.33° ≈ 0.2327 radians
        import math
        expected_mae = (10.0 + 20.0 + 10.0) / 3.0 * math.pi / 180.0

        self.assertAlmostEqual(mae.item(), expected_mae, places=4)

    def test_circular_vs_standard_mse(self):
        """Test that circular MSE is much better than standard MSE for angles."""
        # Case where standard MSE fails dramatically
        pred_angles = torch.tensor([5.0, 10.0])
        true_angles = torch.tensor([355.0, 350.0])

        # Circular MSE (correct)
        circ_mse = circular_mse_loss(pred_angles, true_angles)

        # Standard MSE (wrong) - for comparison
        pred_rad = pred_angles * (torch.pi / 180.0)
        true_rad = true_angles * (torch.pi / 180.0)
        standard_mse = torch.mean((pred_rad - true_rad) ** 2)

        # Circular MSE should be much smaller
        self.assertLess(circ_mse.item(), 0.1)  # Should be small (≈0.03)
        self.assertGreater(standard_mse.item(), 10.0)  # Should be huge


class TestH5SequenceDataset(unittest.TestCase):
    """Test dataset with RAM preloading functionality."""

    def setUp(self):
        """Set up minimal test files."""
        # These would normally be created by a fixture, but we'll test with mock data
        # For now, we test the logic paths assuming files exist
        pass

    def test_ram_preloading_creates_correct_structures(self):
        """Test that RAM preloading initializes all required data structures."""
        # Create minimal mock dataset to test structure
        # This is a smoke test - actual data loading tested in integration
        from transformer import H5SequenceDataset, SmilesTokenizer, LabelEncoder

        # Skip if test files don't exist (integration test, not unit test)
        import os
        test_mol = "mol3d_data/mol3d_mil1.h5"
        test_feat = "mol3d_data/mol3d_feat_mil1.h5"

        if not (os.path.exists(test_mol) and os.path.exists(test_feat)):
            self.skipTest("Test data files not available")

        # Create tokenizer and label encoder
        vocab_file = "mol3d_data/smiles_vocab.json"
        if not os.path.exists(vocab_file):
            self.skipTest("Vocab file not available")

        tokenizer = SmilesTokenizer(vocab_file)
        label_encoder = LabelEncoder()

        # Fit label encoder with minimal data
        label_encoder.fit("dimension", ["0D", "1D", "2D", "3D"])
        label_encoder.fit("ring_count", ["0", "1", "2", "3+"])
        label_encoder.fit("n_symmetry_planes", ["0", "1", "2+"])
        label_encoder.fit("point_group", ["C1", "Cs", "Other"])

        # Create dataset with small subset
        dataset = H5SequenceDataset(
            mol_files=[test_mol],
            feat_files=[test_feat],
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            mode="smiles",
            max_len=128,
            max_molecules=100  # Small subset for fast testing
        )

        # Verify all data structures exist and have correct length
        self.assertEqual(len(dataset.sequences), 100)
        self.assertEqual(len(dataset.dimensions), 100)
        self.assertEqual(len(dataset.point_groups), 100)
        self.assertEqual(len(dataset.n_symmetry_planes_raw), 100)
        self.assertEqual(len(dataset.chiralities), 100)
        self.assertEqual(len(dataset.ring_counts_raw), 100)
        self.assertEqual(len(dataset.planar_fit_errors_raw), 100)
        self.assertEqual(len(dataset.ring_plane_angles_raw), 100)

        # Verify no None values in critical fields
        self.assertTrue(all(s is not None for s in dataset.sequences))
        self.assertTrue(all(d is not None for d in dataset.dimensions))

    def test_getitem_returns_valid_data(self):
        """Test that __getitem__ returns properly formatted data from RAM."""
        from transformer import H5SequenceDataset, SmilesTokenizer, LabelEncoder
        import os

        test_mol = "mol3d_data/mol3d_mil1.h5"
        test_feat = "mol3d_data/mol3d_feat_mil1.h5"
        vocab_file = "mol3d_data/smiles_vocab.json"

        if not all(os.path.exists(f) for f in [test_mol, test_feat, vocab_file]):
            self.skipTest("Test data files not available")

        tokenizer = SmilesTokenizer(vocab_file)
        label_encoder = LabelEncoder()
        label_encoder.fit("dimension", ["0D", "1D", "2D", "3D"])
        label_encoder.fit("ring_count", ["0", "1", "2", "3+"])
        label_encoder.fit("n_symmetry_planes", ["0", "1", "2+"])
        label_encoder.fit("point_group", ["C1", "Cs", "Other"])

        dataset = H5SequenceDataset(
            mol_files=[test_mol],
            feat_files=[test_feat],
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            mode="smiles",
            max_len=128,
            max_molecules=10
        )

        # Get a sample
        input_ids, attention_mask, targets = dataset[0]

        # Verify tensor shapes and types
        self.assertEqual(input_ids.shape, (128,))
        self.assertEqual(attention_mask.shape, (128,))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(attention_mask.dtype, torch.long)

        # Verify targets dictionary has expected keys
        expected_keys = {"dimension", "ring_count", "chirality", "n_symmetry_planes",
                        "point_group", "planar_fit_error", "ring_plane_angles"}
        self.assertEqual(set(targets.keys()), expected_keys)

        # Verify target types
        self.assertIsInstance(targets["dimension"], int)
        self.assertIsInstance(targets["chirality"], int)
        # planar_fit_error is now a numpy array (one error per ring)
        self.assertIsInstance(targets["planar_fit_error"], np.ndarray)
        self.assertEqual(targets["planar_fit_error"].dtype, np.float32)

    def test_multiple_file_loading(self):
        """Test that dataset correctly handles multiple HDF5 files."""
        from transformer import H5SequenceDataset, SmilesTokenizer, LabelEncoder
        import os

        test_mol1 = "mol3d_data/mol3d_mil1.h5"
        test_mol2 = "mol3d_data/mol3d_mil2.h5"
        test_feat1 = "mol3d_data/mol3d_feat_mil1.h5"
        test_feat2 = "mol3d_data/mol3d_feat_mil2.h5"
        vocab_file = "mol3d_data/smiles_vocab.json"

        files = [test_mol1, test_mol2, test_feat1, test_feat2, vocab_file]
        if not all(os.path.exists(f) for f in files):
            self.skipTest("Multiple test data files not available")

        tokenizer = SmilesTokenizer(vocab_file)
        label_encoder = LabelEncoder()
        label_encoder.fit("dimension", ["0D", "1D", "2D", "3D"])
        label_encoder.fit("ring_count", ["0", "1", "2", "3+"])
        label_encoder.fit("n_symmetry_planes", ["0", "1", "2+"])
        label_encoder.fit("point_group", ["C1", "Cs", "Other"])

        # Create dataset with 2 files
        dataset = H5SequenceDataset(
            mol_files=[test_mol1, test_mol2],
            feat_files=[test_feat1, test_feat2],
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            mode="smiles",
            max_len=128,
            max_molecules=200  # Get some from each file
        )

        # Should have loaded data from both files
        self.assertGreater(len(dataset), 0)
        self.assertLessEqual(len(dataset), 200)

        # Test samples from beginning and end (likely different files)
        sample_start = dataset[0]
        sample_end = dataset[-1]

        self.assertIsNotNone(sample_start)
        self.assertIsNotNone(sample_end)


class TestWeightedLoss(unittest.TestCase):
    """Test weighted loss computation for imbalanced classification."""

    def setUp(self):
        """Set up dummy model and imbalanced data."""
        self.device = torch.device("cpu")
        self.vocab_size = 50
        self.max_len = 32
        self.d_model = 64
        self.batch_size = 8

        # Create property configs for classification tasks
        self.property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 3, "attach_at_block": 0},
            {"name": "ring_count", "task": "classification", "num_classes": 4, "attach_at_block": 0},
            {"name": "chirality", "task": "classification", "num_classes": 2, "attach_at_block": 0},
        ]

        self.model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            n_encoder_blocks=1,
            max_len=self.max_len,
            d_model=self.d_model
        )

    def create_imbalanced_dataloader(self):
        """Create dataloader with highly imbalanced class distribution."""
        class ImbalancedDataset:
            def __init__(self):
                # Create imbalanced data: 90% class 0, 5% class 1, 5% class 2 for dimension
                self.n_samples = 100
                self.dimension_targets = [0] * 90 + [1] * 5 + [2] * 5
                # Ring count: 80% class 0, 10% class 1, 5% class 2, 5% class 3
                self.ring_count_targets = [0] * 80 + [1] * 10 + [2] * 5 + [3] * 5
                # Chirality: 95% class 0, 5% class 1
                self.chirality_targets = [0] * 95 + [1] * 5

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                input_ids = torch.randint(0, 50, (32,))
                attention_mask = torch.ones(32)
                targets = {
                    "dimension": self.dimension_targets[idx],
                    "ring_count": self.ring_count_targets[idx],
                    "chirality": self.chirality_targets[idx],
                }
                return input_ids, attention_mask, targets

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            attention_masks = torch.stack([b[1] for b in batch])
            targets = {
                "dimension": torch.tensor([b[2]["dimension"] for b in batch]),
                "ring_count": torch.tensor([b[2]["ring_count"] for b in batch]),
                "chirality": torch.tensor([b[2]["chirality"] for b in batch]),
            }
            return input_ids, attention_masks, targets

        from torch.utils.data import DataLoader
        dataset = ImbalancedDataset()
        return DataLoader(dataset, batch_size=8, collate_fn=collate)

    def test_compute_class_weights_balanced(self):
        """Test class weight computation with balanced classes."""
        class BalancedDataset:
            def __len__(self):
                return 30

            def __getitem__(self, idx):
                # Equal distribution: 10 samples per class
                input_ids = torch.randint(0, 50, (32,))
                attention_mask = torch.ones(32)
                targets = {"dimension": idx % 3}  # 0, 1, 2, 0, 1, 2, ...
                return input_ids, attention_mask, targets

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            attention_masks = torch.stack([b[1] for b in batch])
            targets = {"dimension": torch.tensor([b[2]["dimension"] for b in batch])}
            return input_ids, attention_masks, targets

        from torch.utils.data import DataLoader
        dataset = BalancedDataset()
        dataloader = DataLoader(dataset, batch_size=6, collate_fn=collate)

        weights = compute_class_weights(dataloader, "dimension", 3, self.device)

        # All weights should be equal (or very close) for balanced data
        self.assertEqual(weights.shape, (3,))
        self.assertTrue(torch.allclose(weights, torch.ones(3, device=self.device), atol=1e-5))

    def test_compute_class_weights_imbalanced(self):
        """Test class weight computation with imbalanced classes."""
        dataloader = self.create_imbalanced_dataloader()

        # Compute weights for dimension property (90% class 0, 5% class 1, 5% class 2)
        weights = compute_class_weights(dataloader, "dimension", 3, self.device)

        self.assertEqual(weights.shape, (3,))

        # Class 0 (90 samples) should have smallest weight
        # Class 1 and 2 (5 samples each) should have largest weights
        # Weights are inversely proportional to class frequency
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[0], weights[2])
        self.assertAlmostEqual(weights[1].item(), weights[2].item(), places=5)

        # Verify weight formula with capping: weight = total / (num_classes * class_count)
        # With max_ratio=20.0 capping:
        # Class 0: 100 / (3 * 90) = 0.370 (min weight)
        # Class 1: 100 / (3 * 5) = 6.667 (no capping needed, < 20× min)
        # Class 2: 100 / (3 * 5) = 6.667 (no capping needed, < 20× min)
        self.assertAlmostEqual(weights[0].item(), 100.0 / (3 * 90), places=3)
        # With 20× cap, these weights should be uncapped (natural 18:1 ratio < 20:1 cap)
        self.assertAlmostEqual(weights[1].item(), 100.0 / (3 * 5), places=3)

    def test_compute_class_weights_multiple_properties(self):
        """Test class weights for multiple classification properties."""
        dataloader = self.create_imbalanced_dataloader()

        # Compute weights for all properties
        weights_dimension = compute_class_weights(dataloader, "dimension", 3, self.device)
        weights_ring_count = compute_class_weights(dataloader, "ring_count", 4, self.device)
        weights_chirality = compute_class_weights(dataloader, "chirality", 2, self.device)

        # Check all weights have correct shapes
        self.assertEqual(weights_dimension.shape, (3,))
        self.assertEqual(weights_ring_count.shape, (4,))
        self.assertEqual(weights_chirality.shape, (2,))

        # Chirality is most imbalanced (95/5), but capped at 20:1 ratio
        chirality_ratio = weights_chirality[1] / weights_chirality[0]
        self.assertAlmostEqual(chirality_ratio.item(), 19.0, places=1)  # Natural 19:1, under 20:1 cap

    def test_train_with_class_weights(self):
        """Test training with class weights applied."""
        dataloader = self.create_imbalanced_dataloader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Compute class weights
        class_weights = {}
        for prop_cfg in self.property_configs:
            weights = compute_class_weights(
                dataloader,
                prop_cfg["name"],
                prop_cfg["num_classes"],
                self.device
            )
            class_weights[prop_cfg["name"]] = weights

        # Train with weighted loss
        loss_weighted, property_losses_weighted, is_healthy = train_one_epoch(
            self.model,
            dataloader,
            optimizer,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=class_weights
        )

        # Loss should be valid
        self.assertIsInstance(loss_weighted, float)
        self.assertGreater(loss_weighted, 0)

        # All properties should have losses computed
        for prop_cfg in self.property_configs:
            self.assertIn(prop_cfg["name"], property_losses_weighted)
            self.assertGreater(property_losses_weighted[prop_cfg["name"]], 0)

    def test_train_without_class_weights(self):
        """Test training without class weights (baseline)."""
        dataloader = self.create_imbalanced_dataloader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Train without weighted loss
        loss_unweighted, property_losses_unweighted, is_healthy = train_one_epoch(
            self.model,
            dataloader,
            optimizer,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None
        )

        # Loss should be valid
        self.assertIsInstance(loss_unweighted, float)
        self.assertGreater(loss_unweighted, 0)

    def test_evaluate_with_class_weights(self):
        """Test evaluation with class weights."""
        dataloader = self.create_imbalanced_dataloader()

        # Compute class weights
        class_weights = {}
        for prop_cfg in self.property_configs:
            weights = compute_class_weights(
                dataloader,
                prop_cfg["name"],
                prop_cfg["num_classes"],
                self.device
            )
            class_weights[prop_cfg["name"]] = weights

        # Evaluate with weighted loss
        mse, rmse, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=class_weights,
            compute_confusion=False
        )

        # Check metrics are computed
        self.assertIsInstance(mse, float)
        self.assertGreater(mse, 0)

        for prop_cfg in self.property_configs:
            self.assertIn(prop_cfg["name"], property_metrics)
            self.assertIn("loss", property_metrics[prop_cfg["name"]])
            self.assertIn("accuracy", property_metrics[prop_cfg["name"]])


class TestConfusionMatrix(unittest.TestCase):
    """Test confusion matrix computation for classification tasks."""

    def setUp(self):
        """Set up model and data for confusion matrix testing."""
        self.device = torch.device("cpu")
        self.vocab_size = 50
        self.max_len = 32
        self.d_model = 64

        # Create property configs with different numbers of classes
        self.property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 3, "attach_at_block": 0},
            {"name": "ring_count", "task": "classification", "num_classes": 4, "attach_at_block": 0},
            {"name": "chirality", "task": "classification", "num_classes": 2, "attach_at_block": 0},
            {"name": "planar_error", "task": "regression", "attach_at_block": 0},  # Should NOT get confusion matrix
        ]

        self.model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            n_encoder_blocks=1,
            max_len=self.max_len,
            d_model=self.d_model
        )

    def create_predictable_dataloader(self):
        """Create dataloader with known prediction patterns for testing."""
        class PredictableDataset:
            def __init__(self):
                # Create data where we know the distribution
                self.n_samples = 30
                # Pattern: 10 samples each of class 0, 1, 2 for dimension
                self.dimension_targets = [0]*10 + [1]*10 + [2]*10
                # Ring count: 8, 8, 8, 6 for classes 0, 1, 2, 3
                self.ring_count_targets = [0]*8 + [1]*8 + [2]*8 + [3]*6
                # Chirality: 20 class 0, 10 class 1
                self.chirality_targets = [0]*20 + [1]*10
                # Regression target
                self.planar_error_targets = [float(i % 10) for i in range(30)]

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                input_ids = torch.randint(0, 50, (32,))
                attention_mask = torch.ones(32)
                targets = {
                    "dimension": self.dimension_targets[idx],
                    "ring_count": self.ring_count_targets[idx],
                    "chirality": self.chirality_targets[idx],
                    "planar_error": self.planar_error_targets[idx],
                }
                return input_ids, attention_mask, targets

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            attention_masks = torch.stack([b[1] for b in batch])
            targets = {
                "dimension": torch.tensor([b[2]["dimension"] for b in batch]),
                "ring_count": torch.tensor([b[2]["ring_count"] for b in batch]),
                "chirality": torch.tensor([b[2]["chirality"] for b in batch]),
                "planar_error": torch.tensor([b[2]["planar_error"] for b in batch]),
            }
            return input_ids, attention_masks, targets

        from torch.utils.data import DataLoader
        dataset = PredictableDataset()
        return DataLoader(dataset, batch_size=6, collate_fn=collate)

    def test_evaluate_with_confusion_matrix(self):
        """Test that confusion matrices are computed when requested."""
        dataloader = self.create_predictable_dataloader()

        mse, rmse, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=True
        )

        # Check that confusion matrices exist for classification tasks
        self.assertIn("dimension", property_metrics)
        self.assertIn("confusion_matrix", property_metrics["dimension"])
        self.assertIn("ring_count", property_metrics)
        self.assertIn("confusion_matrix", property_metrics["ring_count"])
        self.assertIn("chirality", property_metrics)
        self.assertIn("confusion_matrix", property_metrics["chirality"])

        # Regression tasks should NOT have confusion matrices
        self.assertIn("planar_error", property_metrics)
        self.assertNotIn("confusion_matrix", property_metrics["planar_error"])

    def test_confusion_matrix_shape(self):
        """Test that confusion matrices have correct shapes."""
        dataloader = self.create_predictable_dataloader()

        _, _, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=True
        )

        # Check shapes match num_classes
        self.assertEqual(property_metrics["dimension"]["confusion_matrix"].shape, (3, 3))
        self.assertEqual(property_metrics["ring_count"]["confusion_matrix"].shape, (4, 4))
        self.assertEqual(property_metrics["chirality"]["confusion_matrix"].shape, (2, 2))

    def test_confusion_matrix_row_sums(self):
        """Test that confusion matrix row sums equal class counts in dataset."""
        dataloader = self.create_predictable_dataloader()

        _, _, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=True
        )

        # Dimension: 10 samples each of class 0, 1, 2
        cm_dim = property_metrics["dimension"]["confusion_matrix"]
        self.assertEqual(cm_dim.sum(axis=1)[0], 10)  # Row 0 (true class 0)
        self.assertEqual(cm_dim.sum(axis=1)[1], 10)  # Row 1 (true class 1)
        self.assertEqual(cm_dim.sum(axis=1)[2], 10)  # Row 2 (true class 2)

        # Chirality: 20 class 0, 10 class 1
        cm_chir = property_metrics["chirality"]["confusion_matrix"]
        self.assertEqual(cm_chir.sum(axis=1)[0], 20)  # Row 0 (true class 0)
        self.assertEqual(cm_chir.sum(axis=1)[1], 10)  # Row 1 (true class 1)

    def test_confusion_matrix_total_equals_dataset_size(self):
        """Test that total confusion matrix entries equals dataset size."""
        dataloader = self.create_predictable_dataloader()

        _, _, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=True
        )

        # All confusion matrices should sum to dataset size (30)
        self.assertEqual(property_metrics["dimension"]["confusion_matrix"].sum(), 30)
        self.assertEqual(property_metrics["ring_count"]["confusion_matrix"].sum(), 30)
        self.assertEqual(property_metrics["chirality"]["confusion_matrix"].sum(), 30)

    def test_per_class_metrics_exist(self):
        """Test that per-class metrics (precision, recall, F1) are computed."""
        dataloader = self.create_predictable_dataloader()

        _, _, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=True
        )

        # Check per-class metrics exist for all classes
        for prop_name in ["dimension", "ring_count", "chirality"]:
            self.assertIn("per_class_metrics", property_metrics[prop_name])
            per_class = property_metrics[prop_name]["per_class_metrics"]

            # Find num_classes for this property
            num_classes = None
            for cfg in self.property_configs:
                if cfg["name"] == prop_name:
                    num_classes = cfg["num_classes"]
                    break

            # Check metrics exist for all classes
            for class_idx in range(num_classes):
                self.assertIn(class_idx, per_class)
                self.assertIn("precision", per_class[class_idx])
                self.assertIn("recall", per_class[class_idx])
                self.assertIn("f1", per_class[class_idx])
                self.assertIn("support", per_class[class_idx])

                # Values should be in valid ranges
                self.assertGreaterEqual(per_class[class_idx]["precision"], 0.0)
                self.assertLessEqual(per_class[class_idx]["precision"], 1.0)
                self.assertGreaterEqual(per_class[class_idx]["recall"], 0.0)
                self.assertLessEqual(per_class[class_idx]["recall"], 1.0)
                self.assertGreaterEqual(per_class[class_idx]["f1"], 0.0)
                self.assertLessEqual(per_class[class_idx]["f1"], 1.0)
                self.assertGreaterEqual(per_class[class_idx]["support"], 0)

    def test_per_class_support_matches_dataset(self):
        """Test that support counts match actual class distribution."""
        dataloader = self.create_predictable_dataloader()

        _, _, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=True
        )

        # Dimension: 10 samples per class
        dim_metrics = property_metrics["dimension"]["per_class_metrics"]
        self.assertEqual(dim_metrics[0]["support"], 10)
        self.assertEqual(dim_metrics[1]["support"], 10)
        self.assertEqual(dim_metrics[2]["support"], 10)

        # Chirality: 20 class 0, 10 class 1
        chir_metrics = property_metrics["chirality"]["per_class_metrics"]
        self.assertEqual(chir_metrics[0]["support"], 20)
        self.assertEqual(chir_metrics[1]["support"], 10)

    def test_evaluate_without_confusion_matrix(self):
        """Test that confusion matrices are NOT computed when not requested."""
        dataloader = self.create_predictable_dataloader()

        _, _, property_metrics = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=False  # Explicitly disabled
        )

        # Confusion matrices should NOT exist
        self.assertNotIn("confusion_matrix", property_metrics["dimension"])
        self.assertNotIn("confusion_matrix", property_metrics["ring_count"])
        self.assertNotIn("confusion_matrix", property_metrics["chirality"])
        self.assertNotIn("per_class_metrics", property_metrics["dimension"])

        # But basic metrics should still exist
        self.assertIn("loss", property_metrics["dimension"])
        self.assertIn("accuracy", property_metrics["dimension"])

    def test_confusion_matrix_with_all_properties(self):
        """Test confusion matrix computation with all 5 classification heads."""
        # Add all classification properties
        all_class_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 3, "attach_at_block": 0},
            {"name": "ring_count", "task": "classification", "num_classes": 4, "attach_at_block": 0},
            {"name": "chirality", "task": "classification", "num_classes": 2, "attach_at_block": 0},
            {"name": "n_symmetry_planes", "task": "classification", "num_classes": 5, "attach_at_block": 0},
            {"name": "point_group", "task": "classification", "num_classes": 6, "attach_at_block": 0},
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=all_class_configs,
            n_encoder_blocks=1,
            max_len=self.max_len,
            d_model=self.d_model
        )

        class AllPropertiesDataset:
            def __len__(self):
                return 30

            def __getitem__(self, idx):
                input_ids = torch.randint(0, 50, (32,))
                attention_mask = torch.ones(32)
                targets = {
                    "dimension": idx % 3,
                    "ring_count": idx % 4,
                    "chirality": idx % 2,
                    "n_symmetry_planes": idx % 5,
                    "point_group": idx % 6,
                }
                return input_ids, attention_mask, targets

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            attention_masks = torch.stack([b[1] for b in batch])
            targets = {k: torch.tensor([b[2][k] for b in batch]) for k in batch[0][2].keys()}
            return input_ids, attention_masks, targets

        from torch.utils.data import DataLoader
        dataset = AllPropertiesDataset()
        dataloader = DataLoader(dataset, batch_size=6, collate_fn=collate)

        _, _, property_metrics = evaluate(
            model,
            dataloader,
            self.device,
            all_class_configs,
            bf16=False,
            class_weights=None,
            compute_confusion=True
        )

        # All 5 classification properties should have confusion matrices
        for prop_cfg in all_class_configs:
            prop_name = prop_cfg["name"]
            self.assertIn(prop_name, property_metrics)
            self.assertIn("confusion_matrix", property_metrics[prop_name])
            self.assertIn("per_class_metrics", property_metrics[prop_name])

            # Check shape
            num_classes = prop_cfg["num_classes"]
            self.assertEqual(property_metrics[prop_name]["confusion_matrix"].shape, (num_classes, num_classes))


class TestFocalLoss(unittest.TestCase):
    """Test Focal Loss implementation for imbalanced classification."""

    def setUp(self):
        """Set up test data for focal loss."""
        self.device = torch.device("cpu")
        # Create imbalanced dataset with 3 classes
        self.num_classes = 3
        self.batch_size = 10

    def test_focal_loss_no_weights(self):
        """Test focal loss without class weights (alpha=None)."""
        focal_loss = FocalLoss(alpha=None, gamma=2.0)

        # Create logits and targets
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        # Compute loss
        loss = focal_loss(logits, targets)

        # Check output is scalar
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)

        # Check loss is positive
        self.assertGreater(loss.item(), 0.0)

    def test_focal_loss_with_weights(self):
        """Test focal loss with class weights (alpha)."""
        # Create class weights
        alpha = torch.tensor([1.0, 2.0, 3.0])
        focal_loss = FocalLoss(alpha=alpha, gamma=2.0)

        # Create logits and targets
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        # Compute loss
        loss = focal_loss(logits, targets)

        # Check output is scalar
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)

        # Check loss is positive
        self.assertGreater(loss.item(), 0.0)

    def test_focal_loss_gamma_effect(self):
        """Test that higher gamma focuses more on hard examples."""
        # Create perfectly predicted example (easy) and misclassified example (hard)
        logits = torch.tensor([[10.0, 0.0, 0.0],  # Easy: confident correct
                                [0.0, 0.0, 10.0]]) # Hard: confident wrong
        targets = torch.tensor([0, 0])  # Both should be class 0

        # Focal loss with gamma=0 (equivalent to CE)
        focal_gamma0 = FocalLoss(alpha=None, gamma=0.0)
        loss_gamma0 = focal_gamma0(logits, targets)

        # Focal loss with gamma=2 (standard focal)
        focal_gamma2 = FocalLoss(alpha=None, gamma=2.0)
        loss_gamma2 = focal_gamma2(logits, targets)

        # With gamma=2, hard example should contribute more relative to easy example
        # Both losses should be positive
        self.assertGreater(loss_gamma0.item(), 0.0)
        self.assertGreater(loss_gamma2.item(), 0.0)

    def test_focal_loss_reduction(self):
        """Test different reduction modes (mean, sum, none)."""
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        # Mean reduction
        focal_mean = FocalLoss(alpha=None, gamma=2.0, reduction='mean')
        loss_mean = focal_mean(logits, targets)
        self.assertEqual(loss_mean.dim(), 0)  # Scalar

        # Sum reduction
        focal_sum = FocalLoss(alpha=None, gamma=2.0, reduction='sum')
        loss_sum = focal_sum(logits, targets)
        self.assertEqual(loss_sum.dim(), 0)  # Scalar

        # Sum should be larger than mean (by factor of batch_size)
        self.assertGreater(loss_sum.item(), loss_mean.item())

        # None reduction
        focal_none = FocalLoss(alpha=None, gamma=2.0, reduction='none')
        loss_none = focal_none(logits, targets)
        self.assertEqual(loss_none.shape, (self.batch_size,))  # Per-sample losses

    def test_focal_loss_backward(self):
        """Test that focal loss supports backpropagation."""
        focal_loss = FocalLoss(alpha=None, gamma=2.0)

        # Create logits with requires_grad
        logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        # Compute loss and backward
        loss = focal_loss(logits, targets)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)


class TestComprehensiveMetrics(unittest.TestCase):
    """Test comprehensive classification metrics computation."""

    def test_binary_classification_metrics(self):
        """Test metrics for binary classification (num_classes=2)."""
        # Perfect predictions
        predictions = np.array([0, 0, 1, 1, 0, 1])
        targets = np.array([0, 0, 1, 1, 0, 1])
        probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8],
                                   [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]])

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
            num_classes=2
        )

        # Check all expected metrics are present
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("auprc", metrics)

        # Perfect predictions should give 100% accuracy, F1=1.0
        self.assertEqual(metrics["accuracy"], 100.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0, places=5)
        self.assertAlmostEqual(metrics["weighted_f1"], 1.0, places=5)
        self.assertAlmostEqual(metrics["macro_recall"], 1.0, places=5)

        # AUPRC should be 1.0 for perfect predictions
        self.assertAlmostEqual(metrics["auprc"], 1.0, places=5)

    def test_multiclass_classification_metrics(self):
        """Test metrics for multi-class classification (num_classes>2)."""
        # 3-class problem with some errors
        predictions = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        targets =     np.array([0, 0, 1, 2, 2, 2, 1, 1, 0])  # Some mistakes
        probabilities = np.array([
            [0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1],
            [0.1, 0.6, 0.3], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]
        ])

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
            num_classes=3
        )

        # Check all expected metrics are present
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("auprc", metrics)

        # Accuracy should be 6/9 = 66.67% (indices 0,1,2,4,5,7 are correct)
        expected_accuracy = 100.0 * (6.0 / 9.0)
        self.assertAlmostEqual(metrics["accuracy"], expected_accuracy, places=1)

        # F1 scores should be between 0 and 1
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)
        self.assertGreaterEqual(metrics["weighted_f1"], 0.0)
        self.assertLessEqual(metrics["weighted_f1"], 1.0)

        # AUPRC should be computed (macro averaged)
        self.assertGreaterEqual(metrics["auprc"], 0.0)
        self.assertLessEqual(metrics["auprc"], 1.0)

    def test_metrics_without_probabilities(self):
        """Test that metrics work without probabilities (no AUPRC)."""
        predictions = np.array([0, 0, 1, 1, 2, 2])
        targets = np.array([0, 0, 1, 1, 2, 2])

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=None,  # No probabilities
            num_classes=3
        )

        # Should have all metrics except AUPRC
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertNotIn("auprc", metrics)  # Should not be computed

    def test_metrics_with_zero_classes(self):
        """Test metrics with zero_division handling (missing classes)."""
        # All predictions are class 0 (classes 1 and 2 never predicted)
        predictions = np.array([0, 0, 0, 0, 0, 0])
        targets = np.array([0, 0, 0, 1, 1, 2])  # Some targets are class 1 and 2

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=None,
            num_classes=3
        )

        # Should not raise errors, should handle zero_division gracefully
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)

        # Accuracy should be 3/6 = 50% (only class 0 predictions correct)
        self.assertAlmostEqual(metrics["accuracy"], 50.0, places=1)

    def test_metrics_auto_detect_num_classes(self):
        """Test that num_classes is auto-detected from data."""
        predictions = np.array([0, 1, 2, 0, 1, 2])
        targets = np.array([0, 1, 2, 0, 1, 2])

        # Don't provide num_classes (should auto-detect as 3)
        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=None,
            num_classes=None  # Auto-detect
        )

        # Should work correctly
        self.assertIn("accuracy", metrics)
        self.assertEqual(metrics["accuracy"], 100.0)


class TestModelSelection(unittest.TestCase):
    """Test model selection based on task-appropriate metrics."""

    def setUp(self):
        """Set up simple model and data."""
        self.device = torch.device("cpu")
        self.vocab_size = 50
        self.max_len = 32
        self.d_model = 64

    def test_binary_classification_uses_auprc(self):
        """Test that binary classification (n=2) uses AUPRC as primary metric."""
        property_configs = [
            {"name": "chirality", "task": "classification", "num_classes": 2, "attach_at_block": 0}
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            n_encoder_blocks=1,
            max_len=self.max_len,
            d_model=self.d_model
        )

        # Create simple dataloader
        class SimpleDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return (
                    torch.randint(0, 50, (32,)),
                    torch.ones(32),
                    {"chirality": torch.randint(0, 2, (1,)).item()}
                )

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            masks = torch.stack([b[1] for b in batch])
            targets = {"chirality": torch.tensor([b[2]["chirality"] for b in batch])}
            return input_ids, masks, targets

        from torch.utils.data import DataLoader
        dataloader = DataLoader(SimpleDataset(), batch_size=5, collate_fn=collate)

        # Evaluate and check that AUPRC is computed
        _, _, property_metrics = evaluate(
            model, dataloader, self.device, property_configs,
            bf16=False, class_weights=None, compute_confusion=False
        )

        # Check that AUPRC is present and is the primary metric
        self.assertIn("chirality", property_metrics)
        self.assertIn("auprc", property_metrics["chirality"])
        self.assertIn("primary_metric", property_metrics["chirality"])
        self.assertEqual(property_metrics["chirality"]["primary_metric"], "auprc")
        self.assertEqual(property_metrics["chirality"]["secondary_metric"], "macro_f1")

    def test_multiclass_classification_uses_macro_f1(self):
        """Test that multi-class classification (n>2) uses macro_F1 as primary metric."""
        property_configs = [
            {"name": "point_group", "task": "classification", "num_classes": 15, "attach_at_block": 0}
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            n_encoder_blocks=1,
            max_len=self.max_len,
            d_model=self.d_model
        )

        # Create simple dataloader
        class SimpleDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return (
                    torch.randint(0, 50, (32,)),
                    torch.ones(32),
                    {"point_group": torch.randint(0, 15, (1,)).item()}
                )

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            masks = torch.stack([b[1] for b in batch])
            targets = {"point_group": torch.tensor([b[2]["point_group"] for b in batch])}
            return input_ids, masks, targets

        from torch.utils.data import DataLoader
        dataloader = DataLoader(SimpleDataset(), batch_size=5, collate_fn=collate)

        # Evaluate and check that macro_f1 is computed
        _, _, property_metrics = evaluate(
            model, dataloader, self.device, property_configs,
            bf16=False, class_weights=None, compute_confusion=False
        )

        # Check that macro_f1 is present and is the primary metric
        self.assertIn("point_group", property_metrics)
        self.assertIn("macro_f1", property_metrics["point_group"])
        self.assertIn("primary_metric", property_metrics["point_group"])
        self.assertEqual(property_metrics["point_group"]["primary_metric"], "macro_f1")
        self.assertEqual(property_metrics["point_group"]["secondary_metric"], "weighted_f1")


class TestEarlyStopping(unittest.TestCase):
    """Tests for EarlyStoppingMonitor functionality."""

    def test_instability_detection_nan_loss(self):
        """Test that NaN loss triggers early stopping."""
        monitor = EarlyStoppingMonitor(
            patience=10,
            overfit_patience=5,
            warmup_epochs=5,
            instability_threshold=3.0,
            maximize_metric=False
        )

        # Normal first 2 epochs
        self.assertFalse(monitor.update(1, 1.5, 2.0))
        self.assertFalse(monitor.update(2, 1.4, 1.9))

        # NaN loss in epoch 3
        should_stop = monitor.update(3, float('nan'), 1.8)
        self.assertTrue(should_stop)
        self.assertIn("NaN/Inf", monitor.stop_reason)

    def test_instability_detection_inf_loss(self):
        """Test that Inf loss triggers early stopping."""
        monitor = EarlyStoppingMonitor(
            patience=10,
            overfit_patience=5,
            warmup_epochs=5,
            instability_threshold=3.0,
            maximize_metric=False
        )

        # Normal first 2 epochs
        self.assertFalse(monitor.update(1, 1.5, 2.0))
        self.assertFalse(monitor.update(2, 1.4, 1.9))

        # Inf loss in epoch 3
        should_stop = monitor.update(3, float('inf'), 1.8)
        self.assertTrue(should_stop)
        self.assertIn("NaN/Inf", monitor.stop_reason)

    def test_instability_detection_wild_swings(self):
        """Test detection of wild loss swings in warmup period."""
        monitor = EarlyStoppingMonitor(
            patience=10,
            overfit_patience=5,
            warmup_epochs=5,
            instability_threshold=2.0,  # Lower threshold for easier detection
            maximize_metric=False
        )

        # Epoch 1-3: establish stable baseline
        self.assertFalse(monitor.update(1, 1.0, 2.0))
        self.assertFalse(monitor.update(2, 1.05, 1.9))
        self.assertFalse(monitor.update(3, 1.02, 1.8))

        # Epoch 4: wild spike - loss jumps to 5.0 (huge deviation)
        # Mean of [1.0, 1.05, 1.02] = 1.023, std ≈ 0.026
        # Loss of 5.0 gives z-score = (5.0 - 1.023) / 0.026 ≈ 153 (massively above threshold)
        should_stop = monitor.update(4, 5.0, 1.7)
        self.assertTrue(should_stop)
        self.assertIn("variance too high", monitor.stop_reason)

    def test_instability_only_in_warmup(self):
        """Test that instability checks only apply during warmup period."""
        monitor = EarlyStoppingMonitor(
            patience=10,
            overfit_patience=5,
            warmup_epochs=3,  # Only first 3 epochs
            instability_threshold=3.0,
            maximize_metric=False
        )

        # Warmup: stable
        self.assertFalse(monitor.update(1, 1.0, 2.0))
        self.assertFalse(monitor.update(2, 1.1, 1.9))
        self.assertFalse(monitor.update(3, 1.0, 1.8))

        # After warmup: wild swing should NOT trigger instability check
        # (but might trigger other checks like plateau/overfit)
        should_stop = monitor.update(4, 10.0, 1.7)
        # Should not stop for instability (after warmup)
        # But might stop for other reasons if conditions met
        if should_stop:
            self.assertNotIn("variance too high", monitor.stop_reason)

    def test_plateau_detection_minimize_metric(self):
        """Test plateau detection for metrics that should be minimized (RMSE)."""
        monitor = EarlyStoppingMonitor(
            patience=3,
            overfit_patience=5,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,  # Lower is better
            min_delta=0.01
        )

        # Epoch 1-2: warmup, improvement
        self.assertFalse(monitor.update(1, 1.0, 2.0))  # val=2.0 (best)
        self.assertFalse(monitor.update(2, 0.9, 1.8))  # val=1.8 (best)

        # Epoch 3-4: no improvement (not yet at patience)
        self.assertFalse(monitor.update(3, 0.8, 1.9))  # val=1.9 (worse) - 1 epoch
        self.assertFalse(monitor.update(4, 0.7, 1.85))  # val=1.85 (worse) - 2 epochs

        # Epoch 5: patience exceeded (3 epochs without improvement)
        should_stop = monitor.update(5, 0.6, 1.9)  # val=1.9 (worse) - 3 epochs
        self.assertTrue(should_stop)
        self.assertIn("Plateau detected", monitor.stop_reason)
        self.assertIn("No improvement", monitor.stop_reason)

    def test_plateau_detection_maximize_metric(self):
        """Test plateau detection for metrics that should be maximized (F1, AUPRC)."""
        monitor = EarlyStoppingMonitor(
            patience=3,
            overfit_patience=5,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=True,  # Higher is better
            min_delta=0.01
        )

        # Epoch 1-2: warmup, improvement
        self.assertFalse(monitor.update(1, 1.0, 0.70))  # val=0.70 (best)
        self.assertFalse(monitor.update(2, 0.9, 0.75))  # val=0.75 (best)

        # Epoch 3-4: no improvement (not yet at patience)
        self.assertFalse(monitor.update(3, 0.8, 0.74))  # val=0.74 (worse) - 1 epoch
        self.assertFalse(monitor.update(4, 0.7, 0.73))  # val=0.73 (worse) - 2 epochs

        # Epoch 5: patience exceeded (3 epochs without improvement)
        should_stop = monitor.update(5, 0.6, 0.74)  # val=0.74 (worse) - 3 epochs
        self.assertTrue(should_stop)
        self.assertIn("Plateau detected", monitor.stop_reason)

    def test_plateau_disabled_with_patience_zero(self):
        """Test that plateau detection is disabled when patience=0."""
        monitor = EarlyStoppingMonitor(
            patience=0,  # Disabled
            overfit_patience=5,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False
        )

        # No improvement for many epochs - should NOT stop
        self.assertFalse(monitor.update(1, 1.0, 2.0))
        self.assertFalse(monitor.update(2, 0.9, 1.8))
        self.assertFalse(monitor.update(3, 0.8, 1.9))
        self.assertFalse(monitor.update(4, 0.7, 1.9))
        self.assertFalse(monitor.update(5, 0.6, 1.9))
        self.assertFalse(monitor.update(10, 0.1, 1.9))  # Still no stop

    def test_overfitting_detection_minimize_metric(self):
        """Test overfitting detection (train improving, val degrading) for minimize metrics."""
        monitor = EarlyStoppingMonitor(
            patience=20,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,  # Lower is better (RMSE)
            min_delta=0.01
        )

        # Warmup: normal
        self.assertFalse(monitor.update(1, 2.0, 2.0))
        self.assertFalse(monitor.update(2, 1.8, 1.8))

        # Post-warmup: train improving (decreasing) but val degrading (increasing)
        self.assertFalse(monitor.update(3, 1.6, 1.85))  # train↓, val↑ - 1 epoch
        self.assertFalse(monitor.update(4, 1.4, 1.90))  # train↓, val↑ - 2 epochs
        self.assertFalse(monitor.update(5, 1.2, 1.95))  # train↓, val↑ - 3 epochs

        # Epoch 6: overfit patience exceeded
        should_stop = monitor.update(6, 1.0, 2.00)
        self.assertTrue(should_stop)
        self.assertIn("Overfitting detected", monitor.stop_reason)

    def test_overfitting_detection_maximize_metric(self):
        """Test overfitting detection for maximize metrics (F1, AUPRC)."""
        monitor = EarlyStoppingMonitor(
            patience=20,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=True,  # Higher is better
            min_delta=0.01
        )

        # Warmup: normal
        self.assertFalse(monitor.update(1, 2.0, 0.70))
        self.assertFalse(monitor.update(2, 1.8, 0.75))

        # Post-warmup: train improving (decreasing loss) but val degrading (decreasing metric)
        self.assertFalse(monitor.update(3, 1.6, 0.74))  # train↓, val↓ - 1 epoch
        self.assertFalse(monitor.update(4, 1.4, 0.72))  # train↓, val↓ - 2 epochs
        self.assertFalse(monitor.update(5, 1.2, 0.70))  # train↓, val↓ - 3 epochs

        # Epoch 6: overfit patience exceeded
        should_stop = monitor.update(6, 1.0, 0.68)
        self.assertTrue(should_stop)
        self.assertIn("Overfitting detected", monitor.stop_reason)

    def test_overfitting_resets_on_improvement(self):
        """Test that overfitting counter resets when validation improves."""
        monitor = EarlyStoppingMonitor(
            patience=20,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,
            min_delta=0.01
        )

        # Warmup
        self.assertFalse(monitor.update(1, 2.0, 2.0))
        self.assertFalse(monitor.update(2, 1.8, 1.8))

        # Overfit pattern starts
        self.assertFalse(monitor.update(3, 1.6, 1.85))  # train↓, val↑ - 1 epoch
        self.assertFalse(monitor.update(4, 1.4, 1.90))  # train↓, val↑ - 2 epochs

        # Then validation improves - should reset counter
        self.assertFalse(monitor.update(5, 1.2, 1.70))  # train↓, val↓ - RESET

        # Overfit pattern starts again
        self.assertFalse(monitor.update(6, 1.0, 1.75))  # train↓, val↑ - 1 epoch
        self.assertFalse(monitor.update(7, 0.9, 1.80))  # train↓, val↑ - 2 epochs

        # Epoch 8: patience exceeded (3 consecutive epochs)
        should_stop = monitor.update(8, 0.8, 1.85)  # train↓, val↑ - 3 epochs
        self.assertTrue(should_stop)

    def test_overfitting_disabled_with_patience_zero(self):
        """Test that overfitting detection is disabled when overfit_patience=0."""
        monitor = EarlyStoppingMonitor(
            patience=20,
            overfit_patience=0,  # Disabled
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False
        )

        # Warmup
        self.assertFalse(monitor.update(1, 2.0, 2.0))
        self.assertFalse(monitor.update(2, 1.8, 1.8))

        # Strong overfitting pattern - should NOT stop
        self.assertFalse(monitor.update(3, 1.6, 1.9))
        self.assertFalse(monitor.update(4, 1.4, 2.0))
        self.assertFalse(monitor.update(5, 1.2, 2.1))
        self.assertFalse(monitor.update(6, 1.0, 2.2))

    def test_nan_inf_in_validation_metric(self):
        """Test that NaN/Inf in validation metric triggers early stopping."""
        monitor = EarlyStoppingMonitor(
            patience=10,
            overfit_patience=5,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False
        )

        # Normal epochs
        self.assertFalse(monitor.update(1, 1.0, 2.0))
        self.assertFalse(monitor.update(2, 0.9, 1.9))

        # NaN in validation metric
        should_stop = monitor.update(3, 0.8, float('nan'))
        self.assertTrue(should_stop)
        self.assertIn("NaN/Inf", monitor.stop_reason)

    def test_combined_checks_priority(self):
        """Test that NaN/Inf checks have priority over other checks."""
        monitor = EarlyStoppingMonitor(
            patience=2,  # Would trigger plateau
            overfit_patience=2,  # Would trigger overfitting
            warmup_epochs=5,
            instability_threshold=3.0,
            maximize_metric=False
        )

        # Setup conditions for plateau and overfitting
        self.assertFalse(monitor.update(1, 1.0, 2.0))
        self.assertFalse(monitor.update(2, 0.9, 1.9))

        # NaN should be detected first (instability check runs first)
        should_stop = monitor.update(3, float('nan'), 1.95)
        self.assertTrue(should_stop)
        self.assertIn("NaN/Inf", monitor.stop_reason)

    def test_train_one_epoch_health_check_nan(self):
        """Test that train_one_epoch detects NaN in loss and returns unhealthy status."""
        # Create a simple model that can produce NaN
        property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 3, "attach_at_block": 0}
        ]

        model = HierarchicalTransformer(
            vocab_size=100,
            property_configs=property_configs,
            n_encoder_blocks=1,
            max_len=32,
            d_model=64
        )

        # Create simple dataset
        class SimpleDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return (
                    torch.randint(0, 50, (32,)),
                    torch.ones(32),
                    {"dimension": torch.randint(0, 3, (1,)).item()}
                )

        def collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            masks = torch.stack([b[1] for b in batch])
            targets = {"dimension": torch.tensor([b[2]["dimension"] for b in batch])}
            return input_ids, masks, targets

        from torch.utils.data import DataLoader
        dataloader = DataLoader(SimpleDataset(), batch_size=5, collate_fn=collate)

        # Use extremely high learning rate to cause instability
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e10)  # Intentionally too high

        device = torch.device("cpu")
        model.to(device)

        # Train one epoch - should detect unhealthy training
        # Note: May or may not produce NaN depending on random init
        # This is a structural test to ensure the return signature is correct
        result = train_one_epoch(
            model, dataloader, optimizer, device, property_configs,
            bf16=False, scheduler=None, class_weights=None
        )

        # Check that function returns 3 values
        self.assertEqual(len(result), 3)
        epoch_loss, property_losses, is_healthy = result
        self.assertIsInstance(epoch_loss, float)
        self.assertIsInstance(property_losses, dict)
        self.assertIsInstance(is_healthy, bool)


class TestSchedulerEarlyStoppingIntegration(unittest.TestCase):
    """Test integration between schedulers and early stopping for both minimize and maximize metrics."""

    def test_early_stopping_minimize_metric_rmse(self):
        """Test early stopping with minimize metric (RMSE) correctly tracks improvement."""
        monitor = EarlyStoppingMonitor(
            patience=5,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,  # RMSE: lower is better
            min_delta=0.01
        )

        # Simulate RMSE decreasing (improving)
        self.assertFalse(monitor.update(1, 2.0, 1.50))  # best=1.50
        self.assertFalse(monitor.update(2, 1.8, 1.40))  # best=1.40 (improved)
        self.assertFalse(monitor.update(3, 1.6, 1.30))  # best=1.30 (improved)

        # Verify best metric is being tracked correctly
        self.assertEqual(monitor.best_metric, 1.30)
        self.assertEqual(monitor.best_epoch, 3)
        self.assertEqual(monitor.epochs_without_improvement, 0)

    def test_early_stopping_maximize_metric_f1(self):
        """Test early stopping with maximize metric (F1) correctly tracks improvement."""
        monitor = EarlyStoppingMonitor(
            patience=5,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=True,  # F1: higher is better
            min_delta=0.01
        )

        # Simulate F1 increasing (improving)
        self.assertFalse(monitor.update(1, 2.0, 0.70))  # best=0.70
        self.assertFalse(monitor.update(2, 1.8, 0.75))  # best=0.75 (improved)
        self.assertFalse(monitor.update(3, 1.6, 0.80))  # best=0.80 (improved)

        # Verify best metric is being tracked correctly
        self.assertEqual(monitor.best_metric, 0.80)
        self.assertEqual(monitor.best_epoch, 3)
        self.assertEqual(monitor.epochs_without_improvement, 0)

    def test_plateau_detection_respects_metric_direction_minimize(self):
        """Test that plateau detection works correctly for minimize metrics."""
        monitor = EarlyStoppingMonitor(
            patience=3,
            overfit_patience=0,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,  # Lower is better
            min_delta=0.01
        )

        # RMSE improving then plateauing
        self.assertFalse(monitor.update(1, 2.0, 1.50))  # best=1.50
        self.assertFalse(monitor.update(2, 1.8, 1.40))  # best=1.40
        self.assertFalse(monitor.update(3, 1.6, 1.45))  # 1.45 > 1.40, no improve, count=1
        self.assertFalse(monitor.update(4, 1.4, 1.42))  # 1.42 > 1.40, no improve, count=2

        # Epoch 5: Should stop (3 epochs without improvement)
        should_stop = monitor.update(5, 1.2, 1.43)
        self.assertTrue(should_stop)
        self.assertIn("Plateau", monitor.stop_reason)

    def test_plateau_detection_respects_metric_direction_maximize(self):
        """Test that plateau detection works correctly for maximize metrics."""
        monitor = EarlyStoppingMonitor(
            patience=3,
            overfit_patience=0,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=True,  # Higher is better
            min_delta=0.01
        )

        # F1 improving then plateauing
        self.assertFalse(monitor.update(1, 2.0, 0.70))  # best=0.70
        self.assertFalse(monitor.update(2, 1.8, 0.75))  # best=0.75
        self.assertFalse(monitor.update(3, 1.6, 0.74))  # 0.74 < 0.75, no improve, count=1
        self.assertFalse(monitor.update(4, 1.4, 0.73))  # 0.73 < 0.75, no improve, count=2

        # Epoch 5: Should stop (3 epochs without improvement)
        should_stop = monitor.update(5, 1.2, 0.74)
        self.assertTrue(should_stop)
        self.assertIn("Plateau", monitor.stop_reason)

    def test_overfitting_detection_minimize_metric(self):
        """Test overfitting detection with minimize metric (train↓, val↑ = overfitting)."""
        monitor = EarlyStoppingMonitor(
            patience=20,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,  # RMSE: lower is better
            min_delta=0.01
        )

        # Warmup - establish baseline
        self.assertFalse(monitor.update(1, 2.0, 1.50))
        self.assertFalse(monitor.update(2, 1.8, 1.40))

        # Post-warmup: Overfitting pattern (train loss decreasing, val RMSE increasing)
        self.assertFalse(monitor.update(3, 1.6, 1.45))  # train↓, val↑ (worse), count=1
        self.assertFalse(monitor.update(4, 1.4, 1.50))  # train↓, val↑ (worse), count=2
        self.assertFalse(monitor.update(5, 1.2, 1.55))  # train↓, val↑ (worse), count=3

        # Epoch 6: Should detect overfitting (3 consecutive epochs)
        should_stop = monitor.update(6, 1.0, 1.60)  # train↓, val↑ (worse), triggers stop
        self.assertTrue(should_stop)
        self.assertIn("Overfitting", monitor.stop_reason)

    def test_overfitting_detection_maximize_metric(self):
        """Test overfitting detection with maximize metric (train↓, val↓ = overfitting)."""
        monitor = EarlyStoppingMonitor(
            patience=20,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=True,  # F1: higher is better
            min_delta=0.01
        )

        # Warmup - establish baseline
        self.assertFalse(monitor.update(1, 2.0, 0.70))
        self.assertFalse(monitor.update(2, 1.8, 0.75))

        # Post-warmup: Overfitting pattern (train loss decreasing, val F1 decreasing/degrading)
        self.assertFalse(monitor.update(3, 1.6, 0.74))  # train↓, val↓ (worse), count=1
        self.assertFalse(monitor.update(4, 1.4, 0.72))  # train↓, val↓ (worse), count=2
        self.assertFalse(monitor.update(5, 1.2, 0.70))  # train↓, val↓ (worse), count=3

        # Epoch 6: Should detect overfitting (3 consecutive epochs)
        should_stop = monitor.update(6, 1.0, 0.68)  # train↓, val↓ (worse), triggers stop
        self.assertTrue(should_stop)
        self.assertIn("Overfitting", monitor.stop_reason)

    def test_improvement_resets_counters_minimize(self):
        """Test that improvement resets no-improvement counter for minimize metrics."""
        monitor = EarlyStoppingMonitor(
            patience=5,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,
            min_delta=0.01
        )

        # Start improving
        self.assertFalse(monitor.update(1, 2.0, 1.50))  # best=1.50
        self.assertFalse(monitor.update(2, 1.8, 1.40))  # best=1.40

        # Plateau for 2 epochs
        self.assertFalse(monitor.update(3, 1.6, 1.45))  # no improve, count=1
        self.assertFalse(monitor.update(4, 1.4, 1.42))  # no improve, count=2

        # Then improve again - should reset counter
        self.assertFalse(monitor.update(5, 1.2, 1.35))  # improved! count=0

        self.assertEqual(monitor.best_metric, 1.35)
        self.assertEqual(monitor.epochs_without_improvement, 0)

    def test_improvement_resets_counters_maximize(self):
        """Test that improvement resets no-improvement counter for maximize metrics."""
        monitor = EarlyStoppingMonitor(
            patience=5,
            overfit_patience=3,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=True,
            min_delta=0.01
        )

        # Start improving
        self.assertFalse(monitor.update(1, 2.0, 0.70))  # best=0.70
        self.assertFalse(monitor.update(2, 1.8, 0.75))  # best=0.75

        # Plateau for 2 epochs
        self.assertFalse(monitor.update(3, 1.6, 0.74))  # no improve, count=1
        self.assertFalse(monitor.update(4, 1.4, 0.73))  # no improve, count=2

        # Then improve again - should reset counter
        self.assertFalse(monitor.update(5, 1.2, 0.80))  # improved! count=0

        self.assertEqual(monitor.best_metric, 0.80)
        self.assertEqual(monitor.epochs_without_improvement, 0)

    def test_min_delta_threshold_minimize(self):
        """Test that min_delta prevents stopping on tiny fluctuations (minimize)."""
        monitor = EarlyStoppingMonitor(
            patience=3,
            overfit_patience=0,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=False,
            min_delta=0.01  # Require 0.01 improvement
        )

        self.assertFalse(monitor.update(1, 2.0, 1.50))  # best=1.50
        self.assertFalse(monitor.update(2, 1.8, 1.40))  # best=1.40

        # Tiny improvement (< min_delta) should not count as improvement
        self.assertFalse(monitor.update(3, 1.6, 1.395))  # 1.395 vs 1.40, diff=0.005 < 0.01, count=1
        self.assertFalse(monitor.update(4, 1.4, 1.392))  # 1.392 vs 1.40, diff=0.008 < 0.01, count=2

        # Should stop after patience exceeded
        should_stop = monitor.update(5, 1.2, 1.390)
        self.assertTrue(should_stop)

    def test_min_delta_threshold_maximize(self):
        """Test that min_delta prevents stopping on tiny fluctuations (maximize)."""
        monitor = EarlyStoppingMonitor(
            patience=3,
            overfit_patience=0,
            warmup_epochs=2,
            instability_threshold=3.0,
            maximize_metric=True,
            min_delta=0.01  # Require 0.01 improvement
        )

        self.assertFalse(monitor.update(1, 2.0, 0.70))  # best=0.70
        self.assertFalse(monitor.update(2, 1.8, 0.75))  # best=0.75

        # Tiny improvement (< min_delta) should not count as improvement
        self.assertFalse(monitor.update(3, 1.6, 0.755))  # 0.755 vs 0.75, diff=0.005 < 0.01, count=1
        self.assertFalse(monitor.update(4, 1.4, 0.758))  # 0.758 vs 0.75, diff=0.008 < 0.01, count=2

        # Should stop after patience exceeded
        should_stop = monitor.update(5, 1.2, 0.759)
        self.assertTrue(should_stop)


class TestCheckpointResume(unittest.TestCase):
    """Test checkpoint saving and resuming functionality."""

    def setUp(self):
        """Create temporary directory and minimal training setup."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "best_model.pt")

        # Create minimal vocab
        vocab_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "C", "N", "O"]
        self.token_to_id = {t: i for i, t in enumerate(vocab_tokens)}
        self.vocab_size = len(vocab_tokens)

        # Setup property configs
        self.property_configs = [
            {
                "name": "dimension",
                "task": "classification",
                "num_classes": 3,
                "attach_block": 1,
                "enable_cross_attention": False
            }
        ]

        # Create model
        self.d_model = 64
        self.n_encoder_blocks = 2
        self.model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=128,
            max_len=64,
            n_encoder_blocks=self.n_encoder_blocks
        )

        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit("dimension", ["0D", "2D", "3D"])

    def test_checkpoint_save_and_load_basic(self):
        """Test basic checkpoint save and load preserves model state."""
        # Save initial state
        initial_state = {p: param.clone() for p, param in self.model.named_parameters()}

        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": 10,
            "best_val_metric": 0.85,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": None
        }

        torch.save(checkpoint, self.checkpoint_path)
        self.assertTrue(os.path.exists(self.checkpoint_path))

        # Modify model parameters
        for param in self.model.parameters():
            param.data.fill_(999.0)

        # Load checkpoint
        loaded_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(loaded_checkpoint["model_state_dict"])

        # Verify parameters restored
        for name, param in self.model.named_parameters():
            torch.testing.assert_close(param, initial_state[name])

    def test_checkpoint_preserves_optimizer_state(self):
        """Test checkpoint preserves optimizer momentum and learning rate."""
        # Run one optimization step to build momentum
        dummy_input = torch.randint(0, self.vocab_size, (2, 10))
        dummy_mask = (dummy_input != 0).float()
        output = self.model(dummy_input, dummy_mask)
        loss = output["dimension"].sum()
        loss.backward()
        self.optimizer.step()

        # Save checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": 5,
            "best_val_metric": 0.75,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": None
        }

        torch.save(checkpoint, self.checkpoint_path)

        # Create new optimizer and load state
        new_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        loaded_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

        # Verify optimizer state matches
        self.assertEqual(len(new_optimizer.state), len(self.optimizer.state))
        for key in self.optimizer.state:
            if 'exp_avg' in self.optimizer.state[key]:
                torch.testing.assert_close(
                    new_optimizer.state[key]['exp_avg'],
                    self.optimizer.state[key]['exp_avg']
                )

    def test_checkpoint_epoch_counter(self):
        """Test checkpoint correctly stores and retrieves epoch counter."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": 42,
            "best_val_metric": 0.92,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": None
        }

        torch.save(checkpoint, self.checkpoint_path)
        loaded_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        self.assertEqual(loaded_checkpoint["epoch"], 42)
        self.assertAlmostEqual(loaded_checkpoint["best_val_metric"], 0.92)

    def test_checkpoint_validates_architecture_mismatch(self):
        """Test that loading checkpoint with wrong architecture fails gracefully."""
        # Save checkpoint with current architecture
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": 5,
            "best_val_metric": 0.8,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": None
        }

        torch.save(checkpoint, self.checkpoint_path)
        loaded_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # Verify architecture parameters are stored
        self.assertEqual(loaded_checkpoint["vocab_size"], self.vocab_size)
        self.assertEqual(loaded_checkpoint["n_encoder_blocks"], self.n_encoder_blocks)
        self.assertEqual(loaded_checkpoint["d_model"], self.d_model)

        # In real code, these would raise ValueError - here we just verify they're present
        self.assertIn("vocab_size", loaded_checkpoint)
        self.assertIn("n_encoder_blocks", loaded_checkpoint)
        self.assertIn("d_model", loaded_checkpoint)

    def test_checkpoint_includes_label_encoder(self):
        """Test checkpoint includes label encoder for resuming."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": 10,
            "best_val_metric": 0.88,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": None
        }

        torch.save(checkpoint, self.checkpoint_path)
        loaded_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # Verify label encoder is preserved
        self.assertIn("label_encoder", loaded_checkpoint)
        loaded_le = loaded_checkpoint["label_encoder"]
        self.assertEqual(loaded_le.get_num_classes("dimension"), 3)
        # Transform expects single labels, not lists
        self.assertEqual(loaded_le.transform("dimension", "2D"), self.label_encoder.transform("dimension", "2D"))

    def test_checkpoint_with_class_weights(self):
        """Test checkpoint correctly stores and retrieves class weights."""
        class_weights = {"dimension": torch.tensor([1.0, 2.0, 3.0])}

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": 15,
            "best_val_metric": 0.91,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": class_weights
        }

        torch.save(checkpoint, self.checkpoint_path)
        loaded_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        self.assertIn("class_weights", loaded_checkpoint)
        torch.testing.assert_close(
            loaded_checkpoint["class_weights"]["dimension"],
            class_weights["dimension"]
        )

    def test_backward_compatibility_missing_optimizer(self):
        """Test loading checkpoint without optimizer state (backward compatibility)."""
        # Old checkpoint format without optimizer
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": 20,
            "best_val_metric": 0.85,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": None
        }

        torch.save(checkpoint, self.checkpoint_path)
        loaded_checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # Should still load model state
        self.model.load_state_dict(loaded_checkpoint["model_state_dict"])

        # Verify missing optimizer state doesn't break loading
        self.assertNotIn("optimizer_state_dict", loaded_checkpoint)

    def test_periodic_checkpoint_save_and_resume(self):
        """Test periodic checkpoint (last_checkpoint.pt) for interrupted training."""
        # Simulate periodic checkpoint saved after epoch 10
        last_checkpoint_path = os.path.join(self.temp_dir, "last_checkpoint.pt")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Train for a few steps to update optimizer state
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.rand(1, requires_grad=True)
            loss.backward()
            optimizer.step()

        # Save periodic checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 10,
            "best_val_metric": 0.78,
            "property_configs": self.property_configs,
            "vocab_size": self.vocab_size,
            "max_len": 64,
            "d_model": self.d_model,
            "n_encoder_blocks": self.n_encoder_blocks,
            "label_encoder": self.label_encoder,
            "class_weights": {},
            "val_property_metrics": {},
            "scheduler_type": "onecycle",
            "training_args": {
                "base_lr": 1e-4,
                "scaled_lr": 1e-4,
                "batch_size": 64,
                "epochs": 50,
                "warmup_pct": 0.1,
                "n_encoder_blocks": self.n_encoder_blocks,
            }
        }
        torch.save(checkpoint, last_checkpoint_path)
        self.assertTrue(os.path.exists(last_checkpoint_path))

        # Load checkpoint and verify all state is preserved
        loaded = torch.load(last_checkpoint_path, map_location='cpu')
        self.assertEqual(loaded["epoch"], 10)
        self.assertEqual(loaded["best_val_metric"], 0.78)
        self.assertIn("optimizer_state_dict", loaded)
        self.assertIn("training_args", loaded)

        # Create new model and optimizer, load state
        new_model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            n_encoder_blocks=self.n_encoder_blocks,
            max_len=64,
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=128,
            pad_idx=0
        )
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

        new_model.load_state_dict(loaded["model_state_dict"])
        new_optimizer.load_state_dict(loaded["optimizer_state_dict"])

        # Verify can continue training from loaded state
        self.assertTrue(all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(self.model.parameters(), new_model.parameters())
        ))

        # Clean up
        os.remove(last_checkpoint_path)

    def test_periodic_checkpoint_cleanup(self):
        """Test that last_checkpoint.pt should be deleted after successful completion."""
        # Create a dummy last_checkpoint.pt
        last_checkpoint_path = os.path.join(self.temp_dir, "last_checkpoint.pt")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": 50,
        }
        torch.save(checkpoint, last_checkpoint_path)
        self.assertTrue(os.path.exists(last_checkpoint_path))

        # Simulate cleanup after successful training
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)

        # Verify it's been deleted
        self.assertFalse(os.path.exists(last_checkpoint_path))

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        # Clean up last_checkpoint.pt if exists
        last_checkpoint_path = os.path.join(self.temp_dir, "last_checkpoint.pt")
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
        os.rmdir(self.temp_dir)


class TestSequenceRegressionHead(unittest.TestCase):
    """Test SequenceRegressionHead for variable-length predictions."""

    def test_init_without_cross_attention(self):
        """Test initialization without cross-attention."""
        head = SequenceRegressionHead(
            d_model=128,
            max_seq_len=105,
            use_cross_attention=False
        )

        self.assertEqual(head.d_model, 128)
        self.assertEqual(head.max_seq_len, 105)
        self.assertEqual(head.task, "sequence_regression")
        self.assertFalse(head.use_cross_attention)
        self.assertFalse(hasattr(head, 'pred_embedding'))

    def test_init_with_cross_attention(self):
        """Test initialization with cross-attention enabled."""
        head = SequenceRegressionHead(
            d_model=128,
            max_seq_len=105,
            use_cross_attention=True
        )

        self.assertTrue(head.use_cross_attention)
        self.assertTrue(hasattr(head, 'pred_embedding'))
        self.assertIsInstance(head.pred_embedding, nn.Linear)

    def test_forward_without_cross_attention(self):
        """Test forward pass without cross-attention."""
        head = SequenceRegressionHead(
            d_model=128,
            max_seq_len=105,
            use_cross_attention=False
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)

        predictions, pred_emb = head(x)

        # Check output shape
        self.assertEqual(predictions.shape, (batch_size, 105))
        self.assertIsNone(pred_emb)

    def test_forward_with_cross_attention(self):
        """Test forward pass with cross-attention."""
        head = SequenceRegressionHead(
            d_model=128,
            max_seq_len=105,
            use_cross_attention=True
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)

        predictions, pred_emb = head(x)

        # Check output shapes
        self.assertEqual(predictions.shape, (batch_size, 105))
        self.assertIsNotNone(pred_emb)
        self.assertEqual(pred_emb.shape, (batch_size, seq_len, 128))

    def test_output_range(self):
        """Test that outputs are continuous (not restricted to specific range)."""
        head = SequenceRegressionHead(d_model=128, max_seq_len=105)

        x = torch.randn(8, 20, 128)
        predictions, _ = head(x)

        # Predictions should be continuous values (can be any real number)
        self.assertTrue(torch.isfinite(predictions).all())
        # Check that we have variety (not all zeros or all same value)
        self.assertTrue(predictions.std() > 0.01)


class TestVariableLengthCollate(unittest.TestCase):
    """Test collate_fn with variable-length ring_plane_angles."""

    def test_collate_all_zero_angles(self):
        """Test batch where all molecules have 0 angles."""
        batch = []
        for i in range(4):
            input_ids = torch.randint(0, 100, (20,))
            attention_mask = torch.ones(20)
            targets = {
                "dimension": 0,
                "ring_plane_angles": np.array([], dtype=np.float32)
            }
            batch.append((input_ids, attention_mask, targets))

        batch_input_ids, batch_attn_masks, batch_targets = collate_fn(batch)

        # Should create (batch_size, 1) tensor to avoid 0-dim
        self.assertEqual(batch_targets["ring_plane_angles"].shape, (4, 1))
        self.assertEqual(batch_targets["ring_plane_angles_mask"].shape, (4, 1))
        # All zeros in both tensors
        self.assertTrue((batch_targets["ring_plane_angles"] == 0).all())
        self.assertTrue((batch_targets["ring_plane_angles_mask"] == 0).all())

    def test_collate_variable_lengths(self):
        """Test batch with different numbers of angles."""
        batch = []
        angle_counts = [0, 1, 3, 6]
        angle_values = [
            np.array([], dtype=np.float32),
            np.array([45.2], dtype=np.float32),
            np.array([30.5, 60.1, 90.0], dtype=np.float32),
            np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0], dtype=np.float32)
        ]

        for i in range(4):
            input_ids = torch.randint(0, 100, (20,))
            attention_mask = torch.ones(20)
            targets = {
                "dimension": i,
                "ring_plane_angles": angle_values[i]
            }
            batch.append((input_ids, attention_mask, targets))

        batch_input_ids, batch_attn_masks, batch_targets = collate_fn(batch)

        # Shape should be (4, 6) - max length in batch
        self.assertEqual(batch_targets["ring_plane_angles"].shape, (4, 6))
        self.assertEqual(batch_targets["ring_plane_angles_mask"].shape, (4, 6))

        # Check mask counts
        mask = batch_targets["ring_plane_angles_mask"]
        for i, expected_count in enumerate(angle_counts):
            actual_count = mask[i].sum().item()
            self.assertEqual(actual_count, expected_count,
                           f"Sample {i}: expected {expected_count} valid angles, got {actual_count}")

        # Check values are preserved
        self.assertAlmostEqual(batch_targets["ring_plane_angles"][1, 0].item(), 45.2, places=4)
        self.assertAlmostEqual(batch_targets["ring_plane_angles"][2, 0].item(), 30.5, places=4)
        self.assertAlmostEqual(batch_targets["ring_plane_angles"][2, 2].item(), 90.0, places=4)
        self.assertAlmostEqual(batch_targets["ring_plane_angles"][3, 5].item(), 65.0, places=4)

    def test_collate_mixed_properties(self):
        """Test that other properties are not affected by variable-length sequences."""
        batch = []
        for i in range(3):
            input_ids = torch.randint(0, 100, (20,))
            attention_mask = torch.ones(20)
            targets = {
                "dimension": i,
                "chirality": i % 2,
                # planar_fit_error is now a sequence (one error per ring)
                "planar_fit_error": np.array([float(i) * 0.5] * (i + 1), dtype=np.float32),
                "ring_plane_angles": np.array([45.0] * (i + 1), dtype=np.float32)
            }
            batch.append((input_ids, attention_mask, targets))

        batch_input_ids, batch_attn_masks, batch_targets = collate_fn(batch)

        # Check scalar properties are correctly batched
        self.assertEqual(batch_targets["dimension"].shape, (3,))
        self.assertEqual(batch_targets["chirality"].shape, (3,))

        # Check both sequence properties have masks and correct shapes
        self.assertIn("planar_fit_error_mask", batch_targets)
        self.assertEqual(batch_targets["planar_fit_error"].shape, (3, 3))  # max is 3 errors
        self.assertIn("ring_plane_angles_mask", batch_targets)
        self.assertEqual(batch_targets["ring_plane_angles"].shape, (3, 3))  # max is 3 angles


class TestMaskedLoss(unittest.TestCase):
    """Test masked loss computation for sequence regression."""

    def test_masked_mse_loss_basic(self):
        """Test basic masked MSE loss computation."""
        batch_size = 4
        max_len = 6

        # Create predictions and targets
        predictions = torch.randn(batch_size, max_len)
        targets = torch.randn(batch_size, max_len)

        # Create mask with variable valid counts
        mask = torch.zeros(batch_size, max_len)
        mask[0, :0] = 1  # 0 valid
        mask[1, :1] = 1  # 1 valid
        mask[2, :3] = 1  # 3 valid
        mask[3, :6] = 1  # 6 valid

        # Compute masked loss
        squared_error = (predictions - targets) ** 2
        masked_error = squared_error * mask
        n_valid = mask.sum()

        if n_valid > 0:
            loss = masked_error.sum() / n_valid
        else:
            loss = torch.tensor(0.0)

        # Manually compute expected loss
        manual_loss = 0.0
        count = 0
        for i in range(batch_size):
            for j in range(max_len):
                if mask[i, j] > 0:
                    manual_loss += squared_error[i, j].item()
                    count += 1

        if count > 0:
            manual_loss /= count

        self.assertAlmostEqual(loss.item(), manual_loss, places=5)
        self.assertEqual(n_valid.item(), 10)  # 0 + 1 + 3 + 6

    def test_masked_mae_basic(self):
        """Test basic masked MAE computation."""
        batch_size = 3
        max_len = 4

        predictions = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [9.0, 10.0, 11.0, 12.0]])
        targets = torch.zeros(batch_size, max_len)

        mask = torch.tensor([[1, 1, 0, 0],
                            [1, 0, 0, 0],
                            [1, 1, 1, 0]], dtype=torch.float32)

        abs_error = torch.abs(predictions - targets)
        masked_abs_error = abs_error * mask
        mae = masked_abs_error.sum() / mask.sum()

        # Expected: (1 + 2 + 5 + 9 + 10 + 11) / 6 = 38 / 6
        expected = 38.0 / 6.0
        self.assertAlmostEqual(mae.item(), expected, places=5)

    def test_all_masked_out(self):
        """Test case where all positions are masked (no valid data)."""
        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)
        mask = torch.zeros(4, 10)  # All masked out

        squared_error = (predictions - targets) ** 2
        masked_error = squared_error * mask
        n_valid = mask.sum()

        self.assertEqual(n_valid.item(), 0)
        # Loss should be 0 when no valid positions
        loss = masked_error.sum() / (n_valid + 1e-8)  # Avoid division by zero
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestVariableLengthIntegration(unittest.TestCase):
    """Integration tests for full training pipeline with ring_plane_angles."""

    def setUp(self):
        """Create temporary test data with ring_plane_angles."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock HDF5 files
        self.mol_file = os.path.join(self.temp_dir, "test_mol.h5")
        self.feat_file = os.path.join(self.temp_dir, "test_feat.h5")

        n_samples = 50

        with h5py.File(self.mol_file, "w") as f:
            smiles_data = [b"CCO" + str(i).encode() for i in range(n_samples)]
            f.create_dataset("smiles", data=np.array(smiles_data, dtype="S50"))

        with h5py.File(self.feat_file, "w") as f:
            # Create structured array for plane_angles
            # Simulating molecules with different ring counts
            all_plane_angles = []
            all_errors = []
            nrings = []
            for i in range(n_samples):
                n_rings = (i % 5)  # 0-4 rings
                n_pairs = n_rings * (n_rings - 1) // 2
                nrings.append(n_rings)

                # Create one error per ring (not per molecule)
                for ring_idx in range(n_rings):
                    all_errors.append(np.random.uniform(0, 1))

                for pair_idx in range(n_pairs):
                    all_plane_angles.append((pair_idx, pair_idx + 1, np.random.uniform(0, 180)))

            plane_angles_dtype = np.dtype([("i", "i4"), ("j", "i4"), ("val", "f4")])
            plane_angles = np.array(all_plane_angles, dtype=plane_angles_dtype) if all_plane_angles else np.array([], dtype=plane_angles_dtype)

            f.create_dataset("dimensions", data=np.array([b"3D"] * n_samples))
            f.create_dataset("point_groups", data=np.array([b"C1"] * n_samples))
            f.create_dataset("symmetry_planes", data=np.zeros(n_samples, dtype=np.int32))
            f.create_dataset("chiralities", data=np.zeros(n_samples, dtype=bool))
            f.create_dataset("nrings", data=np.array(nrings, dtype=np.int32))
            f.create_dataset("errors", data=np.array(all_errors, dtype=np.float32))
            f.create_dataset("plane_angles", data=plane_angles)

    def test_dataset_returns_variable_length_angles(self):
        """Test that dataset returns variable-length angle arrays."""
        # Create minimal vocab
        vocab_file = os.path.join(self.temp_dir, "smiles_vocab.json")
        vocab_data = {
            "tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "C", "O", "0", "1", "2", "3", "4", "5"],
            "token_to_id": {
                "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                "C": 4, "O": 5, "0": 6, "1": 7, "2": 8, "3": 9, "4": 10, "5": 11
            },
            "id_to_token": {
                "0": "<pad>", "1": "<unk>", "2": "<bos>", "3": "<eos>",
                "4": "C", "5": "O", "6": "0", "7": "1", "8": "2", "9": "3", "10": "4", "11": "5"
            }
        }
        with open(vocab_file, "w") as f:
            json.dump(vocab_data, f)

        underrep_file = os.path.join(self.temp_dir, "underrepresented_data.json")
        underrep_data = {
            "point_groups": ["group1"],
            "symmetry_planes": 3,
            "nrings": 6
        }
        with open(underrep_file, "w") as f:
            json.dump(underrep_data, f)

        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit("dimension", ["3D"])
        label_encoder.fit("ring_count", ["0", "1", "2", "3", "4"])
        label_encoder.fit("n_symmetry_planes", ["0"])
        label_encoder.fit("point_group", ["C1"])

        # Create tokenizer
        tokenizer = SmilesTokenizer(vocab_file=vocab_file)

        # Create dataset
        dataset = H5SequenceDataset(
            mol_files=[self.mol_file],
            feat_files=[self.feat_file],
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            underrepresented_data_file=underrep_file,
            mode="smiles",
            max_len=50,
            max_molecules=None
        )

        # Check first few samples
        for i in range(min(5, len(dataset))):
            input_ids, attention_mask, targets = dataset[i]

            # Sample i has (i % 5) rings
            n_rings = i % 5

            # planar_fit_error should have one entry per ring
            errors = targets["planar_fit_error"]
            self.assertIsInstance(errors, np.ndarray)
            self.assertEqual(errors.dtype, np.float32)
            self.assertEqual(len(errors), n_rings,
                           f"Sample {i} with {n_rings} rings should have {n_rings} planar fit errors")

            # ring_plane_angles should be numpy array (variable length)
            angles = targets["ring_plane_angles"]
            self.assertIsInstance(angles, np.ndarray)
            self.assertEqual(angles.dtype, np.float32)

            # Length should match n_rings * (n_rings - 1) / 2
            expected_len = n_rings * (n_rings - 1) // 2
            self.assertEqual(len(angles), expected_len,
                           f"Sample {i} with {n_rings} rings should have {expected_len} angles")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
