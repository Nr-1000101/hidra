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

# Import all components from transformer
from transformer import (
    load_vocab,
    LabelEncoder,
    SelfiesTokenizer,
    SmilesTokenizer,
    H5SequenceDataset,
    collate_fn,
    LearnedPositionalEncoding,
    CrossAttention,
    PropertyHead,
    HierarchicalTransformer,
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
            {"name": "planar_fit_error", "task": "regression", "attach_at_block": 1},
            {"name": "ring_plane_angles", "task": "regression", "attach_at_block": 1},
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
                targets = {
                    "dimension": torch.randint(0, 3, (1,)).item(),
                    "ring_count": torch.randint(0, 4, (1,)).item(),
                    "chirality": torch.randint(0, 2, (1,)).item(),
                    "n_symmetry_planes": torch.randint(0, 5, (1,)).item(),
                    "point_group": torch.randint(0, 5, (1,)).item(),
                    "planar_fit_error": torch.randn(1).item(),
                    "ring_plane_angles": torch.randint(0, 5, (1,)).item(),
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
                "planar_fit_error": torch.tensor([b[2]["planar_fit_error"] for b in batch]),
                "ring_plane_angles": torch.tensor([b[2]["ring_plane_angles"] for b in batch]),
            }
            return input_ids, attention_masks, targets

        from torch.utils.data import DataLoader
        dataset = DummyDataset(n_batches * self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate)

    def test_train_one_epoch(self):
        """Test training for one epoch runs without errors."""
        dataloader = self.create_dummy_dataloader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        loss = train_one_epoch(
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

    def test_evaluate(self):
        """Test evaluation runs without errors."""
        dataloader = self.create_dummy_dataloader()

        mse, rmse = evaluate(
            self.model,
            dataloader,
            self.device,
            self.property_configs,
            bf16=False
        )

        # Check metrics are valid
        self.assertIsInstance(mse, float)
        self.assertIsInstance(rmse, float)
        self.assertGreater(mse, 0)
        self.assertGreater(rmse, 0)
        self.assertAlmostEqual(rmse, mse ** 0.5, places=5)

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


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
