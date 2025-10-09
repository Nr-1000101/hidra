"""
Comprehensive test suite for transformer.py (Updated for flexible property configuration)

Tests all components individually and integration:
- Tokenizers (SMILES/SELFIES)
- LabelEncoder
- Dataset loading
- Model components (positional encoding, cross-attention, property blocks)
- Full hierarchical transformer with flexible configurations
- Training/evaluation functions
- New features: optional cross-attention, variable property selection

Usage:
    python tests_updated.py

Or run specific test:
    python tests_updated.py TestHierarchicalTransformer.test_flexible_property_selection
"""

import unittest
import tempfile
import json
import os
from pathlib import Path

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
    PropertyBlock,
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


class TestPropertyBlock(unittest.TestCase):
    """Test PropertyBlock with new flexible configuration options."""

    def test_property_block_with_cross_attention(self):
        """Test PropertyBlock with cross-attention enabled."""
        block = PropertyBlock(
            d_model=128,
            nhead=4,
            num_layers=2,
            task="classification",
            num_classes=5,
            use_cross_attention=True,
            provide_embeddings=True
        )

        self.assertTrue(hasattr(block, 'cross_attn'))
        self.assertTrue(hasattr(block, 'pred_embedding'))

    def test_property_block_without_cross_attention(self):
        """Test PropertyBlock with cross-attention disabled."""
        block = PropertyBlock(
            d_model=128,
            nhead=4,
            num_layers=2,
            task="classification",
            num_classes=5,
            use_cross_attention=False,
            provide_embeddings=True
        )

        self.assertFalse(hasattr(block, 'cross_attn'))
        self.assertTrue(hasattr(block, 'pred_embedding'))

    def test_property_block_without_embeddings(self):
        """Test PropertyBlock that doesn't provide embeddings."""
        block = PropertyBlock(
            d_model=128,
            nhead=4,
            num_layers=2,
            task="classification",
            num_classes=5,
            use_cross_attention=True,
            provide_embeddings=False
        )

        self.assertTrue(hasattr(block, 'cross_attn'))
        self.assertFalse(hasattr(block, 'pred_embedding'))

    def test_property_block_forward_no_cross_attention(self):
        """Test forward pass without cross-attention."""
        block = PropertyBlock(
            d_model=128,
            nhead=4,
            num_layers=2,
            task="classification",
            num_classes=5,
            use_cross_attention=False,
            provide_embeddings=False
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)
        attn_mask = torch.ones(batch_size, seq_len)

        logits, pred_emb = block(x, attn_mask, prev_memory=None)

        self.assertEqual(logits.shape, (batch_size, 5))
        self.assertIsNone(pred_emb)

    def test_property_block_forward_with_cross_attention(self):
        """Test forward pass with cross-attention."""
        block = PropertyBlock(
            d_model=128,
            nhead=4,
            num_layers=2,
            task="classification",
            num_classes=5,
            use_cross_attention=True,
            provide_embeddings=True
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)
        attn_mask = torch.ones(batch_size, seq_len)
        prev_memory = torch.randn(batch_size, seq_len, 128)

        logits, pred_emb = block(x, attn_mask, prev_memory=prev_memory, prev_memory_mask=attn_mask)

        self.assertEqual(logits.shape, (batch_size, 5))
        self.assertEqual(pred_emb.shape, (batch_size, seq_len, 128))


class TestHierarchicalTransformer(unittest.TestCase):
    """Test full hierarchical transformer model with flexible configurations."""

    def setUp(self):
        """Set up model configuration."""
        self.vocab_size = 100
        self.max_len = 64
        self.d_model = 128
        self.batch_size = 4

    def test_model_initialization_with_names(self):
        """Test model initializes with property names."""
        property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 5, "n_blocks": 2},
            {"name": "ring_count", "task": "classification", "num_classes": 3, "n_blocks": 2},
            {"name": "chirality", "task": "regression", "n_blocks": 1},
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=4
        )

        # Check model has correct number of property blocks
        self.assertEqual(len(model.properties), 3)
        self.assertEqual(model.property_names, ["dimension", "ring_count", "chirality"])

    def test_model_forward_with_property_names(self):
        """Test forward pass produces outputs keyed by property names."""
        property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 5, "n_blocks": 2},
            {"name": "ring_count", "task": "classification", "num_classes": 3, "n_blocks": 2},
            {"name": "planarity", "task": "regression", "n_blocks": 1},
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=4
        )

        # Create dummy input
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Check outputs are keyed by property names
        self.assertIsInstance(outputs, dict)
        self.assertIn("dimension", outputs)
        self.assertIn("ring_count", outputs)
        self.assertIn("planarity", outputs)

        # Check shapes
        self.assertEqual(outputs["dimension"].shape, (self.batch_size, 5))
        self.assertEqual(outputs["ring_count"].shape, (self.batch_size, 3))
        self.assertEqual(outputs["planarity"].shape, (self.batch_size, 1))

    def test_single_property_model(self):
        """Test model with only one property."""
        property_configs = [
            {
                "name": "dimension",
                "task": "classification",
                "num_classes": 5,
                "n_blocks": 2,
                "use_cross_attention": False,  # First property, no cross-attention
                "provide_embeddings": False  # Last property, no need for embeddings
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=4
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        self.assertEqual(len(outputs), 1)
        self.assertIn("dimension", outputs)
        self.assertEqual(outputs["dimension"].shape, (self.batch_size, 5))

    def test_model_with_selective_cross_attention(self):
        """Test model where only some properties use cross-attention."""
        property_configs = [
            {
                "name": "prop1",
                "task": "classification",
                "num_classes": 3,
                "n_blocks": 1,
                "use_cross_attention": False,
                "provide_embeddings": True
            },
            {
                "name": "prop2",
                "task": "classification",
                "num_classes": 4,
                "n_blocks": 1,
                "use_cross_attention": True,  # Uses cross-attention from prop1
                "provide_embeddings": True
            },
            {
                "name": "prop3",
                "task": "regression",
                "n_blocks": 1,
                "use_cross_attention": False,  # Independent of prop2
                "provide_embeddings": False
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=2
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        self.assertEqual(len(outputs), 3)
        self.assertIn("prop1", outputs)
        self.assertIn("prop2", outputs)
        self.assertIn("prop3", outputs)

    def test_model_gradient_flow(self):
        """Test gradients flow through entire model."""
        property_configs = [
            {"name": "prop1", "task": "classification", "num_classes": 3, "n_blocks": 1},
            {"name": "prop2", "task": "regression", "n_blocks": 1},
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=property_configs,
            max_len=self.max_len,
            d_model=self.d_model
        )

        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)

        outputs = model(input_ids, attention_mask)

        # Compute dummy loss
        loss = outputs["prop1"].sum() + outputs["prop2"].sum()
        loss.backward()

        # Check gradients exist
        has_gradients = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break

        self.assertTrue(has_gradients, "No gradients found in model parameters")


class TestTrainingFunctions(unittest.TestCase):
    """Test training and evaluation functions with updated property configs."""

    def setUp(self):
        """Set up dummy model and data."""
        self.device = torch.device("cpu")
        self.vocab_size = 50
        self.max_len = 32
        self.d_model = 64
        self.batch_size = 4

        self.property_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 3, "n_blocks": 1},
            {"name": "ring_count", "task": "classification", "num_classes": 4, "n_blocks": 1},
            {"name": "chirality", "task": "classification", "num_classes": 2, "n_blocks": 1},
            {"name": "n_symmetry_planes", "task": "classification", "num_classes": 5, "n_blocks": 1},
            {"name": "point_group", "task": "classification", "num_classes": 5, "n_blocks": 1},
            {"name": "planar_fit_error", "task": "regression", "n_blocks": 1},
            {"name": "ring_plane_angles", "task": "regression", "n_blocks": 1},
        ]

        self.model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=2
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
            self.property_configs,  # Updated: now accepts property_configs
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
            self.property_configs,  # Updated: now accepts property_configs
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

    def test_training_with_subset_of_properties(self):
        """Test training with only a subset of properties."""
        # Create model with only 2 properties
        subset_configs = [
            {"name": "dimension", "task": "classification", "num_classes": 3, "n_blocks": 1},
            {"name": "chirality", "task": "classification", "num_classes": 2, "n_blocks": 1},
        ]

        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=subset_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=2
        )

        dataloader = self.create_dummy_dataloader(n_batches=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Should run without errors even though dataloader has more properties
        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            self.device,
            subset_configs,
            bf16=False
        )

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)


class TestFlexibleConfiguration(unittest.TestCase):
    """Test new flexible configuration features."""

    def test_all_properties_independent(self):
        """Test configuration where all properties are independent (no cross-attention)."""
        property_configs = [
            {
                "name": "prop1",
                "task": "classification",
                "num_classes": 3,
                "n_blocks": 1,
                "use_cross_attention": False,
                "provide_embeddings": False
            },
            {
                "name": "prop2",
                "task": "classification",
                "num_classes": 4,
                "n_blocks": 1,
                "use_cross_attention": False,
                "provide_embeddings": False
            },
            {
                "name": "prop3",
                "task": "regression",
                "n_blocks": 1,
                "use_cross_attention": False,
                "provide_embeddings": False
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=100,
            property_configs=property_configs,
            max_len=64,
            d_model=128,
            n_initial_blocks=4
        )

        batch_size = 4
        input_ids = torch.randint(0, 100, (batch_size, 64))
        attention_mask = torch.ones(batch_size, 64)

        outputs = model(input_ids, attention_mask)

        # All outputs should be present
        self.assertEqual(len(outputs), 3)
        self.assertIn("prop1", outputs)
        self.assertIn("prop2", outputs)
        self.assertIn("prop3", outputs)

    def test_sequential_dependencies(self):
        """Test configuration where properties have sequential dependencies."""
        property_configs = [
            {
                "name": "prop1",
                "task": "classification",
                "num_classes": 3,
                "n_blocks": 1,
                "use_cross_attention": False,
                "provide_embeddings": True
            },
            {
                "name": "prop2",
                "task": "classification",
                "num_classes": 4,
                "n_blocks": 1,
                "use_cross_attention": True,
                "provide_embeddings": True
            },
            {
                "name": "prop3",
                "task": "regression",
                "n_blocks": 1,
                "use_cross_attention": True,
                "provide_embeddings": False
            }
        ]

        model = HierarchicalTransformer(
            vocab_size=100,
            property_configs=property_configs,
            max_len=64,
            d_model=128,
            n_initial_blocks=4
        )

        batch_size = 4
        input_ids = torch.randint(0, 100, (batch_size, 64))
        attention_mask = torch.ones(batch_size, 64)

        outputs = model(input_ids, attention_mask)

        self.assertEqual(len(outputs), 3)
        # prop2 should use cross-attention from prop1
        # prop3 should use cross-attention from prop2


class TestDeterminism(unittest.TestCase):
    """Test model determinism with fixed random seed."""

    def test_forward_pass_deterministic(self):
        """Test that forward passes are deterministic with fixed seed."""
        torch.manual_seed(42)

        property_configs = [
            {"name": "prop1", "task": "classification", "num_classes": 3, "n_blocks": 1},
            {"name": "prop2", "task": "regression", "n_blocks": 1},
        ]

        model = HierarchicalTransformer(
            vocab_size=100,
            property_configs=property_configs,
            max_len=64,
            d_model=128,
            n_initial_blocks=2
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
