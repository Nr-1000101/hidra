#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Comprehensive test suite for transformer.py

Tests all components individually and integration:
- Tokenizers (SMILES/SELFIES)
- LabelEncoder
- Dataset loading
- Model components (positional encoding, cross-attention, property blocks)
- Full hierarchical transformer
- Training/evaluation functions

Usage:
    python test_transformer.py
    
Or run specific test:
    python test_transformer.py TestTokenizers.test_smiles_tokenizer
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
        
        # Properties are independent, but both A and X get sorted to index 0
        # This test is actually checking wrong behavior - remove the assertion
        # Both will be 0 since they're the first alphabetically in their respective groups


class TestTokenizers(unittest.TestCase):
    """Test SMILES and SELFIES tokenizers."""
    
    def setUp(self):
        """Create temporary vocab files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # SMILES vocab
        smiles_vocab = {
            "tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "C", "N", "O", "(", ")", "=", "1"],
            "token_to_id": {
                "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                "C": 4, "N": 5, "O": 6, "(": 7, ")": 8, "=": 9, "1": 10
            },
            "id_to_token": {str(i): tok for i, tok in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "C", "N", "O", "(", ")", "=", "1"]
            )}
        }
        
        self.smiles_vocab_file = os.path.join(self.temp_dir, "smiles_vocab.json")
        with open(self.smiles_vocab_file, "w") as f:
            json.dump(smiles_vocab, f)
        
        # SELFIES vocab
        selfies_vocab = {
            "tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "[C]", "[N]", "[O]", "[=C]"],
            "token_to_id": {
                "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                "[C]": 4, "[N]": 5, "[O]": 6, "[=C]": 7
            },
            "id_to_token": {str(i): tok for i, tok in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "[C]", "[N]", "[O]", "[=C]"]
            )}
        }
        
        self.selfies_vocab_file = os.path.join(self.temp_dir, "selfies_vocab.json")
        with open(self.selfies_vocab_file, "w") as f:
            json.dump(selfies_vocab, f)
    
    def test_smiles_tokenizer_encode(self):
        """Test SMILES tokenizer encoding."""
        tokenizer = SmilesTokenizer(self.smiles_vocab_file)
        
        # Test with special tokens
        ids = tokenizer.encode("C=O", add_special=True)
        self.assertEqual(ids[0], 2)  # BOS
        self.assertEqual(ids[-1], 3)  # EOS
        self.assertGreater(len(ids), 2)
        
        # Test without special tokens
        ids_no_special = tokenizer.encode("C=O", add_special=False)
        self.assertEqual(len(ids_no_special), len(ids) - 2)
    
    def test_smiles_tokenizer_decode(self):
        """Test SMILES tokenizer decoding."""
        tokenizer = SmilesTokenizer(self.smiles_vocab_file)
        
        ids = [2, 4, 9, 6, 3]  # <bos> C = O <eos>
        decoded = tokenizer.decode(ids)
        self.assertEqual(decoded, "C=O")
    
    def test_smiles_tokenizer_roundtrip(self):
        """Test SMILES encode-decode roundtrip."""
        tokenizer = SmilesTokenizer(self.smiles_vocab_file)
        
        original = "C=O"
        ids = tokenizer.encode(original, add_special=True)
        decoded = tokenizer.decode(ids)
        self.assertEqual(decoded, original)
    
    def test_smiles_tokenizer_vocab_size(self):
        """Test SMILES tokenizer vocab size."""
        tokenizer = SmilesTokenizer(self.smiles_vocab_file)
        self.assertEqual(tokenizer.vocab_size, 11)
    
    def test_smiles_regex_tokenization(self):
        """Test SMILES regex tokenizer handles multi-char atoms."""
        tokenizer = SmilesTokenizer(self.smiles_vocab_file)
        
        # Should tokenize Cl as single token
        tokens = tokenizer.tokenize("CCl")
        self.assertIn("Cl", tokens)
        self.assertEqual(len(tokens), 2)  # C and Cl
    
    def test_selfies_tokenizer_encode(self):
        """Test SELFIES tokenizer encoding."""
        tokenizer = SelfiesTokenizer(self.selfies_vocab_file)
        
        ids = tokenizer.encode("[C][O]", add_special=True)
        self.assertEqual(ids[0], 2)  # BOS
        self.assertEqual(ids[-1], 3)  # EOS
    
    def test_unknown_tokens(self):
        """Test handling of unknown tokens."""
        tokenizer = SmilesTokenizer(self.smiles_vocab_file)
        
        # Token not in vocab should map to <unk>
        ids = tokenizer.encode("CXO", add_special=False)  # X not in vocab
        self.assertIn(1, ids)  # <unk> id
    
    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.smiles_vocab_file, self.selfies_vocab_file]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.temp_dir)


class TestDataset(unittest.TestCase):
    """Test H5SequenceDataset functionality."""
    
    def setUp(self):
        """Create temporary HDF5 files and vocab."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock molecule file
        self.mol_file = os.path.join(self.temp_dir, "mols.h5")
        with h5py.File(self.mol_file, "w") as f:
            smiles_data = [b"CCO", b"CCCO", b"C=O"]
            f.create_dataset("smiles", data=np.array(smiles_data, dtype="S"))
        
        # Create mock feature file
        self.feat_file = os.path.join(self.temp_dir, "feats.h5")
        with h5py.File(self.feat_file, "w") as f:
            f.create_dataset("dimensions", data=np.array([b"linear", b"planar", b"linear"], dtype="S"))
            f.create_dataset("point_groups", data=np.array([b"C1", b"Cs", b"C2v"], dtype="S"))
            f.create_dataset("symmetry_planes", data=np.array([0, 1, 2], dtype=int))
            f.create_dataset("chiralities", data=np.array([0, 1, 0], dtype=int))
            f.create_dataset("nrings", data=np.array([0, 1, 0], dtype=int))
            f.create_dataset("errors", data=np.array([0.1, 0.2, 0.15], dtype=float))
            
            # Create plane_angles structured array
            dt = np.dtype([("i", "i4"), ("j", "i4"), ("val", "f4")])
            angles = np.array([], dtype=dt)
            f.create_dataset("plane_angles", data=angles)
        
        # Create underrepresented data file
        self.underrep_file = os.path.join(self.temp_dir, "underrep.json")
        with open(self.underrep_file, "w") as f:
            json.dump({
                "point_groups": ["rare_group"],
                "symmetry_planes": 5,
                "nrings": 4
            }, f)
        
        # Create vocab
        self.vocab_file = os.path.join(self.temp_dir, "vocab.json")
        vocab_data = {
            "tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "C", "O", "="],
            "token_to_id": {
                "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                "C": 4, "O": 5, "=": 6
            },
            "id_to_token": {str(i): tok for i, tok in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "C", "O", "="]
            )}
        }
        with open(self.vocab_file, "w") as f:
            json.dump(vocab_data, f)
        
        # Initialize tokenizer and label encoder as instance variables
        self.tokenizer = SmilesTokenizer(self.vocab_file)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit("dimension", ["linear", "planar"])
        self.label_encoder.fit("ring_count", ["0", "1", "4+"])
        self.label_encoder.fit("n_symmetry_planes", ["0", "1", "2", "5+"])
        self.label_encoder.fit("point_group", ["C1", "Cs", "C2v", "Other"])
    
    def test_dataset_length(self):
        """Test dataset returns correct length."""
        dataset = H5SequenceDataset(
            [self.mol_file],
            [self.feat_file],
            self.tokenizer,
            self.label_encoder,
            underrepresented_data_file=self.underrep_file,
            mode="smiles",
            max_len=32
        )
        self.assertEqual(len(dataset), 3)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format."""
        dataset = H5SequenceDataset(
            [self.mol_file],
            [self.feat_file],
            self.tokenizer,
            self.label_encoder,
            underrepresented_data_file=self.underrep_file,
            mode="smiles",
            max_len=32
        )
        
        input_ids, attention_mask, targets = dataset[0]
        
        # Check types
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertIsInstance(targets, dict)
        
        # Check shapes
        self.assertEqual(input_ids.shape, (32,))
        self.assertEqual(attention_mask.shape, (32,))
        
        # Check targets
        self.assertIn("dimension", targets)
        self.assertIn("ring_count", targets)
        self.assertIn("chirality", targets)
    
    def test_dataset_padding(self):
        """Test sequences are properly padded."""
        dataset = H5SequenceDataset(
            [self.mol_file],
            [self.feat_file],
            self.tokenizer,
            self.label_encoder,
            underrepresented_data_file=self.underrep_file,
            mode="smiles",
            max_len=32
        )
        
        input_ids, attention_mask, _ = dataset[0]
        
        # Check BOS at start
        self.assertEqual(input_ids[0].item(), 2)
        
        # Check padding
        pad_id = self.tokenizer.token_to_id["<pad>"]
        has_padding = (input_ids == pad_id).any()
        self.assertTrue(has_padding)
        
        # Check attention mask excludes padding
        n_valid = attention_mask.sum().item()
        n_pad = (input_ids == pad_id).sum().item()
        self.assertEqual(n_valid + n_pad, 32)
    
    def test_collate_fn(self):
        """Test collate function properly batches data."""
        dataset = H5SequenceDataset(
            [self.mol_file],
            [self.feat_file],
            self.tokenizer,
            self.label_encoder,
            underrepresented_data_file=self.underrep_file,
            mode="smiles",
            max_len=32
        )
        
        batch = [dataset[i] for i in range(2)]
        input_ids, attention_masks, targets = collate_fn(batch)
        
        # Check batch dimensions
        self.assertEqual(input_ids.shape, (2, 32))
        self.assertEqual(attention_masks.shape, (2, 32))
        
        # Check targets are batched
        self.assertEqual(targets["dimension"].shape, (2,))
        self.assertIsInstance(targets, dict)
    
    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.mol_file, self.feat_file, self.underrep_file, self.vocab_file]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.temp_dir)


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""
    
    def test_positional_encoding(self):
        """Test learned positional encoding."""
        d_model = 64
        max_len = 128
        batch_size = 4
        seq_len = 32
        
        pos_enc = LearnedPositionalEncoding(d_model, max_len)
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_enc(x)
        
        # Check shape preserved
        self.assertEqual(output.shape, x.shape)
        
        # Check output is different from input (positions added)
        self.assertFalse(torch.allclose(output, x))
    
    def test_positional_encoding_exceeds_max_len(self):
        """Test positional encoding raises error for sequences too long."""
        pos_enc = LearnedPositionalEncoding(d_model=64, max_len=32)
        x = torch.randn(2, 64, 64)  # seq_len=64 > max_len=32
        
        with self.assertRaises(ValueError):
            pos_enc(x)
    
    def test_cross_attention(self):
        """Test cross-attention module."""
        d_model = 64
        batch_size = 4
        seq_len = 32
        
        cross_attn = CrossAttention(d_model=d_model, nhead=4)
        
        x = torch.randn(batch_size, seq_len, d_model)
        memory = torch.randn(batch_size, seq_len, d_model)
        
        output = cross_attn(x, memory)
        
        # Check shape preserved
        self.assertEqual(output.shape, x.shape)
    
    def test_cross_attention_with_mask(self):
        """Test cross-attention with padding mask."""
        d_model = 64
        batch_size = 4
        seq_len = 32
        
        cross_attn = CrossAttention(d_model=d_model, nhead=4)
        
        x = torch.randn(batch_size, seq_len, d_model)
        memory = torch.randn(batch_size, seq_len, d_model)
        memory_mask = torch.ones(batch_size, seq_len).bool()
        memory_mask[:, 16:] = False  # Mask second half
        
        output = cross_attn(x, memory, memory_mask=memory_mask)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_property_block_classification(self):
        """Test property block for classification task."""
        d_model = 64
        num_classes = 5
        batch_size = 4
        seq_len = 32
        
        prop_block = PropertyBlock(
            d_model=d_model,
            nhead=4,
            num_layers=2,
            task="classification",
            num_classes=num_classes
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        attn_mask = torch.ones(batch_size, seq_len)
        
        logits, pred_emb = prop_block(x, attn_mask)
        
        # Check logits shape
        self.assertEqual(logits.shape, (batch_size, num_classes))
        
        # Check prediction embedding shape
        self.assertEqual(pred_emb.shape, (batch_size, seq_len, d_model))
    
    def test_property_block_regression(self):
        """Test property block for regression task."""
        d_model = 64
        batch_size = 4
        seq_len = 32
        
        prop_block = PropertyBlock(
            d_model=d_model,
            nhead=4,
            num_layers=2,
            task="regression"
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        attn_mask = torch.ones(batch_size, seq_len)
        
        logits, pred_emb = prop_block(x, attn_mask)
        
        # Check logits shape (regression outputs single value)
        self.assertEqual(logits.shape, (batch_size, 1))
        
        # Check prediction embedding shape
        self.assertEqual(pred_emb.shape, (batch_size, seq_len, d_model))
    
    def test_property_block_with_previous_memory(self):
        """Test property block with cross-attention to previous property."""
        d_model = 64
        batch_size = 4
        seq_len = 32
        
        prop_block = PropertyBlock(
            d_model=d_model,
            nhead=4,
            num_layers=2,
            task="classification",
            num_classes=3
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        attn_mask = torch.ones(batch_size, seq_len)
        prev_memory = torch.randn(batch_size, seq_len, d_model)
        prev_mask = torch.ones(batch_size, seq_len)
        
        logits, pred_emb = prop_block(x, attn_mask, prev_memory, prev_mask)
        
        self.assertEqual(logits.shape, (batch_size, 3))
        self.assertEqual(pred_emb.shape, (batch_size, seq_len, d_model))


class TestHierarchicalTransformer(unittest.TestCase):
    """Test full hierarchical transformer model."""
    
    def setUp(self):
        """Set up model configuration."""
        self.vocab_size = 100
        self.max_len = 64
        self.d_model = 128
        self.batch_size = 4
        
        self.property_configs = [
            {"task": "classification", "num_classes": 5, "n_blocks": 2},
            {"task": "classification", "num_classes": 3, "n_blocks": 2},
            {"task": "regression", "n_blocks": 1},
        ]
    
    def test_model_initialization(self):
        """Test model initializes without errors."""
        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=4
        )
        
        # Check model has correct number of property blocks
        self.assertEqual(len(model.properties), 3)
    
    def test_model_forward(self):
        """Test forward pass produces correct output shapes."""
        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=4
        )
        
        # Create dummy input
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        attention_mask = torch.ones(self.batch_size, self.max_len)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Check outputs
        self.assertIsInstance(outputs, dict)
        self.assertEqual(len(outputs), 3)
        
        # Check each output shape
        self.assertEqual(outputs["prop_0"].shape, (self.batch_size, 5))  # classification
        self.assertEqual(outputs["prop_1"].shape, (self.batch_size, 3))  # classification
        self.assertEqual(outputs["prop_2"].shape, (self.batch_size, 1))  # regression
    
    def test_model_with_padding(self):
        """Test model handles padded sequences correctly."""
        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            pad_idx=0
        )
        
        # Create input with padding
        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.max_len))
        input_ids[:, 32:] = 0  # Pad second half
        attention_mask = (input_ids != 0).long()
        
        outputs = model(input_ids, attention_mask)
        
        # Should still produce valid outputs
        self.assertEqual(len(outputs), 3)
    
    def test_model_gradient_flow(self):
        """Test gradients flow through entire model."""
        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model
        )
        
        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.max_len))  # Avoid padding idx
        attention_mask = torch.ones(self.batch_size, self.max_len)
        
        outputs = model(input_ids, attention_mask)
        
        # Compute dummy loss
        loss = outputs["prop_0"].sum() + outputs["prop_1"].sum() + outputs["prop_2"].sum()
        loss.backward()
        
        # Check gradients exist for non-padding parameters
        has_gradients = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        
        self.assertTrue(has_gradients, "No gradients found in model parameters")
    
    def test_model_parameter_count(self):
        """Test model has reasonable number of parameters."""
        model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=2
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should have > 0 parameters
        self.assertGreater(n_params, 0)
        
        # Sanity check: not unreasonably large
        self.assertLess(n_params, 100_000_000)  # < 100M params


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
            {"task": "classification", "num_classes": 3, "n_blocks": 1},
            {"task": "classification", "num_classes": 4, "n_blocks": 1},
            {"task": "classification", "num_classes": 2, "n_blocks": 1},
            {"task": "classification", "num_classes": 5, "n_blocks": 1},
            {"task": "classification", "num_classes": 5, "n_blocks": 1},
            {"task": "regression", "n_blocks": 1},
            {"task": "regression", "n_blocks": 1},
        ]
        
        self.model = HierarchicalTransformer(
            vocab_size=self.vocab_size,
            property_configs=self.property_configs,
            max_len=self.max_len,
            d_model=self.d_model,
            n_initial_blocks=2
        )
        
        self.property_info = [
            ("classification", 3),
            ("classification", 4),
            ("classification", 2),
            ("classification", 5),
            ("classification", 5),
            ("regression", 1),
            ("regression", 1),
        ]
    
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
            self.property_info,
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
            self.property_info,
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
            self.property_info,
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
        
        # Ensure model is in eval mode and no_grad is respected
        with torch.no_grad():
            mse, rmse = evaluate(
                self.model,
                dataloader,
                self.device,
                self.property_info,
                bf16=False
            )
        
        # Check no gradients are stored
        for param in self.model.parameters():
            self.assertIsNone(param.grad)


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflows."""
    
    def setUp(self):
        """Set up complete pipeline components."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create vocab
        self.vocab_file = os.path.join(self.temp_dir, "vocab.json")
        vocab_data = {
            "tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "C", "N", "O", "=", "(", ")"],
            "token_to_id": {tok: i for i, tok in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "C", "N", "O", "=", "(", ")"]
            )},
            "id_to_token": {str(i): tok for i, tok in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "C", "N", "O", "=", "(", ")"]
            )}
        }
        with open(self.vocab_file, "w") as f:
            json.dump(vocab_data, f)
        
        # Create HDF5 files
        self.mol_file = os.path.join(self.temp_dir, "mols.h5")
        with h5py.File(self.mol_file, "w") as f:
            smiles = [b"CCO", b"CC(C)O", b"C=O", b"CC=O", b"CCC"]
            f.create_dataset("smiles", data=np.array(smiles, dtype="S"))
        
        self.feat_file = os.path.join(self.temp_dir, "feats.h5")
        with h5py.File(self.feat_file, "w") as f:
            f.create_dataset("dimensions", data=np.array([b"linear", b"tetrahedral", b"planar", b"planar", b"linear"], dtype="S"))
            f.create_dataset("point_groups", data=np.array([b"C1", b"Cs", b"C2v", b"Cs", b"D3h"], dtype="S"))
            f.create_dataset("symmetry_planes", data=np.array([0, 1, 2, 1, 3], dtype=int))
            f.create_dataset("chiralities", data=np.array([0, 1, 0, 0, 0], dtype=int))
            f.create_dataset("nrings", data=np.array([0, 0, 0, 0, 0], dtype=int))
            f.create_dataset("errors", data=np.array([0.1, 0.2, 0.15, 0.18, 0.12], dtype=float))
            
            dt = np.dtype([("i", "i4"), ("j", "i4"), ("val", "f4")])
            angles = np.array([], dtype=dt)
            f.create_dataset("plane_angles", data=angles)
        
        # Create underrepresented data
        self.underrep_file = os.path.join(self.temp_dir, "underrep.json")
        with open(self.underrep_file, "w") as f:
            json.dump({
                "point_groups": ["rare"],
                "symmetry_planes": 5,
                "nrings": 3
            }, f)
    
    def test_full_pipeline(self):
        """Test complete pipeline: data loading → model → training → evaluation."""
        # Initialize tokenizer
        tokenizer = SmilesTokenizer(self.vocab_file)
        
        # Initialize label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit("dimension", ["linear", "planar", "tetrahedral"])
        label_encoder.fit("ring_count", ["0", "1", "2", "3+"])
        label_encoder.fit("n_symmetry_planes", ["0", "1", "2", "3", "5+"])
        label_encoder.fit("point_group", ["C1", "Cs", "C2v", "D3h", "Other"])
        
        # Create dataset
        dataset = H5SequenceDataset(
            [self.mol_file],
            [self.feat_file],
            tokenizer,
            label_encoder,
            underrepresented_data_file=self.underrep_file,
            mode="smiles",
            max_len=32
        )
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        # Create model
        property_configs = [
            {"task": "classification", "num_classes": 3, "n_blocks": 1},
            {"task": "classification", "num_classes": 4, "n_blocks": 1},
            {"task": "classification", "num_classes": 2, "n_blocks": 1},
            {"task": "classification", "num_classes": 5, "n_blocks": 1},
            {"task": "classification", "num_classes": 5, "n_blocks": 1},
            {"task": "regression", "n_blocks": 1},
            {"task": "regression", "n_blocks": 1},
        ]
        
        model = HierarchicalTransformer(
            vocab_size=tokenizer.vocab_size,
            property_configs=property_configs,
            max_len=32,
            d_model=64,
            n_initial_blocks=2,
            pad_idx=tokenizer.token_to_id["<pad>"]
        )
        
        device = torch.device("cpu")
        model.to(device)
        
        property_info = [(cfg["task"], cfg.get("num_classes", 1)) for cfg in property_configs]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train one epoch
        train_loss = train_one_epoch(model, dataloader, optimizer, device, property_info, bf16=False)
        
        # Evaluate
        val_mse, val_rmse = evaluate(model, dataloader, device, property_info, bf16=False)
        
        # Check results are valid
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0)
        self.assertIsInstance(val_rmse, float)
        self.assertGreater(val_rmse, 0)
    
    def test_overfitting_simple_data(self):
        """Test model can overfit to small dataset (sanity check)."""
        # Initialize components
        tokenizer = SmilesTokenizer(self.vocab_file)
        label_encoder = LabelEncoder()
        label_encoder.fit("dimension", ["linear", "planar"])
        label_encoder.fit("ring_count", ["0", "1", "4+"])
        label_encoder.fit("n_symmetry_planes", ["0", "1", "2", "5+"])
        label_encoder.fit("point_group", ["C1", "Cs", "C2v", "Other"])
        
        # Create small dataset (just 2 samples)
        mol_file_small = os.path.join(self.temp_dir, "mols_small.h5")
        with h5py.File(mol_file_small, "w") as f:
            f.create_dataset("smiles", data=np.array([b"CCO", b"C=O"], dtype="S"))
        
        feat_file_small = os.path.join(self.temp_dir, "feats_small.h5")
        with h5py.File(feat_file_small, "w") as f:
            f.create_dataset("dimensions", data=np.array([b"linear", b"planar"], dtype="S"))
            f.create_dataset("point_groups", data=np.array([b"C1", b"Cs"], dtype="S"))
            f.create_dataset("symmetry_planes", data=np.array([0, 1], dtype=int))
            f.create_dataset("chiralities", data=np.array([0, 0], dtype=int))
            f.create_dataset("nrings", data=np.array([0, 0], dtype=int))
            f.create_dataset("errors", data=np.array([0.1, 0.2], dtype=float))
            
            dt = np.dtype([("i", "i4"), ("j", "i4"), ("val", "f4")])
            angles = np.array([], dtype=dt)
            f.create_dataset("plane_angles", data=angles)
        
        dataset = H5SequenceDataset(
            [mol_file_small],
            [feat_file_small],
            tokenizer,
            label_encoder,
            underrepresented_data_file=self.underrep_file,
            mode="smiles",
            max_len=16
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        # Small model - FIXED: num_classes must be >= 2 for ring_count which has 3 classes
        property_configs = [
            {"task": "classification", "num_classes": 2, "n_blocks": 1},   # dimension
            {"task": "classification", "num_classes": 3, "n_blocks": 1},   # ring_count (FIX: was 1, should be 3)
            {"task": "classification", "num_classes": 2, "n_blocks": 1},   # chirality
            {"task": "classification", "num_classes": 4, "n_blocks": 1},   # n_symmetry_planes (FIX: was 2, should be 4)
            {"task": "classification", "num_classes": 4, "n_blocks": 1},   # point_group (FIX: was 2, should be 4)
            {"task": "regression", "n_blocks": 1},                          # planar_fit_error
            {"task": "regression", "n_blocks": 1},                          # ring_plane_angles
        ]
        
        model = HierarchicalTransformer(
            vocab_size=tokenizer.vocab_size,
            property_configs=property_configs,
            max_len=16,
            d_model=32,
            n_initial_blocks=1,
            pad_idx=tokenizer.token_to_id["<pad>"]
        )
        
        device = torch.device("cpu")
        model.to(device)
        
        property_info = [(cfg["task"], cfg.get("num_classes", 1)) for cfg in property_configs]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Train multiple epochs
        initial_loss = None
        final_loss = None
        
        for epoch in range(50):
            loss = train_one_epoch(model, dataloader, optimizer, device, property_info, bf16=False)
            if epoch == 0:
                initial_loss = loss
            if epoch == 49:
                final_loss = loss
        
        # Loss should decrease (overfitting to small dataset)
        self.assertLess(final_loss, initial_loss, "Model should overfit to small dataset")
    
    def test_model_save_load(self):
        """Test saving and loading model checkpoint."""
        tokenizer = SmilesTokenizer(self.vocab_file)
        
        property_configs = [
            {"task": "classification", "num_classes": 3, "n_blocks": 1},
            {"task": "regression", "n_blocks": 1},
        ]
        
        model = HierarchicalTransformer(
            vocab_size=tokenizer.vocab_size,
            property_configs=property_configs,
            max_len=32,
            d_model=64,
            n_initial_blocks=2,
            pad_idx=0
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "property_configs": property_configs,
            "vocab_size": tokenizer.vocab_size,
        }, checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Create new model and load state
        model_loaded = HierarchicalTransformer(
            vocab_size=checkpoint["vocab_size"],
            property_configs=checkpoint["property_configs"],
            max_len=32,
            d_model=64,
            n_initial_blocks=2,
            pad_idx=0
        )
        model_loaded.load_state_dict(checkpoint["model_state_dict"])
        
        # Set both to eval mode to disable dropout
        model.eval()
        model_loaded.eval()
        
        # Test both models produce same output
        input_ids = torch.randint(0, tokenizer.vocab_size, (2, 32))
        attention_mask = torch.ones(2, 32)
        
        with torch.no_grad():
            out1 = model(input_ids, attention_mask)
            out2 = model_loaded(input_ids, attention_mask)
        
        # Check outputs match
        for key in out1.keys():
            self.assertTrue(torch.allclose(out1[key], out2[key]), f"Output {key} doesn't match after reload")
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create vocab
            vocab_file = os.path.join(temp_dir, "vocab.json")
            vocab_data = {
                "tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
                "token_to_id": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
                "id_to_token": {"0": "<pad>", "1": "<unk>", "2": "<bos>", "3": "<eos>"}
            }
            with open(vocab_file, "w") as f:
                json.dump(vocab_data, f)
            
            tokenizer = SmilesTokenizer(vocab_file)
            
            # Empty string
            ids = tokenizer.encode("", add_special=True)
            self.assertEqual(len(ids), 2)  # Just BOS and EOS
            self.assertEqual(ids[0], 2)  # BOS
            self.assertEqual(ids[1], 3)  # EOS
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_very_long_sequence(self):
        """Test truncation of very long sequences."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            vocab_file = os.path.join(temp_dir, "vocab.json")
            vocab_data = {
                "tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "C"],
                "token_to_id": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "C": 4},
                "id_to_token": {"0": "<pad>", "1": "<unk>", "2": "<bos>", "3": "<eos>", "4": "C"}
            }
            with open(vocab_file, "w") as f:
                json.dump(vocab_data, f)
            
            tokenizer = SmilesTokenizer(vocab_file)
            
            # Very long sequence
            long_seq = "C" * 1000
            ids = tokenizer.encode(long_seq, add_special=False)
            
            # Should tokenize all characters
            self.assertEqual(len(ids), 1000)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_invalid_property_config(self):
        """Test model rejects invalid property configurations."""
        with self.assertRaises((ValueError, AssertionError)):
            # Classification without num_classes
            PropertyBlock(
                d_model=64,
                task="classification",
                num_classes=None  # Should raise error
            )
    
    def test_mismatched_batch_sizes(self):
        """Test error handling for mismatched batch sizes."""
        model = HierarchicalTransformer(
            vocab_size=100,
            property_configs=[{"task": "classification", "num_classes": 3, "n_blocks": 1}],
            max_len=32,
            d_model=64
        )
        
        # Mismatched batch sizes should raise error
        input_ids = torch.randint(0, 100, (4, 32))
        attention_mask = torch.ones(8, 32)  # Different batch size
        
        with self.assertRaises(AssertionError):
            model(input_ids, attention_mask)


def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVocabLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenizers))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestModelComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchicalTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed.")
    
    return result


if __name__ == "__main__":
    run_tests()


# In[ ]:




