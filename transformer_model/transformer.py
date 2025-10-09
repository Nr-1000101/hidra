#!/usr/bin/env python
# coding: utf-8

"""
HIDRA: Flexible Transformer for predicting molecular properties from SMILES/SELFIES.

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
import random
import re
from typing import List, Dict, Tuple, Optional

import h5py
import json
import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Vocab Loader ---
def load_vocab(vocab_file: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Load vocabulary from JSON file.

    Args:
        vocab_file: Path to vocab JSON file

    Returns:
        tokens: List of all tokens
        token_to_id: Token string to ID mapping
        id_to_token: ID to token string mapping
    """
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    return vocab["tokens"], vocab["token_to_id"], {int(k): v for k, v in vocab["id_to_token"].items()}


# --- Label Encoder ---
class LabelEncoder:
    """Encodes string labels to integer indices for classification tasks."""
    
    def __init__(self):
        self.label_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_label: Dict[str, Dict[int, str]] = {}
    
    def fit(self, property_name: str, labels: List[str]):
        """Create encoding for a property's labels.
        
        Args:
            property_name: Name of the property (e.g., 'dimension')
            labels: List of unique string labels
        """
        unique_labels = sorted(set(labels))
        self.label_to_idx[property_name] = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label[property_name] = {idx: label for idx, label in enumerate(unique_labels)}
    
    def transform(self, property_name: str, label: str) -> int:
        """Convert label to index."""
        return self.label_to_idx[property_name].get(label, 0)  # Default to 0 if unknown
    
    def inverse_transform(self, property_name: str, idx: int) -> str:
        """Convert index back to label."""
        if property_name not in self.idx_to_label:
            return "unknown"
        return self.idx_to_label[property_name].get(idx, "unknown")
    
    def get_num_classes(self, property_name: str) -> int:
        """Get number of classes for a property."""
        return len(self.label_to_idx.get(property_name, {}))


# --- SELFIES Tokenizer ---
class SelfiesTokenizer:
    """Tokenizer for SELFIES molecular representations."""
    
    def __init__(self, vocab_file: str):
        """Initialize tokenizer from vocabulary file.
        
        Args:
            vocab_file: Path to SELFIES vocabulary JSON
        """
        self.tokens, self.token_to_id, self.id_to_token = load_vocab(vocab_file)
        self.reserved_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def encode(self, s: str, add_special: bool = True) -> List[int]:
        """Encode SELFIES string to token IDs.
        
        Args:
            s: SELFIES string
            add_special: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        toks = list(sf.split_selfies(s))
        ids = [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in toks]
        if add_special:
            return [self.token_to_id["<bos>"]] + ids + [self.token_to_id["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to SELFIES string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            SELFIES string
        """
        toks = [self.id_to_token.get(i, "?") for i in ids]
        toks = [t for t in toks if t not in self.reserved_tokens]
        return sf.decoder("".join(toks))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)


# --- SMILES Tokenizer ---
SMILES_REGEX = re.compile(
    r"""([A-Z][a-z]?               # single and multi-letter atoms
        | \[ [^\]]+ \]             # bracketed expressions
        | @@?                      # stereochemistry @ / @@
        | [=#\+/\\-]               # bonds (hyphen at end to avoid range)
        | \(|\)|\.|\*              # parentheses, dot, wildcard
        | \d                       # single-digit ring closures
        | %\d{2}                   # two-digit ring closures
    )""",
    re.X
)

class SmilesTokenizer:
    """Tokenizer for SMILES molecular representations."""
    
    def __init__(self, vocab_file: str):
        """Initialize tokenizer from vocabulary file.
        
        Args:
            vocab_file: Path to SMILES vocabulary JSON
        """
        self.tokens, self.token_to_id, self.id_to_token = load_vocab(vocab_file)
        self.reserved_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into tokens using regex pattern."""
        return [t for t in SMILES_REGEX.findall(smiles) if t]

    def encode(self, s: str, add_special: bool = True) -> List[int]:
        """Encode SMILES string to token IDs.

        Args:
            s: SMILES string
            add_special: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        toks = self._tokenize_smiles(s)
        ids = [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in toks]
        if add_special:
            return [self.token_to_id["<bos>"]] + ids + [self.token_to_id["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to SMILES string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            SMILES string
        """
        toks = [self.id_to_token.get(i, "?") for i in ids]
        toks = [t for t in toks if t not in self.reserved_tokens]
        return "".join(toks)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)


# --- Dataset from HDF5 ---
class H5SequenceDataset(Dataset):
    """Dataset for loading molecular sequences and features from HDF5 files.
    
    Supports multiple HDF5 files and handles:
    - SMILES/SELFIES tokenization
    - Feature extraction (dimensions, symmetry, rings, etc.)
    - Label binning for underrepresented classes
    """
    
    def __init__(
        self,
        mol_files: List[str],
        feat_files: List[str],
        tokenizer,
        label_encoder: LabelEncoder,
        underrepresented_data_file: str = "mol3d_data/underrepresented_data.json",
        mode: str = "smiles",
        max_len: int = 512,
        max_molecules: Optional[int] = None
    ):
        """Initialize dataset.
        
        Args:
            mol_files: List of HDF5 files with molecule sequences
            feat_files: List of HDF5 files with molecular features
            tokenizer: Tokenizer instance (SMILES or SELFIES)
            label_encoder: LabelEncoder for string→int conversion
            underrepresented_data_file: JSON with binning thresholds
            mode: "smiles" or "selfies"
            max_len: Maximum sequence length
            max_molecules: Maximum number of molecules to load (None = all)
        """
        self.mol_files = mol_files
        self.feat_files = feat_files
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_len = max_len
        self.mode = mode
        self.max_molecules = max_molecules

        # Precompute file offsets for efficient indexing across multiple files
        self.mol_file_offsets = [0]
        for mf in mol_files:
            with h5py.File(mf, "r") as f:
                n_mols = len(f[mode])
                self.mol_file_offsets.append(self.mol_file_offsets[-1] + n_mols)

        # Precompute index map: (file_id, index_in_file)
        self.entries = []
        total_added = 0
        for fid, mf in enumerate(mol_files):
            with h5py.File(mf, "r") as f:
                n_mols = len(f[mode])
                for idx in range(n_mols):
                    if self.max_molecules is not None and total_added >= self.max_molecules:
                        break
                    self.entries.append((fid, idx))
                    total_added += 1
            if self.max_molecules is not None and total_added >= self.max_molecules:
                break

        # Load ring info into memory for fast access
        self.load_ring_info()
        
        # Load binning thresholds
        with open(underrepresented_data_file, "r") as f:
            self.underrepresented_groups = json.load(f)

    def load_ring_info(self):
        """Load ring-related datasets into memory for fast access.

        Memory-efficient: stores data as numpy arrays with offset-based indexing
        instead of creating nested Python lists. Only loads molecules in self.entries.
        """
        # Find max molecule index per file to minimize data loading
        max_mol_idx_per_file = {}
        for file_id, mol_idx in self.entries:
            if file_id not in max_mol_idx_per_file:
                max_mol_idx_per_file[file_id] = mol_idx
            else:
                max_mol_idx_per_file[file_id] = max(max_mol_idx_per_file[file_id], mol_idx)

        all_ring_counts = []
        all_planar_errors = []
        all_plane_angles = []
        plane_angle_offsets = [0]

        for file_id, feat_file in enumerate(self.feat_files):
            if file_id not in max_mol_idx_per_file:
                continue

            with h5py.File(feat_file, "r") as f:
                # Load only needed molecules from this file
                max_mol_idx = max_mol_idx_per_file[file_id] + 1
                file_ring_counts = f["nrings"][:max_mol_idx]
                all_ring_counts.extend(file_ring_counts)
                all_planar_errors.extend(f["errors"][:max_mol_idx])

                # Calculate number of ring pair angles needed
                n_angles_needed = sum(n_rings * (n_rings - 1) // 2 for n_rings in file_ring_counts)

                # Load plane angles as numpy array
                plane_angles_arr = f["plane_angles"][:n_angles_needed]
                all_plane_angles.append(plane_angles_arr)

                # Build offset map for fast molecule→angles lookup
                offset = plane_angle_offsets[-1]
                for n_rings in file_ring_counts:
                    n_pairs = n_rings * (n_rings - 1) // 2
                    offset += n_pairs
                    plane_angle_offsets.append(offset)

        # Concatenate all arrays
        if all_plane_angles:
            self.ring_plane_angles_data = np.concatenate(all_plane_angles)
        else:
            self.ring_plane_angles_data = np.array([], dtype=[("i", "i4"), ("j", "i4"), ("val", "f4")])

        self.ring_counts = np.array(all_ring_counts)
        self.planar_fit_errors = np.array(all_planar_errors)
        self.ring_angle_offsets = plane_angle_offsets

    def __len__(self):
        return len(self.entries)

    def bin_label(self, label: int, threshold: int) -> str:
        """Bin integer labels with catch-all for values >= threshold.

        Args:
            label: Integer label value
            threshold: Threshold for catch-all bin

        Returns:
            String label (e.g., "0", "1", "5+")
        """
        return str(label) if label < threshold else f"{threshold}+"

    def map_underrepresented_label(self, label: str, underrep_labels: List[str], catch_all: str = "Other") -> str:
        """Map underrepresented labels to a catch-all category.

        Args:
            label: Original label
            underrep_labels: List of labels to map to catch-all
            catch_all: Catch-all category name

        Returns:
            Mapped label
        """
        return catch_all if label in underrep_labels else label

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a single data sample.

        Args:
            i: Sample index

        Returns:
            input_ids: Tokenized sequence (padded)
            attention_mask: Mask for valid tokens
            targets: Dictionary of target values for each property
        """
        file_id, mol_idx = self.entries[i]

        # Load and tokenize sequence
        with h5py.File(self.mol_files[file_id], "r") as f_mol:
            seq = f_mol[self.mode][mol_idx].decode("utf-8")

        # Tokenize with BOS/EOS and pad/truncate
        pad_id = self.tokenizer.token_to_id["<pad>"]
        bos_id = self.tokenizer.token_to_id["<bos>"]
        eos_id = self.tokenizer.token_to_id["<eos>"]

        tokens = [bos_id] + self.tokenizer.encode(seq, add_special=False)
        if len(tokens) >= self.max_len:
            tokens = tokens[:self.max_len - 1]
        tokens = tokens + [eos_id]

        if len(tokens) < self.max_len:
            tokens = tokens + [pad_id] * (self.max_len - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()

        # Load molecular features from HDF5
        with h5py.File(self.feat_files[file_id], "r") as f_feat:
            dimension = f_feat["dimensions"][mol_idx].decode("utf-8")
            point_group = self.map_underrepresented_label(
                f_feat["point_groups"][mol_idx].decode("utf-8"),
                self.underrepresented_groups["point_groups"]
            )
            n_symmetry_planes = self.bin_label(
                int(f_feat["symmetry_planes"][mol_idx]),
                self.underrepresented_groups["symmetry_planes"]
            )
            chirality = bool(f_feat["chiralities"][mol_idx])

        # Get preloaded ring info using dataset-wide index
        dataset_idx = self.mol_file_offsets[file_id] + mol_idx
        ring_count = self.bin_label(
            int(self.ring_counts[dataset_idx]),
            self.underrepresented_groups["nrings"]
        )
        planar_fit_error = float(self.planar_fit_errors[dataset_idx])

        # Get ring plane angles using offset map
        angle_start = self.ring_angle_offsets[dataset_idx]
        angle_end = self.ring_angle_offsets[dataset_idx + 1]
        ring_plane_angles = self.ring_plane_angles_data[angle_start:angle_end]

        # Build target dictionary
        targets = {
            "dimension": self.label_encoder.transform("dimension", dimension),
            "ring_count": self.label_encoder.transform("ring_count", ring_count),
            "chirality": int(chirality),
            "n_symmetry_planes": self.label_encoder.transform("n_symmetry_planes", n_symmetry_planes),
            "point_group": self.label_encoder.transform("point_group", point_group),
            "planar_fit_error": planar_fit_error,
            "ring_plane_angles": len(ring_plane_angles),  # Simplified: just count for now
        }

        return input_ids, attention_mask, targets


def collate_fn(batch):
    """Collate function for DataLoader batching.

    Args:
        batch: List of (input_ids, attention_mask, targets) tuples

    Returns:
        Batched input_ids, attention_masks, and targets dict
    """
    batch_input_ids = torch.stack([sample[0] for sample in batch])
    batch_attn_masks = torch.stack([sample[1] for sample in batch])

    # Stack targets for each property
    prop_names = batch[0][2].keys()
    batch_targets = {
        prop: torch.tensor([sample[2][prop] for sample in batch])
        for prop in prop_names
    }

    return batch_input_ids, batch_attn_masks, batch_targets


# --- Positional Encoding ---
class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for sequence modeling."""
    
    def __init__(self, d_model: int = 512, max_len: int = 512):
        """Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.d_model = d_model
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            x + positional embeddings
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pe(positions)
        return x + pos_emb


# --- Cross-Attention Module ---
class CrossAttention(nn.Module):
    """Cross-attention layer for conditioning on previous property predictions."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        """Initialize cross-attention.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross-attention.
        
        Args:
            x: Query tensor (current property)
            memory: Key/Value tensor (previous property embeddings)
            memory_mask: Padding mask for memory
            
        Returns:
            Updated x with cross-attention applied
        """
        attn_out, _ = self.attn(x, memory, memory, key_padding_mask=memory_mask)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x

# --- Property Head ---
class PropertyHead(nn.Module):
    """Prediction head for a single property with optional embedding for cross-attention.

    Architecture:
        1. Prediction head (classification or regression)
        2. Optional prediction embedding for cross-attention feedback
    """

    def __init__(
        self,
        d_model: int = 512,
        task: str = "classification",
        num_classes: Optional[int] = None,
        use_cross_attention: bool = False
    ):
        """Initialize property head.

        Args:
            d_model: Model dimension
            task: "classification" or "regression"
            num_classes: Number of classes (for classification)
            use_cross_attention: Whether to generate embeddings for cross-attention feedback
        """
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.use_cross_attention = use_cross_attention

        # Prediction head
        if task == "classification":
            assert num_classes is not None, "num_classes required for classification"
            self.head = nn.Linear(d_model, num_classes)
        elif task == "regression":
            self.head = nn.Linear(d_model, 1)
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        # Embed prediction for cross-attention (optional)
        if use_cross_attention:
            if task == "classification":
                self.pred_embedding = nn.Embedding(num_classes, d_model)
            else:
                self.pred_embedding = nn.Linear(1, d_model)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            logits: Prediction logits
            pred_embedding: Embedded prediction for cross-attention (None if use_cross_attention=False)
        """
        # Pool for prediction (use CLS token = first token)
        pooled = x[:, 0]  # (batch_size, d_model)
        logits = self.head(pooled)

        # Embed prediction for cross-attention (if enabled)
        pred_emb = None
        if self.use_cross_attention:
            if self.task == "classification":
                pred_idx = logits.argmax(dim=-1)  # (batch_size,)
                pred_emb = self.pred_embedding(pred_idx)  # (batch_size, d_model)
            else:
                pred_emb = self.pred_embedding(logits)  # (batch_size, d_model)

            # Expand to sequence length for cross-attention
            pred_emb = pred_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_len, d_model)

        return logits, pred_emb


# --- Hierarchical Transformer ---
class HierarchicalTransformer(nn.Module):
    """Flexible Transformer for molecular property prediction with attachment points.

    Architecture:
        1. Token + positional embeddings
        2. Stack of N encoder blocks (configurable)
        3. Prediction heads attached to specific encoder blocks:
           - Multiple heads can attach to the same block
           - Heads can attach to any block (0 to N-1)
           - Each head optionally embeds prediction for cross-attention feedback
        4. Optional cross-attention from prediction embeddings to subsequent blocks

    Example configurations:
        - Default: N=4 blocks, 1 head attached to block 3 (last)
        - Parallel: N=2 blocks, 7 heads attached to block 1 (all parallel)
        - Hierarchical: Head1@block0 (cross-attn) → Head2@block1 (cross-attn) → Head3@block2
    """

    def __init__(
        self,
        vocab_size: int,
        property_configs: List[Dict],
        n_encoder_blocks: int = 4,
        max_len: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        """Initialize flexible transformer.

        Args:
            vocab_size: Size of token vocabulary
            property_configs: List of dicts with property configurations:
                [
                    {
                        "name": "dimension",
                        "task": "classification",
                        "num_classes": 5,
                        "attach_at_block": 3,  # Attach head to block 3 output
                        "use_cross_attention": True  # Embed prediction and feed back
                    },
                    {
                        "name": "ring_count",
                        "task": "classification",
                        "num_classes": 7,
                        "attach_at_block": 3,  # Attach to same block as dimension
                        "use_cross_attention": False
                    },
                    ...
                ]
            n_encoder_blocks: Total number of encoder blocks in stack
            max_len: Maximum sequence length
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.n_encoder_blocks = n_encoder_blocks

        # Validate n_encoder_blocks
        if n_encoder_blocks < 1:
            raise ValueError(f"n_encoder_blocks must be >= 1, got {n_encoder_blocks}")

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len)

        # Create stack of encoder blocks (individual layers for flexible attachment)
        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            for _ in range(n_encoder_blocks)
        ])

        # Create cross-attention modules for each encoder block (except first)
        self.cross_attn_modules = nn.ModuleList([
            CrossAttention(d_model, nhead, dropout) if i > 0 else None
            for i in range(n_encoder_blocks)
        ])

        # Create property heads
        self.property_heads = nn.ModuleDict()
        self.property_configs = {}

        for cfg in property_configs:
            prop_name = cfg["name"]
            self.property_configs[prop_name] = cfg

            self.property_heads[prop_name] = PropertyHead(
                d_model=d_model,
                task=cfg["task"],
                num_classes=cfg.get("num_classes"),
                use_cross_attention=cfg.get("use_cross_attention", False)
            )

        # Group properties by attachment block for efficient processing
        self.attachment_map = {}  # block_idx -> [property_names]
        for cfg in property_configs:
            attach_block = cfg.get("attach_at_block", n_encoder_blocks - 1)

            # Validate attachment block is within valid range
            if attach_block < 0 or attach_block >= n_encoder_blocks:
                raise ValueError(
                    f"Property '{cfg['name']}' has attach_at_block={attach_block}, "
                    f"but must be in range [0, {n_encoder_blocks - 1}]"
                )

            if attach_block not in self.attachment_map:
                self.attachment_map[attach_block] = []
            self.attachment_map[attach_block].append(cfg["name"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder stack with head attachments.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len) where 1=valid, 0=padding

        Returns:
            Dictionary of predictions for each property (keyed by property name)
        """
        # Embed tokens and add positional encoding
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)

        # Track outputs and cross-attention memories
        outputs = {}
        cross_attn_memories = []  # List of (pred_embedding, mask) tuples for cross-attention

        # Process encoder blocks sequentially
        for block_idx, encoder_block in enumerate(self.encoder_blocks):

            # Apply cross-attention from prediction embeddings generated so far
            if block_idx > 0 and len(cross_attn_memories) > 0 and self.cross_attn_modules[block_idx] is not None:
                # Concatenate all prediction embeddings along sequence dimension
                # Each pred_emb has shape (batch, seq_len, d_model)
                all_memories = torch.cat([mem for mem, _ in cross_attn_memories], dim=1)
                # All masks have shape (batch, seq_len) - concatenate to (batch, n_preds * seq_len)
                all_masks = torch.cat([mask for _, mask in cross_attn_memories], dim=1)

                # Cross-attend from current representation to all previous predictions
                x = self.cross_attn_modules[block_idx](
                    x,
                    all_memories,
                    memory_mask=~all_masks.bool()  # CrossAttention expects True for positions to mask out
                )

            # Process through encoder block (self-attention + FFN)
            x = encoder_block(x, src_key_padding_mask=~attention_mask.bool())

            # Apply any prediction heads attached at this block
            if block_idx in self.attachment_map:
                for prop_name in self.attachment_map[block_idx]:
                    # Generate prediction for this property
                    logits, pred_emb = self.property_heads[prop_name](x)
                    outputs[prop_name] = logits

                    # Store prediction embedding for cross-attention to future blocks (if enabled)
                    if pred_emb is not None:
                        cross_attn_memories.append((pred_emb, attention_mask))

        return outputs


# --- Training & Evaluation ---
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    property_configs: List[Dict],
    bf16: bool = True
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        property_configs: List of property configurations with 'name' and 'task' keys
        bf16: Whether to use bfloat16 autocast

    Returns:
        Average training loss
    """
    model.train()
    epoch_loss = 0.0

    # Build loss functions for each property
    loss_functions = {}
    for prop_cfg in property_configs:
        if prop_cfg["task"] == "classification":
            loss_functions[prop_cfg["name"]] = nn.CrossEntropyLoss()
        else:
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
                    batch_loss += loss_functions[prop_name](prop_pred, prop_target.long())
                else:
                    batch_loss += loss_functions[prop_name](prop_pred.squeeze(-1), prop_target.float())

        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item() * batch_inputs.size(0)

    return epoch_loss / len(dataloader.dataset)

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    property_configs: List[Dict],
    bf16: bool = False
) -> Tuple[float, float]:
    """Evaluate model.

    Args:
        model: Model to evaluate
        dataloader: Validation/test data loader
        device: Device to evaluate on
        property_configs: List of property configurations with 'name' and 'task' keys
        bf16: Whether to use bfloat16 autocast

    Returns:
        Tuple of (MSE, RMSE)
    """
    model.eval()
    eval_loss = 0.0

    # Build loss functions for each property
    loss_functions = {}
    for prop_cfg in property_configs:
        if prop_cfg["task"] == "classification":
            loss_functions[prop_cfg["name"]] = nn.CrossEntropyLoss()
        else:
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

                # Accumulate loss for each property
                for prop_cfg in property_configs:
                    prop_name = prop_cfg["name"]
                    if prop_name not in batch_outputs:
                        continue

                    prop_target = batch_targets[prop_name]
                    prop_pred = batch_outputs[prop_name]

                    if prop_cfg["task"] == "classification":
                        batch_loss += loss_functions[prop_name](prop_pred, prop_target.long())
                    else:
                        batch_loss += loss_functions[prop_name](prop_pred.squeeze(-1), prop_target.float())

                eval_loss += batch_loss.item() * batch_inputs.size(0)

    mse = eval_loss / len(dataloader.dataset)
    rmse = mse ** 0.5
    return mse, rmse


# --- Main Training Script ---
def main():
    """Main training loop with flexible attachment point architecture."""
    parser = argparse.ArgumentParser(description="Train Flexible Transformer for molecular property prediction")
    parser.add_argument("--mol_files", nargs='+', required=True, help="HDF5 files with molecules")
    parser.add_argument("--feat_files", nargs='+', required=True, help="HDF5 files with features")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
            "task": "regression"
        },
        "ring_plane_angles": {
            "task": "regression"
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
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_rmse = float("inf")

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, property_configs, bf16=True)
        val_mse, val_rmse = evaluate(model, val_loader, device, property_configs, bf16=True)

        print(f"Epoch {epoch:3d} | Train loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "model_state_dict": model.state_dict(),
                "property_configs": property_configs,
                "vocab_size": tokenizer.vocab_size,
                "max_len": args.max_len,
                "d_model": args.d_model,
                "n_encoder_blocks": args.n_encoder_blocks,
                "label_encoder": label_encoder,
            }, "best_model.pt")
            print(f"  -> Saved best model (val RMSE: {val_rmse:.4f})")

    # Final test evaluation
    test_mse, test_rmse = evaluate(model, test_loader, device, property_configs, bf16=True)
    print(f"\nTest RMSE: {test_rmse:.4f} (MSE {test_mse:.6f})")
    print(f"Best validation RMSE: {best_val_rmse:.4f}")
    print("Training complete!")


if __name__ == "__main__":
    main()
