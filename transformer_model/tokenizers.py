#!/usr/bin/env python
# coding: utf-8

"""
Tokenizers and vocabulary utilities for molecular sequence encoding.

This module provides tokenization for SMILES and SELFIES molecular representations,
along with label encoding for classification tasks.
"""

import json
import re
from typing import List, Dict, Tuple

import selfies as sf


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
