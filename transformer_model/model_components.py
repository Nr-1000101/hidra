#!/usr/bin/env python
# coding: utf-8

"""
Reusable transformer building blocks for molecular property prediction.

This module provides the core components used to build the hierarchical transformer:
- Positional encoding
- Cross-attention modules
- Property prediction heads (classification, regression, sequence regression)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


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


class SequenceRegressionHead(nn.Module):
    """Prediction head for variable-length sequence regression (e.g., ring plane angles).

    Outputs a fixed-size tensor where only the first N values are used per sample,
    determined by a mask. Unused outputs are ignored during loss computation.

    Example:
        - Molecule with 3 rings → 3 angles → use outputs[0:3], mask rest
        - Molecule with 15 rings → 105 angles → use all outputs[0:105]
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 105,
        use_cross_attention: bool = False
    ):
        """Initialize sequence regression head.

        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length (e.g., 105 for 15 rings → 15*14/2 angles)
            use_cross_attention: Whether to generate embeddings for cross-attention feedback
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_cross_attention = use_cross_attention
        self.task = "sequence_regression"

        # Prediction head: d_model → max_seq_len outputs
        self.head = nn.Linear(d_model, max_seq_len)

        # Embed prediction for cross-attention (optional)
        # Average the predicted sequence into a single embedding
        if use_cross_attention:
            self.pred_embedding = nn.Linear(max_seq_len, d_model)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            predictions: Predicted sequence (batch_size, max_seq_len)
            pred_embedding: Embedded prediction for cross-attention (None if use_cross_attention=False)
        """
        # Pool for prediction (use CLS token = first token)
        pooled = x[:, 0]  # (batch_size, d_model)
        predictions = self.head(pooled)  # (batch_size, max_seq_len)

        # Embed prediction for cross-attention (if enabled)
        pred_emb = None
        if self.use_cross_attention:
            pred_emb = self.pred_embedding(predictions)  # (batch_size, d_model)
            # Expand to sequence length for cross-attention
            pred_emb = pred_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_len, d_model)

        return predictions, pred_emb
