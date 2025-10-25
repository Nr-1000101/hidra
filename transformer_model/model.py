#!/usr/bin/env python
# coding: utf-8

"""
Main transformer model architecture for molecular property prediction.

This module contains the HierarchicalTransformer model with flexible attachment points.
"""

from typing import List, Dict, Optional

import torch
import torch.nn as nn

from model_components import (
    LearnedPositionalEncoding,
    CrossAttention,
    PropertyHead,
    SequenceRegressionHead
)

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

            # Use SequenceRegressionHead for variable-length sequence tasks
            if cfg["task"] == "sequence_regression":
                self.property_heads[prop_name] = SequenceRegressionHead(
                    d_model=d_model,
                    max_seq_len=cfg.get("max_seq_len", 105),
                    use_cross_attention=cfg.get("use_cross_attention", False)
                )
            else:
                # Standard PropertyHead for classification and scalar regression
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


