"""
HIDRA Transformer Model Package

Modular transformer architecture for molecular property prediction from SMILES/SELFIES.
"""

from .tokenizers import (
    load_vocab,
    LabelEncoder,
    SelfiesTokenizer,
    SmilesTokenizer,
    SMILES_REGEX
)

from .dataset import (
    H5SequenceDataset,
    collate_fn
)

from .model_components import (
    LearnedPositionalEncoding,
    CrossAttention,
    PropertyHead,
    SequenceRegressionHead
)

from .model import HierarchicalTransformer

from .metrics import (
    classification_accuracy,
    compute_mae,
    circular_mse_loss,
    circular_mae_loss,
    FocalLoss,
    compute_classification_metrics,
    compute_class_weights
)

from .early_stopping import EarlyStoppingMonitor

from .training import (
    train_one_epoch,
    evaluate
)

__all__ = [
    # Tokenizers
    'load_vocab',
    'LabelEncoder',
    'SelfiesTokenizer',
    'SmilesTokenizer',
    'SMILES_REGEX',
    # Dataset
    'H5SequenceDataset',
    'collate_fn',
    # Model Components
    'LearnedPositionalEncoding',
    'CrossAttention',
    'PropertyHead',
    'SequenceRegressionHead',
    # Model
    'HierarchicalTransformer',
    # Metrics
    'classification_accuracy',
    'compute_mae',
    'circular_mse_loss',
    'circular_mae_loss',
    'FocalLoss',
    'compute_classification_metrics',
    'compute_class_weights',
    # Early Stopping
    'EarlyStoppingMonitor',
    # Training
    'train_one_epoch',
    'evaluate',
]
