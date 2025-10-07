# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HIDRA (HIerarchical DFT accuracy transformer model to Reconstruct Atomic geometry) is a transformer-based machine learning model that predicts molecular 3D geometry and properties from SMILES or SELFIES string representations.

The model uses a hierarchical architecture where molecular properties are predicted sequentially, with each property prediction cross-attending to previous predictions:
- **Flow**: dimension → rings → chirality → symmetry → point_group → planarity → angles
- **Input**: SMILES or SELFIES molecular representations
- **Output**: Multiple molecular properties (dimensionality, ring count, chirality, symmetry planes, point groups, planarity metrics)

## Data Architecture

### Dataset Structure
- **Location**: HDF5 files in `mol3d_data/` directory (4 molecule files + 4 feature files)
- **Size**: ~3.9 million molecular structures (~1M molecules per file pair)
- **Format**: Each molecule has paired entries in:
  - Molecule files (`mol3d_mil*.h5`): SMILES and SELFIES strings
  - Feature files (`mol3d_feat_mil*.h5`): coordinates, symmetry, chirality, point groups, rings

### Key Data Files
- `smiles_vocab.json` / `selfies_vocab.json`: Token vocabularies with special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`)
- `underrepresented_data.json`: Binning thresholds for underrepresented classes (symmetry planes, rings, point groups)

### Data Loading
- `H5SequenceDataset` handles multi-file loading with precomputed offsets for efficient indexing
- Ring information (counts, plane angles, planarity errors) is preloaded into memory for performance
- Labels are binned using thresholds from `underrepresented_data.json` (e.g., "6+" for 6 or more rings)

## Model Architecture

### Core Components (transformer.py)
1. **Tokenizers**: `SmilesTokenizer` (regex-based) and `SelfiesTokenizer` (uses selfies library)
2. **HierarchicalTransformer**: Main model with:
   - Shared initial encoder blocks (configurable via `n_initial_blocks`)
   - Property-specific blocks with cross-attention to previous predictions
   - Each `PropertyBlock` has encoder layers + prediction head + embedding layer
3. **Label Encoding**: `LabelEncoder` converts string labels to indices for classification

### Training Configuration
- Default: 4 initial shared encoder blocks, 2 blocks per property
- Properties predicted in order: dimension, ring_count, chirality, n_symmetry_planes, point_group, planar_fit_error, ring_plane_angles
- Mixed task types: classification (dimension, rings, chirality, symmetry, point group) + regression (planarity)
- BF16 autocast for training efficiency
- 80/10/10 train/val/test split

## Common Commands

### Training the Model
```bash
python transformer.py \
  --mol_files mol3d_data/mol3d_mil1.h5 mol3d_data/mol3d_mil2.h5 mol3d_data/mol3d_mil3.h5 mol3d_data/mol3d_mil4.h5 \
  --feat_files mol3d_data/mol3d_feat_mil1.h5 mol3d_data/mol3d_feat_mil2.h5 mol3d_data/mol3d_feat_mil3.h5 mol3d_data/mol3d_feat_mil4.h5 \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4 \
  --mode smiles \
  --d_model 512 \
  --n_initial_blocks 4
```

### Key Arguments
- `--mode`: Choose between "smiles" or "selfies" input format
- `--max_len`: Maximum sequence length (default: 512)
- `--d_model`: Model dimension (default: 512)
- `--n_initial_blocks`: Number of shared encoder blocks before property predictions (default: 4)
- `--device`: Specify "cuda" or "cpu"

### Model Output
- Saves best model to `best_model.pt` based on validation RMSE
- Checkpoint includes: model weights, property configs, vocab size, label encoder

## Development Notes

### File Paths
- The code expects vocabularies at `mol3d_data/{mode}_vocab.json`
- Underrepresented data config at `mol3d_data/underrepresented_data.json`
- HDF5 files follow naming: `mol3d_mil*.h5` and `mol3d_feat_mil*.h5`

### Property Order (Critical)
The target keys in dataset must match the property block order:
```python
["dimension", "ring_count", "chirality", "n_symmetry_planes",
 "point_group", "planar_fit_error", "ring_plane_angles"]
```

### Reproducibility
- Random seed set to 42 for PyTorch and Python random module
- Deterministic behavior requires setting CUDNN flags separately if needed

## Coding Mistakes to Avoid

*(This section should be filled in as mistakes are discovered during development)*

### General Patterns to Follow

- Always use `self.` prefix for instance variables in test setUp methods
- Match num_classes in property_configs to actual label encoder output (check with `get_num_classes()`)
- Set models to `.eval()` mode before comparing deterministic outputs
- When testing gradients, check for "at least some" rather than "all" to account for unused parameters

### Known Limitations (Not Bugs)

1. **Ring Plane Angles Data Loss** (line 398): Currently only storing count of ring plane angles, not the actual angle values
   - Comment: "Simplified: just count for now"
   - This means the model predicts number of angle pairs but doesn't learn from actual angle values
   - To fix: Would need to modify dataset __getitem__ to return variable-length angle lists and update collate_fn accordingly

## Documentation Requirements

**All generated code must include:**
1. Function/class docstrings with:
   - Purpose description
   - Args with types
   - Returns with types
   - Example usage if complex
2. Inline comments for non-obvious logic
3. Type hints for function signatures

**If code is provided without documentation, Claude should add comprehensive documentation before proceeding.**

## Change Tracking

**All code changes must be logged:**
- Maintain a separate `changes_log.md` file to track all modifications
- Do NOT add change logs or modification history to CLAUDE.md
- Each entry in `changes_log.md` should include:
  - Date and time of change
  - Files modified with specific line numbers
  - Description of what was changed and why
  - Related issue or request reference (if applicable)

## Edit Approval Protocol

**Before making any edits:**
1. Claude must list all proposed changes with file paths and line numbers
2. Clearly describe what will be modified and why
3. Wait for user approval (+ or y means approved)
4. **Only edit the necessary parts** - do not refactor or modify unrelated code
5. Preserve existing code style and formatting

**Example approval request:**
```
Proposed edits:
1. transformer.py:145-150 - Fix SMILES regex to handle aromatic atoms
2. transformer.py:823 - Update loss function to handle class imbalance

Proceed with these changes? (+ or y to approve)
```
