#!/usr/bin/env python
# coding: utf-8

"""
Dataset and data loading utilities for molecular sequences.

This module handles loading molecular sequences and features from HDF5 files,
with support for multiple files, RAM preloading, and variable-length properties.
"""

import json
from typing import List, Dict, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizers import LabelEncoder


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

        # Load binning thresholds
        with open(underrepresented_data_file, "r") as f:
            self.underrepresented_groups = json.load(f)

        # Preload all data into RAM
        print("Preloading all molecular data into RAM...")
        self.load_all_data_into_ram()
        print(f"Preloaded {len(self)} molecules into RAM")

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

    def load_all_data_into_ram(self):
        """Preload all molecular sequences and features into RAM for fast access.

        This eliminates HDF5 I/O bottleneck during training by loading all data upfront.
        Memory usage: ~3.9M molecules * (avg 200 bytes/sequence + 50 bytes/features) ~= 1GB
        """
        n_entries = len(self.entries)

        # Pre-allocate arrays/lists for all data
        self.sequences = [None] * n_entries
        self.dimensions = [None] * n_entries
        self.point_groups = [None] * n_entries
        self.n_symmetry_planes_raw = [0] * n_entries
        self.chiralities = [False] * n_entries
        self.ring_counts_raw = [0] * n_entries
        self.planar_fit_errors_raw = [0.0] * n_entries
        self.ring_plane_angles_raw = [None] * n_entries

        # Load sequences file by file
        for file_id, mol_file in enumerate(self.mol_files):
            # Get all indices for this file
            file_indices = [(i, mol_idx) for i, (fid, mol_idx) in enumerate(self.entries) if fid == file_id]
            if not file_indices:
                continue

            with h5py.File(mol_file, "r") as f:
                for dataset_idx, mol_idx in file_indices:
                    self.sequences[dataset_idx] = f[self.mode][mol_idx].decode("utf-8")

        # Load features file by file
        for file_id, feat_file in enumerate(self.feat_files):
            file_indices = [(i, mol_idx) for i, (fid, mol_idx) in enumerate(self.entries) if fid == file_id]
            if not file_indices:
                continue

            with h5py.File(feat_file, "r") as f:
                # Load all feature datasets for this file into memory first
                dims = f["dimensions"][:]
                pgs = f["point_groups"][:]
                sym_planes = f["symmetry_planes"][:]
                chirs = f["chiralities"][:]
                nrings = f["nrings"][:]
                errors = f["errors"][:]

                # Build cumulative offset map for ring-indexed data (errors and plane angles)
                # Both errors and plane_angles arrays are indexed by ring offset, not molecule index
                ring_offsets = [0]  # Offset for errors (one per ring)
                plane_angle_offsets = [0]  # Offset for plane angles (one per ring pair)
                for mol_idx in range(len(nrings)):
                    n_rings = int(nrings[mol_idx])
                    ring_offsets.append(ring_offsets[-1] + n_rings)  # Cumulative ring count
                    n_pairs = n_rings * (n_rings - 1) // 2
                    plane_angle_offsets.append(plane_angle_offsets[-1] + n_pairs)

                plane_angles = f["plane_angles"][:]

                # Now populate data for molecules in this file
                for dataset_idx, mol_idx in file_indices:
                    self.dimensions[dataset_idx] = dims[mol_idx].decode("utf-8")
                    self.point_groups[dataset_idx] = pgs[mol_idx].decode("utf-8")
                    self.n_symmetry_planes_raw[dataset_idx] = int(sym_planes[mol_idx])
                    self.chiralities[dataset_idx] = bool(chirs[mol_idx])

                    ring_count = int(nrings[mol_idx])
                    self.ring_counts_raw[dataset_idx] = ring_count

                    # Get planar fit errors using ring offset (one error per ring)
                    error_start = ring_offsets[mol_idx]
                    error_end = ring_offsets[mol_idx + 1]
                    if error_end > error_start:
                        self.planar_fit_errors_raw[dataset_idx] = errors[error_start:error_end]
                    else:
                        self.planar_fit_errors_raw[dataset_idx] = np.array([], dtype=np.float32)

                    # Get ring plane angles using the offset map
                    angle_start = plane_angle_offsets[mol_idx]
                    angle_end = plane_angle_offsets[mol_idx + 1]
                    if angle_end > angle_start:
                        self.ring_plane_angles_raw[dataset_idx] = plane_angles[angle_start:angle_end]
                    else:
                        self.ring_plane_angles_raw[dataset_idx] = np.array([])

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
        """Get a single data sample from preloaded RAM data.

        Args:
            i: Sample index

        Returns:
            input_ids: Tokenized sequence (padded)
            attention_mask: Mask for valid tokens
            targets: Dictionary of target values for each property
        """
        # Get preloaded sequence (no HDF5 I/O!)
        seq = self.sequences[i]

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

        # Get preloaded molecular features (no HDF5 I/O!)
        dimension = self.dimensions[i]
        point_group = self.map_underrepresented_label(
            self.point_groups[i],
            self.underrepresented_groups["point_groups"]
        )
        n_symmetry_planes = self.bin_label(
            self.n_symmetry_planes_raw[i],
            self.underrepresented_groups["symmetry_planes"]
        )
        chirality = self.chiralities[i]

        ring_count = self.bin_label(
            self.ring_counts_raw[i],
            self.underrepresented_groups["nrings"]
        )
        planar_fit_error = self.planar_fit_errors_raw[i]
        ring_plane_angles_data = self.ring_plane_angles_raw[i]

        # Extract actual angle values from structured array
        # ring_plane_angles_data is structured array with dtype=[('i', 'i4'), ('j', 'i4'), ('val', 'f4')]
        if len(ring_plane_angles_data) > 0:
            ring_plane_angles = ring_plane_angles_data['val']  # Extract angle values
        else:
            ring_plane_angles = np.array([], dtype=np.float32)

        # Build target dictionary
        targets = {
            "dimension": self.label_encoder.transform("dimension", dimension),
            "ring_count": self.label_encoder.transform("ring_count", ring_count),
            "chirality": int(chirality),
            "n_symmetry_planes": self.label_encoder.transform("n_symmetry_planes", n_symmetry_planes),
            "point_group": self.label_encoder.transform("point_group", point_group),
            "planar_fit_error": planar_fit_error,
            "ring_plane_angles": ring_plane_angles,  # Now returns actual angle values (variable length)
        }

        return input_ids, attention_mask, targets


def collate_fn(batch):
    """Collate function for DataLoader batching.

    Handles variable-length sequences (ring_plane_angles) by padding to max length in batch.

    Args:
        batch: List of (input_ids, attention_mask, targets) tuples

    Returns:
        Batched input_ids, attention_masks, and targets dict (with masks for variable-length properties)
    """
    batch_input_ids = torch.stack([sample[0] for sample in batch])
    batch_attn_masks = torch.stack([sample[1] for sample in batch])

    # Collate targets for each property
    prop_names = batch[0][2].keys()
    batch_targets = {}

    for prop in prop_names:
        values = [sample[2][prop] for sample in batch]

        # Handle variable-length sequences (ring_plane_angles, planar_fit_error)
        if prop in ["ring_plane_angles", "planar_fit_error"]:
            # Find max length in this batch
            max_len = max(len(v) for v in values)

            # Pad sequences to max_len (or 1 if all empty to avoid 0-dim tensors)
            # This ensures tensor shape is always (batch_size, max_len)
            effective_max_len = max(max_len, 1)

            padded_values = []
            masks = []
            for v in values:
                n_values = len(v)
                if n_values > 0:
                    # Pad with zeros
                    padded = np.pad(v, (0, effective_max_len - n_values), mode='constant', constant_values=0.0)
                    mask = np.concatenate([np.ones(n_values), np.zeros(effective_max_len - n_values)])
                else:
                    # No rings for this molecule
                    padded = np.zeros(effective_max_len)
                    mask = np.zeros(effective_max_len)

                padded_values.append(padded)
                masks.append(mask)

            batch_targets[prop] = torch.tensor(np.array(padded_values), dtype=torch.float32)
            batch_targets[prop + "_mask"] = torch.tensor(np.array(masks), dtype=torch.float32)
        else:
            # Scalar targets (classification indices or regression values)
            batch_targets[prop] = torch.tensor(values)

    return batch_input_ids, batch_attn_masks, batch_targets
