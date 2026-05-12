"""
LMDB Dataset classes for narrowband FFT data.

Fast, memory-efficient dataset using LMDB for storage.
Supports train/val/test splits with separate LMDB databases.
"""
import os
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from typing import Optional


class LMDBDataset(Dataset):
    """
    LMDB Dataset for narrowband FFT data.

    Advantages over HDF5:
    - Fast random access
    - Memory-mapped for efficient reading
    - No file handle issues with multiprocessing
    - Better compression for this use case

    Args:
        lmdb_path: Path to LMDB database directory
        split: Which split to use ('train', 'val', 'test', 'full')
        normalize: Normalize spectra to [0, 1]
        transform: Optional transform function
    """

    def __init__(
        self,
        lmdb_path: str,
        split: str = 'train',
        normalize: bool = False,
        eps: float = 1e-12,
        transform=None,
    ):
        super().__init__()

        self.lmdb_path = lmdb_path
        self.split = split
        self.normalize = normalize
        self.eps = eps
        self.transform = transform

        # Open LMDB environment
        self.env = lmdb.open(
            os.path.join(lmdb_path, f'{split}.lmdb'),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        # Get dataset length
        with self.env.begin() as txn:
            len_data = pickle.loads(txn.get(b'__len__'))
            self.N = len_data['n_samples']

            # Load metadata
            metadata = pickle.loads(txn.get(b'__metadata__'))
            self.sampling_rate = metadata.get('sampling_rate', 1e6)
            self.baseband_sampling_rate = metadata.get('baseband_sampling_rate', 1e6)
            self.fft_size = metadata.get('fft_size', 1024)
            self.fid_length = metadata.get('fid_length', 1024)

        print(f"  LMDBDataset ({split}): {self.N} samples")
        print(f"    FFT size: {self.fft_size}")
        print(f"    Baseband sampling rate: {self.baseband_sampling_rate:.0f} Hz")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f'sample_{idx:08d}'.encode('ascii')
            data = pickle.loads(txn.get(key))

            x = data['fft_low'].astype(np.float32)
            y = data['fft_hr'].astype(np.float32)

        # Optional normalization
        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + self.eps)
            y = (y - y.min()) / (y.max() - y.min() + self.eps)

        # Add feature dimension (T, 1)
        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)

        # Apply transform if provided
        if self.transform is not None:
            x, y = self.transform(x, y)

        return torch.from_numpy(x), torch.from_numpy(y)

    def get_metadata(self, idx):
        """Get metadata for a sample."""
        with self.env.begin() as txn:
            key = f'sample_{idx:08d}'.encode('ascii')
            data = pickle.loads(txn.get(key))

        return {
            'formula': data['formula'],
            'cluster_id': data['cluster_id'],
            'center_freq': data['center_freq'],
            'n_points': data['n_points'],
            'fft_size': data['fft_size']
        }

    def __del__(self):
        """Close LMDB environment."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class LMDBDataModule:
    """
    Data module for handling train/val/test splits from LMDB.

    Convenience class that creates datasets and dataloaders
    for training, validation, and testing.

    Args:
        lmdb_path: Path to LMDB database directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        pin_memory: Whether to pin memory for faster GPU transfer
        normalize: Normalize spectra to [0, 1]
        **dataset_kwargs: Additional arguments for LMDBDataset
    """

    def __init__(
        self,
        lmdb_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = False,
        **dataset_kwargs
    ):
        self.lmdb_path = lmdb_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.dataset_kwargs = dataset_kwargs

        # Create datasets
        self.train_dataset = LMDBDataset(
            lmdb_path, split='train', normalize=normalize, **dataset_kwargs
        )
        self.val_dataset = LMDBDataset(
            lmdb_path, split='val', normalize=normalize, **dataset_kwargs
        )
        self.test_dataset = LMDBDataset(
            lmdb_path, split='test', normalize=normalize, **dataset_kwargs
        )

    def train_dataloader(self):
        """Get training dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def test_dataloader(self):
        """Get test dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )
