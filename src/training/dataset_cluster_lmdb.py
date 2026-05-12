"""
LMDB Dataset classes for cluster prof_v2 FFT data.

Fast, memory-efficient dataset using LMDB for storage.
Supports train/val/test splits with separate LMDB databases.
"""
import os
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle



class ClusterLMDBDataset(Dataset):
    """
    LMDB Dataset for cluster prof_v2 FFT data.

    Uses fft_lr (low-res from truncated acquisition) and fft_hr (high-res from full acquisition).

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
            self.n_points_fid = metadata.get('n_points_fid', 2048)
            self.lr_points = metadata.get('lr_points', 1024)
            self.fft_size = metadata.get('fft_size', 4096)
            self.fft_output_size = metadata.get('fft_output_size', 2048)
            self.acquisition_reduction_factor = metadata.get('acquisition_reduction_factor', 2)

        print(f"  ClusterLMDBDataset ({split}): {self.N} samples")
        print(f"    FID points (full): {self.n_points_fid}")
        print(f"    FID points (truncated): {self.lr_points}")
        print(f"    FFT output size: {self.fft_output_size}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f'sample_{idx:08d}'.encode('ascii')
            data = pickle.loads(txn.get(key))

            x = data['fft_lr'].astype(np.float32)  # Low-res from truncated acquisition
            y = data['fft_hr'].astype(np.float32)  # High-res from full acquisition
            y_raw = data.get('fft_hr_raw', y)  # Raw (un-normalized) HR for amplitude ratios

        # Optional normalization
        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + self.eps)
            y = (y - y.min()) / (y.max() - y.min() + self.eps)

        # Add feature dimension (T, 1)
        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        y_raw = np.expand_dims(y_raw.astype(np.float32), axis=-1)

        # Apply transform if provided
        if self.transform is not None:
            x, y = self.transform(x, y)

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_raw)

    def __getitem_with_metadata__(self, idx):
        """Get item with full metadata including peaks info."""
        with self.env.begin() as txn:
            key = f'sample_{idx:08d}'.encode('ascii')
            data = pickle.loads(txn.get(key))

            x = data['fft_lr'].astype(np.float32)
            y = data['fft_hr'].astype(np.float32)
            y_raw = data.get('fft_hr_raw', y)

        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + self.eps)
            y = (y - y.min()) / (y.max() - y.min() + self.eps)

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        y_raw = np.expand_dims(y_raw.astype(np.float32), axis=-1)

        if self.transform is not None:
            x, y = self.transform(x, y)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(y_raw),
            {
                'compound_formula': data.get('compound_formula', 'unknown'),
                'cluster_mass': data.get('cluster_mass', 0),
                'n_peaks': data.get('n_peaks', 0),
                'sample_id': data.get('sample_id', idx),
                'peaks': data.get('peaks', []),  # list of (mass, prob) tuples
            }
        )

    def get_metadata(self, idx):
        """Get metadata for a sample."""
        with self.env.begin() as txn:
            key = f'sample_{idx:08d}'.encode('ascii')
            data = pickle.loads(txn.get(key))

        return {
            'compound_formula': data.get('compound_formula', 'unknown'),
            'cluster_mass': data.get('cluster_mass', 0),
            'n_peaks': data.get('n_peaks', 0),
            'sample_id': data.get('sample_id', idx),
            'peaks': data.get('peaks', []),  # list of (mass, prob) tuples
        }

    def __del__(self):
        """Close LMDB environment."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class ClusterLMDBDataModule:
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
        **dataset_kwargs: Additional arguments for ClusterLMDBDataset
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
        self.train_dataset = ClusterLMDBDataset(
            lmdb_path, split='train', normalize=normalize, **dataset_kwargs
        )
        self.val_dataset = ClusterLMDBDataset(
            lmdb_path, split='val', normalize=normalize, **dataset_kwargs
        )
        self.test_dataset = ClusterLMDBDataset(
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
