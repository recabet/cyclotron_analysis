import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5SpectraDataset(Dataset):
    """
    Streaming dataset for spectra stored in HDF5.
    Applies per-sample min-max normalization (optional).
    NOTE: Use num_workers=0 in DataLoader to avoid h5py multiprocessing issues.
    """

    def __init__(self,
                 h5_path: str,
                 x_key: str,
                 y_key: str,
                 indices: np.ndarray = None,
                 normalize: bool = True,
                 eps: float = 1e-12):

        super().__init__()
        self.h5_path = h5_path
        self.x_key = x_key
        self.y_key = y_key
        self.normalize = normalize
        self.eps = eps
        self._file = None

        # Determine total number of samples and prepare indices
        with h5py.File(self.h5_path, 'r') as f:
            N = len(f[self.x_key])
        all_idx = np.arange(N)
        self.indices = all_idx if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        i = int(self.indices[idx])
        x = self._file[self.x_key][i]  # (L,)
        y = self._file[self.y_key][i]  # (L,)

        if self.normalize:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x = (x - x_min) / (x_max - x_min + self.eps)
            y = (y - y_min) / (y_max - y_min + self.eps)

        # Add feature dimension
        x = np.expand_dims(x.astype(np.float32), axis=-1)  # (L, 1)
        y = np.expand_dims(y.astype(np.float32), axis=-1)  # (L, 1)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass