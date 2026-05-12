import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5SpectraDataset(Dataset):
    """
    HDF5 Spectra Dataset

    Modes:
    - interval_size=None  -> returns full spectrum (L=8192)
    - interval_size=int   -> returns peak-centered window of that size

    Performance:
    - preload=True (default): loads all data into RAM once at init, eliminating
      per-sample H5 I/O during training. Requires enough RAM to hold the dataset.
    - preload=False: streams from H5 on each __getitem__. Use num_workers=0
      or implement worker_init_fn to avoid h5py multiprocessing issues.
    """

    def __init__(self,
                 h5_path: str,
                 x_key: str,
                 y_key: str,
                 interval_size: int = None,
                 centered: bool = False,
                 indices: np.ndarray = None,
                 normalize: bool = False,
                 eps: float = 1e-12,
                 preload: bool = True):

        super().__init__()

        self.h5_path = h5_path
        self.x_key = x_key
        self.y_key = y_key
        self.interval_size = interval_size
        self.centered = centered
        self.normalize = normalize
        self.eps = eps
        self.preload = preload
        self._file = None
        self.x_data = None
        self.y_data = None

        if interval_size is not None:
            assert interval_size % 2 == 0, "interval_size must be even"
            self.half_window = interval_size // 2

        with h5py.File(self.h5_path, "r") as f:
            N = len(f[self.x_key])
            if self.preload:
                print(f"  Preloading {self.h5_path} into memory... ", end="", flush=True)
                self.x_data = f[self.x_key][:]
                self.y_data = f[self.y_key][:]
                mb = (self.x_data.nbytes + self.y_data.nbytes) / 1024 ** 2
                print(f"done ({mb:.0f} MB)")

        all_idx = np.arange(N)
        self.indices = all_idx if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def _extract_centered_window(self, arr, peak_idx):
        """
        Extract interval centered exactly at peak_idx.
        Pads with zeros if necessary.
        """
        L = len(arr)

        start = peak_idx - self.half_window
        end = peak_idx + self.half_window

        pad_left = max(0, -start)
        pad_right = max(0, end - L)

        start = max(start, 0)
        end = min(end, L)

        window = arr[start:end]

        if pad_left > 0 or pad_right > 0:
            window = np.pad(window,
                            (pad_left, pad_right),
                            mode="constant")

        return window

    def _extract_quarter_peak_window(self, arr, peak_idx):
        """
        Extract interval where peak is located at 1/4 of the window.
        Pads with zeros if necessary.
        """
        L = len(arr)
        quarter = self.interval_size // 4

        # Make peak appear at 1/4 position
        start = peak_idx - quarter
        end = start + self.interval_size

        pad_left = max(0, -start)
        pad_right = max(0, end - L)

        start = max(start, 0)
        end = min(end, L)

        window = arr[start:end]

        if pad_left > 0 or pad_right > 0:
            window = np.pad(window,
                            (pad_left, pad_right),
                            mode="constant")

        return window

    def __getitem__(self, idx):
        i = int(self.indices[idx])

        # Load from preloaded arrays or stream from H5
        if self.preload:
            x = self.x_data[i]
            y = self.y_data[i]
        else:
            self._ensure_open()
            x = self._file[self.x_key][i]
            y = self._file[self.y_key][i]

        # ---- Optional interval mode ----
        if self.interval_size is not None:
            peak_idx = np.argmax(y)
            if self.centered:
                x = self._extract_centered_window(x, peak_idx)
                y = self._extract_centered_window(y, peak_idx)
            else:
                x = self._extract_quarter_peak_window(x, peak_idx)
                y = self._extract_quarter_peak_window(y, peak_idx)

        # ---- Optional normalization ----
        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + self.eps)
            y = (y - y.min()) / (y.max() - y.min() + self.eps)

        # Add feature dimension (T,1)
        x = np.expand_dims(x.astype(np.float32), axis=-1)
        y = np.expand_dims(y.astype(np.float32), axis=-1)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception as e:
            print(f"Warning: Failed to close HDF5 file: {e}")
            pass