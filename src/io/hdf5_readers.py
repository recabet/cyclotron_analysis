"""
HDF5 reading operations for visualization and analysis.
"""
import h5py
import numpy as np
from typing import Tuple, Dict, Any


def load_fid_compound(
        fid_file: str, compound_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load FID signals for a single compound at all resolutions.

    Returns:
        (high_res_fid, medium_res_fid, low_res_fid)
    """
    with h5py.File(fid_file, "r") as f:
        hr = f["fid_hr"][compound_idx]
        mid = f["fid_mid"][compound_idx]
        low = f["fid_low"][compound_idx]
    return hr, mid, low


def load_spectrum_compound(
        fft_file: str, compound_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load normalized FFT magnitude spectra for a single compound.

    Returns:
        (frequency_axis, high_res_spec, medium_res_spec, low_res_spec)
    """
    with h5py.File(fft_file, "r") as f:
        freq = f["fft_freq"][0]
        hr = f["fft_hr"][compound_idx]
        mid = f["fft_mid"][compound_idx]
        low = f["fft_low"][compound_idx]
    return freq, hr, mid, low


def load_training_data(
        training_file: str
) -> Dict[str, Any]:
    """
    Load the complete training dataset with metadata.

    Returns:
        Dictionary containing:
        - 'fft_hr', 'fft_mid', 'fft_low': cropped spectra
        - 'fft_freq': full frequency axis
        - 'spans_eff': effective (start, end) indices for reconstruction
        - 'kept_len': actual lengths per compound
        - 'compounds': formula strings
        - 'indices_kept': original indices in full dataset
        - plus attributes
    """
    data = {}
    with h5py.File(training_file, "r") as f:
        # Spectra
        data['fft_hr'] = f['fft_hr'][:]
        data['fft_mid'] = f['fft_mid'][:]
        data['fft_low'] = f['fft_low'][:]

        # Metadata
        data['fft_freq'] = f['fft_freq'][:]
        data['spans_eff'] = f['fft_spans_eff'][:]
        data['kept_len'] = f['kept_len'][:]
        data['compounds'] = f['compounds'][:]
        data['indices_kept'] = f['indices_kept'][:]

        # Attributes
        data['attrs'] = dict(f.attrs)

    return data


def reconstruct_full_spectrum(
        cropped_spectrum: np.ndarray,
        start_idx: int,
        full_length: int,
        pad_value: float = 0.0
) -> np.ndarray:
    """
    Reconstruct a full-length spectrum from a cropped segment.

    Args:
        cropped_spectrum: 1D array of peak region
        start_idx: Starting index in the full spectrum
        full_length: Length of the full spectrum
        pad_value: Value for regions outside the cropped segment

    Returns:
        Full-length spectrum with peaks placed at correct positions
    """
    full = np.full(full_length, pad_value, dtype=cropped_spectrum.dtype)
    end_idx = min(start_idx + len(cropped_spectrum), full_length)
    full[start_idx:end_idx] = cropped_spectrum[:end_idx - start_idx]
    return full