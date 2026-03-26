"""
FFT computation and normalization for mass spectra.
"""
import numpy as np
from typing import Tuple


def compute_fft_magnitude( signal: np.ndarray,
                           n_fft: int,
                           sampling_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute real FFT magnitude normalized by n_fft.

    Args:
        signal: Time-domain signal (FID)
        n_fft: Number of FFT points (zero-padding applied if > len(signal))
        sampling_interval: Time between samples [s]

    Returns:
        freq_axis: Frequency bins [Hz]
        magnitude: |FFT| / n_fft, float32
    """
    mag = np.abs(np.fft.rfft(signal, n=n_fft)) / n_fft
    freqs = np.fft.rfftfreq(n_fft, d=sampling_interval)
    return freqs, mag.astype(np.float32)


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Normalize a spectrum to [0, 1] by its maximum.

    Args:
        spectrum: 1D or 2D array (rows = spectra, columns = frequency bins)

    Returns:
        Normalized spectrum (same shape)
    """
    if spectrum.ndim == 1:
        return spectrum / spectrum.max()
    else:
        # Per-row normalization
        return spectrum / spectrum.max(axis=1, keepdims=True)
    
    
    
def extract_middle_segment(signal: np.ndarray,
                           start_ratio: float = 0.25,
                           end_ratio: float = 0.5) -> np.ndarray:
    """
    Extract a middle segment of the signal (around the 2nd quartile)
    to avoid noisy beginning and end regions.

    Args:
        signal: 1D FID signal
        start_ratio: Start position as fraction of signal length
        end_ratio: End position as fraction of signal length

    Returns:
        Cropped signal segment
    """
    n = len(signal)

    start = int(n * start_ratio)
    end = int(n * end_ratio)

    return signal[start:end]