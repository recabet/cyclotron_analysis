"""
Adaptive peak region detection and cropping for mass spectra.
"""
from typing import List, Tuple, Optional
import numpy as np


def _smooth_moving_avg(x: np.ndarray, win: int = 7) -> np.ndarray:
    """1D moving average with odd window length."""
    win = max(1, win)
    if win % 2 == 0:
        win += 1
    if win == 1:
        return x.copy()
    kernel = np.ones(win, dtype=x.dtype) / win
    return np.convolve(x, kernel, mode="same")


def find_peak_span(
        x: np.ndarray,
        k_sigma: float = 3.0,
        min_floor: float = 0.01,
        smooth_win: int = 7,
        pad_margin: int = 10,
        gap_merge: int = 5,
) -> Optional[Tuple[int, int]]:
    """
    Locate the contiguous region covering all significant peaks in a spectrum.

    Strategy:
        1. Smooth to reduce noise
        2. Estimate noise level from low-percentile region
        3. Threshold = baseline + k_sigma * noise_std
        4. Find contiguous regions above threshold
        5. Merge nearby regions
        6. Add margins

    Args:
        x: 1D spectrum (normalized [0,1] recommended)
        k_sigma: Multiplier for noise standard deviation
        min_floor: Minimum absolute threshold
        smooth_win: Moving average window length
        pad_margin: Extra bins to add on each side
        gap_merge: Merge peak groups separated by ≤ this many bins

    Returns:
        (start, end) indices, or None if no peaks detected
    """
    xs = _smooth_moving_avg(x, win=smooth_win)

    # Estimate baseline & noise from low percentiles
    low_vals = xs[xs <= np.percentile(xs, 20)]
    noise_std = np.std(low_vals) if low_vals.size > 0 else np.std(xs)
    baseline = np.percentile(xs, 5)
    threshold = max(baseline + k_sigma * noise_std, min_floor)

    # Mask of indices above threshold
    mask = xs >= threshold
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None

    # Group consecutive indices
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    # Merge groups that are very close
    if starts.size > 1 and gap_merge > 0:
        merged_starts = [starts[0]]
        merged_ends = [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - merged_ends[-1] - 1 <= gap_merge:
                merged_ends[-1] = e
            else:
                merged_starts.append(s)
                merged_ends.append(e)
        starts, ends = np.array(merged_starts), np.array(merged_ends)

    # Return span from first to last peak group with margins
    start = max(0, int(starts[0]) - pad_margin)
    end = min(x.shape[0], int(ends[-1]) + pad_margin + 1)
    return start, end


def extract_peak_segments(
        spectra_hr: np.ndarray,
        spectra_mid: np.ndarray,
        spectra_low: np.ndarray,
        k_sigma: float = 3.0,
        min_floor: float = 0.01,
        smooth_win: int = 7,
        pad_margin: int = 10,
        gap_merge: int = 5,
        pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], int]:
    """
    Crop all spectra to their peak regions and right-pad to uniform length.

    Args:
        spectra_hr: 2D array (n_compounds, n_bins) - high resolution spectra
        spectra_mid: 2D array - medium resolution spectra
        spectra_low: 2D array - low resolution spectra
        ... (peak detection parameters as in find_peak_span)
        pad_value: Value to use for right-padding

    Returns:
        segments_hr, segments_mid, segments_low: Cropped/padded arrays (N, max_len)
        spans: List of (start, end) for each input row
        max_len: Length of the longest segment
    """
    # Ensure 2D
    arr_hr = spectra_hr if spectra_hr.ndim == 2 else spectra_hr[None, :]
    arr_mid = spectra_mid if spectra_mid.ndim == 2 else spectra_mid[None, :]
    arr_low = spectra_low if spectra_low.ndim == 2 else spectra_low[None, :]

    n_rows = arr_hr.shape[0]
    spans = []
    lengths = []

    for i in range(n_rows):
        span = find_peak_span(
            arr_hr[i],
            k_sigma=k_sigma,
            min_floor=min_floor,
            smooth_win=smooth_win,
            pad_margin=pad_margin,
            gap_merge=gap_merge,
        )
        if span is None:
            spans.append((None, None))
            lengths.append(0)
        else:
            s, e = span
            spans.append((s, e))
            lengths.append(e - s)

    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        return (np.empty((n_rows, 0), dtype=arr_hr.dtype),
                np.empty((n_rows, 0), dtype=arr_hr.dtype),
                np.empty((n_rows, 0), dtype=arr_hr.dtype),
                spans, 0)

    # Allocate padded arrays
    seg_hr = np.full((n_rows, max_len), pad_value, dtype=arr_hr.dtype)
    seg_mid = np.full((n_rows, max_len), pad_value, dtype=arr_mid.dtype)
    seg_low = np.full((n_rows, max_len), pad_value, dtype=arr_low.dtype)

    for i, (s, e) in enumerate(spans):
        if s is None:
            continue
        length_i = e - s
        seg_hr[i, :length_i] = arr_hr[i, s:e]
        seg_mid[i, :length_i] = arr_mid[i, s:e]
        seg_low[i, :length_i] = arr_low[i, s:e]

    return seg_hr, seg_mid, seg_low, spans, max_len