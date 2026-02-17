"""
HDF5 writing operations for FIDs, FFTs, and training segments.
"""
import numpy as np
import h5py
from typing import List, Tuple
import os
from src.config.settings import SimulationConfig



def write_fid_batch(
    h5_hr: h5py.Dataset,
    h5_mid: h5py.Dataset,
    h5_low: h5py.Dataset,
    fid_buffer: np.ndarray,
    start_idx: int,
    end_idx: int,
    n_high: int,
    n_mid: int,
    n_low: int,
    win_hr: np.ndarray,
    win_mid: np.ndarray,
    win_low: np.ndarray,
) -> None:
    """
    Write a batch of FID signals to the three resolution datasets.
    Windowing is applied before storage.
    """
    h5_hr[start_idx:end_idx, :n_high] = fid_buffer[:, :n_high] * win_hr[None, :]
    h5_mid[start_idx:end_idx, :n_mid] = fid_buffer[:, :n_mid] * win_mid[None, :]
    h5_low[start_idx:end_idx, :n_low] = fid_buffer[:, :n_low] * win_low[None, :]


def write_fft_batch(
    h5_hr: h5py.Dataset,
    h5_mid: h5py.Dataset,
    h5_low: h5py.Dataset,
    mag_buffer_hr: np.ndarray,
    mag_buffer_mid: np.ndarray,
    mag_buffer_low: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> None:
    """
    Normalise each spectrum by its own maximum and write to HDF5.
    """
    # Per-row normalization
    norm_hr = mag_buffer_hr / mag_buffer_hr.max(axis=1, keepdims=True)
    norm_mid = mag_buffer_mid / mag_buffer_mid.max(axis=1, keepdims=True)
    norm_low = mag_buffer_low / mag_buffer_low.max(axis=1, keepdims=True)

    h5_hr[start_idx:end_idx] = norm_hr
    h5_mid[start_idx:end_idx] = norm_mid
    h5_low[start_idx:end_idx] = norm_low


def save_filtered_segments(
    config: SimulationConfig,
    formulas: List[str],
    fft_freq: np.ndarray,
    segments_hr: np.ndarray,
    segments_mid: np.ndarray,
    segments_low: np.ndarray,
    spans: List[Tuple[int, int]],
    keep_mask: np.ndarray,
    upper_bound: float,
) -> None:
    """
    Save cropped, outlier‑filtered spectra with reconstruction metadata.
    This is the final training dataset.
    """
    # Keep only rows that passed the IQR filter
    kept_idx = np.where(keep_mask)[0]
    seg_hr_kept = segments_hr[keep_mask, : int(upper_bound)]
    seg_mid_kept = segments_mid[keep_mask, : int(upper_bound)]
    seg_low_kept = segments_low[keep_mask, : int(upper_bound)]

    # Original start/end for each kept row
    spans_arr = np.array(spans, dtype=np.int64)
    starts_kept = spans_arr[kept_idx, 0]
    ends_orig_kept = spans_arr[kept_idx, 1]

    # Effective end after cropping to upper_bound
    kept_len = np.minimum(ends_orig_kept - starts_kept, int(upper_bound)).astype(np.int64)
    ends_eff_kept = starts_kept + kept_len

    spans_orig_kept = np.stack([starts_kept, ends_orig_kept], axis=1)
    spans_eff_kept = np.stack([starts_kept, ends_eff_kept], axis=1)

    # Compounds in same order
    compounds_np = np.array(formulas, dtype=h5py.string_dtype())
    compounds_kept = compounds_np[kept_idx]

    # Output file name
    out_file = config.SEGMENTS_H5.format(n_high=config.N_POINTS_FID)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with h5py.File(out_file, "w") as h5f:
        # Spectra
        h5f.create_dataset(
            "fft_hr", data=seg_hr_kept,
            compression="gzip", compression_opts=9, chunks=True, shuffle=True
        )
        h5f.create_dataset(
            "fft_mid", data=seg_mid_kept,
            compression="gzip", compression_opts=9, chunks=True, shuffle=True
        )
        h5f.create_dataset(
            "fft_low", data=seg_low_kept,
            compression="gzip", compression_opts=9, chunks=True, shuffle=True
        )

        # Metadata for reconstruction
        h5f.create_dataset("fft_spans_orig", data=spans_orig_kept)
        h5f.create_dataset("fft_spans_eff", data=spans_eff_kept)
        h5f.create_dataset("kept_len", data=kept_len)
        h5f.create_dataset("upper_used", data=np.array([upper_bound], dtype=np.int64))
        h5f.create_dataset("indices_kept", data=kept_idx.astype(np.int64))
        h5f.create_dataset("compounds", data=compounds_kept)
        h5f.create_dataset("fft_freq", data=fft_freq)

        # Global attributes
        h5f.attrs["N_target"] = config.N_POINTS_FID * config.ZERO_FILL_FACTOR
        h5f.attrs["N_high"] = config.N_POINTS_FID
        h5f.attrs["sampling_rate"] = config.SAMPLING_RATE
        h5f.attrs["sampling_interval"] = 1.0 / config.SAMPLING_RATE
        h5f.attrs["note"] = (
            "Segments cropped to 'upper_used' bins. Use 'fft_spans_eff' and 'kept_len' "
            "to map predictions back to the full frequency axis ('fft_freq')."
        )

    print(f"\n✅ Training data saved to '{out_file}'")
    print(f"   Kept compounds: {len(kept_idx)} / {len(formulas)}")
    print(f"   Segment length: {int(upper_bound)} bins")
    print(f"   Output shape: {seg_hr_kept.shape}")