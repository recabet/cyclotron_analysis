#!/usr/bin/env python
"""
FT‑ICR MS Training Data Generation Pipeline.

Run this script to generate complete training datasets:
1. FID signals (3 resolutions) → fid_levels.h5
2. Full FFT spectra (3 resolutions) → fft_full_spectra.h5
3. Cropped, filtered peak segments → training_segments_65536.h5

This script contains NO visualization code - it only generates and saves data.
"""
import os
import numpy as np
import h5py

from src.config import SimulationConfig
from src.isotope import load_compounds, process_all_compounds
from src.signal_processing import damping_envelope, kaiser_window, generate_fid
from src.signal_processing import compute_fft_magnitude
from src.processing import extract_peak_segments, iqr_outlier_filter
from src.io import write_fid_batch, write_fft_batch, save_filtered_segments



def generate_and_save_fids(config,
                           formulas,
                           frequencies,
                           amplitudes):
    """Step 1: Generate and save FID signals."""
    print("\n" + "=" * 60)
    print("STEP 1/3: Generating FID signals")
    print("=" * 60)

    n_compounds = len(formulas)

    # Time vectors
    t_full = np.arange(config.N_POINTS_FID, dtype=np.float32) / config.SAMPLING_RATE
    damp = damping_envelope(t_full, config.DAMPING_FINAL_AMP)

    # Resolution parameters
    n_high = config.N_POINTS_FID
    n_mid = n_high // config.MID_RES_FACTOR
    n_low = n_high // config.LOW_RES_FACTOR

    # Kaiser windows
    win_hr = kaiser_window(n_high, config.KAISER_BETA)
    win_mid = kaiser_window(n_mid, config.KAISER_BETA)
    win_low = kaiser_window(n_low, config.KAISER_BETA)

    # Create output directory
    os.makedirs(os.path.dirname(config.FID_H5), exist_ok=True)

    with h5py.File(config.FID_H5, "w") as h5f:
        dset_hr = h5f.create_dataset(
            "fid_hr",
            shape=(n_compounds, n_high),
            dtype="float32",
            chunks=True,
            compression=config.COMPRESSION
        )
        dset_mid = h5f.create_dataset(
            "fid_mid",
            shape=(n_compounds, n_mid),
            dtype="float32",
            chunks=True,
            compression=config.COMPRESSION
        )
        dset_low = h5f.create_dataset(
            "fid_low",
            shape=(n_compounds, n_low),
            dtype="float32",
            chunks=True,
            compression=config.COMPRESSION
        )

        # Batch processing
        batch_size = config.BATCH_SIZE
        fid_buffer = np.zeros((batch_size, n_high), dtype=np.float32)
        buffer_idx = 0

        for i in range(n_compounds):
            fid = generate_fid(
                frequencies[i], amplitudes[i], t_full, damp,
                phase=0.0,
                max_amp=config.MAX_AMPLITUDE,
                noise_level=config.NOISE_LEVEL
            )
            fid_buffer[buffer_idx] = fid
            buffer_idx += 1

            print(f"  FID generated: {i + 1:5d} / {n_compounds}", end="\r")

            if buffer_idx == batch_size or i + 1 == n_compounds:
                end = i + 1
                start = end - buffer_idx
                write_fid_batch(
                    dset_hr, dset_mid, dset_low,
                    fid_buffer[:buffer_idx], start, end,
                    n_high, n_mid, n_low,
                    win_hr, win_mid, win_low
                )
                buffer_idx = 0
                fid_buffer = np.zeros((batch_size, n_high), dtype=np.float32)

    print(f"\n✅ FID signals saved to '{config.FID_H5}'")


def compute_and_save_ffts(config, n_compounds):
    """Step 2: Compute and save FFT magnitudes."""
    print("\n" + "=" * 60)
    print("STEP 2/3: Computing FFT magnitudes")
    print("=" * 60)

    n_target = config.N_POINTS_FID * config.ZERO_FILL_FACTOR
    fft_size = n_target // 2 + 1
    sampling_interval = 1.0 / config.SAMPLING_RATE

    with h5py.File(config.FID_H5, "r") as fid_file, \
            h5py.File(config.FFT_H5, "w") as fft_file:

        # Frequency axis
        freq_axis = np.fft.rfftfreq(n_target, d=sampling_interval).astype(np.float32)
        fft_file.create_dataset(
            "fft_freq",
            data=freq_axis[None, :],
            compression=config.COMPRESSION
        )

        # Datasets for normalized spectra
        dset_hr = fft_file.create_dataset(
            "fft_hr", shape=(n_compounds, fft_size),
            dtype="float32",
            chunks=True,
            compression=config.COMPRESSION, shuffle=True
        )
        dset_mid = fft_file.create_dataset(
            "fft_mid",
            shape=(n_compounds, fft_size),
            dtype="float32",
            chunks=True,
            compression=config.COMPRESSION,
            shuffle=True
        )
        dset_low = fft_file.create_dataset(
            "fft_low",
            shape=(n_compounds, fft_size),
            dtype="float32",
            chunks=True,
            compression=config.COMPRESSION,
            shuffle=True
        )

        # Batch buffers
        batch_size = config.BATCH_SIZE
        buf_hr = np.zeros((batch_size, fft_size), dtype=np.float32)
        buf_mid = np.zeros((batch_size, fft_size), dtype=np.float32)
        buf_low = np.zeros((batch_size, fft_size), dtype=np.float32)
        buf_idx = 0

        for i in range(n_compounds):
            sig_hr = fid_file["fid_hr"][i]
            sig_mid = fid_file["fid_mid"][i]
            sig_low = fid_file["fid_low"][i]

            _, mag_hr = compute_fft_magnitude(sig_hr, n_target, sampling_interval)
            _, mag_mid = compute_fft_magnitude(sig_mid, n_target, sampling_interval)
            _, mag_low = compute_fft_magnitude(sig_low, n_target, sampling_interval)

            buf_hr[buf_idx] = mag_hr
            buf_mid[buf_idx] = mag_mid
            buf_low[buf_idx] = mag_low
            buf_idx += 1

            print(f"  FFT computed: {i + 1:5d} / {n_compounds}", end="\r")

            if buf_idx == batch_size or i + 1 == n_compounds:
                end = i + 1
                start = end - buf_idx
                write_fft_batch(
                    dset_hr, dset_mid, dset_low,
                    buf_hr[:buf_idx], buf_mid[:buf_idx], buf_low[:buf_idx],
                    start, end
                )
                buf_idx = 0
                buf_hr = np.zeros((batch_size, fft_size), dtype=np.float32)
                buf_mid = np.zeros((batch_size, fft_size), dtype=np.float32)
                buf_low = np.zeros((batch_size, fft_size), dtype=np.float32)

    print(f"\n✅ Normalized FFT spectra saved to '{config.FFT_H5}'")


def extract_and_filter_peaks(config, formulas):
    """Step 3: Extract peak regions, filter outliers, save training data."""
    print("\n" + "=" * 60)
    print("STEP 3/3: Extracting peak regions and filtering")
    print("=" * 60)

    # Load full spectra
    with h5py.File(config.FFT_H5, "r") as f:
        fft_freq = f["fft_freq"][0]
        spec_hr = f["fft_hr"][:]
        spec_mid = f["fft_mid"][:]
        spec_low = f["fft_low"][:]

    print(f"  Loaded spectra: HR {spec_hr.shape}, MID {spec_mid.shape}, LOW {spec_low.shape}")

    seg_hr, seg_mid, seg_low, spans, max_len = extract_peak_segments(
        spec_hr, spec_mid, spec_low,
        k_sigma=config.PEAK_K_SIGMA,
        min_floor=config.PEAK_MIN_FLOOR,
        smooth_win=config.PEAK_SMOOTH_WIN,
        pad_margin=config.PEAK_PAD_MARGIN,
        gap_merge=config.PEAK_GAP_MERGE,
        pad_value=0.0,
    )

    lengths = np.array([e - s if s is not None else 0 for s, e in spans])
    print(f"  Peak regions extracted. Max length: {max_len} bins")

    keep_mask, drop_mask, lower, upper = iqr_outlier_filter(
        lengths, k=config.IQR_K
    )
    print(f"  IQR outlier detection: lower={lower:.1f}, upper={upper:.1f}")
    print(f"  Keep: {keep_mask.sum()}, Drop: {drop_mask.sum()}")

    save_filtered_segments(
        config, formulas, fft_freq,
        seg_hr, seg_mid, seg_low,
        spans, keep_mask, upper
    )

    return keep_mask, drop_mask, lower, upper, lengths


def main():
    """Run the complete data generation pipeline."""
    config = SimulationConfig()

    print("\n" + "=" * 60)
    print("FT-ICR MS TRAINING DATA GENERATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Compounds file: {config.COMPOUNDS_FILE}")
    print(f"  FID points: {config.N_POINTS_FID}")
    print(f"  Zero fill factor: {config.ZERO_FILL_FACTOR}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Output directory: data/")
    print("=" * 60 + "\n")


    formulas, masses, _, rel_abund = load_compounds(
        config.COMPOUNDS_FILE, coverage=config.COVERAGE_PROB
    )

    frequencies = process_all_compounds(
        masses, config.AVOGADRO, config.MAGNETIC_FIELD,
        config.ION_CHARGE, config.ELECTRON_CHARGE
    )

    generate_and_save_fids(config, formulas, frequencies, rel_abund)

    compute_and_save_ffts(config, len(formulas))

    extract_and_filter_peaks(config, formulas)

    print("\n" + "=" * 60)
    print("✅ DATA GENERATION COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - {config.FID_H5}         (raw FID signals)")
    print(f"  - {config.FFT_H5}  (full FFT spectra)")
    print(f"  - {config.SEGMENTS_H5.format(n_high=config.N_POINTS_FID)}   (cropped training segments)")
    print("\nNext steps:")
    print("  1. Run 'python inspect_generated_data.py' to visualize the generated data")
    print("  2. Use the training segments for machine learning")


if __name__ == "__main__":
    main()