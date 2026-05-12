#!/usr/bin/env python
"""
FT‑ICR MS Training Data Generation with LMDB Storage and Sanity Check Plots.

Generates full-spectrum training data and saves to LMDB databases:
- full.lmdb: All data
- train.lmdb: Training split (75%)
- val.lmdb: Validation split (20%)
- test.lmdb: Test split (5%)

Creates sanity check train_plots at each step:
1. FID generation (raw signals at different resolutions)
2. Apodization and noise (windowing and noise addition)
3. Zero-filling and FFT (frequency domain transformation)
4. Low vs high resolution comparison
5. Pipeline summary

LMDB advantages:
- Fast random access
- Memory-mapped for efficient reading
- No file handle issues with multiprocessing
"""
import os
import sys
import lmdb
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.config import SimulationConfig
from src.isotope import load_compounds, process_all_compounds
from src.signal_processing import damping_envelope, kaiser_window, generate_fid, add_noise
from src.signal_processing.fft import compute_fft_magnitude
from src.processing import extract_peak_segments, iqr_outlier_filter


def create_lmdb_databases(lmdb_dir: str, map_size: int = 200 * 1024**3):
    """Create LMDB databases for full, train, val, test splits."""
    os.makedirs(lmdb_dir, exist_ok=True)

    dbs = {}
    for split in ['full', 'train', 'val', 'test']:
        db_path = os.path.join(lmdb_dir, f'{split}.lmdb')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        env = lmdb.open(db_path, map_size=map_size, writemap=True)
        dbs[split] = env

    return dbs


def write_to_lmdb(env, key: str, data: dict):
    """Write a data sample to LMDB."""
    txn = env.begin(write=True)
    txn.put(key.encode('ascii'), pickle.dumps(data))
    txn.commit()


def plot_fid_generation_sample(fid_hr, fid_mid, fid_low, t_hr, t_mid, t_low,
                                sample_idx, output_dir):
    """Plot FID signals at different resolutions."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # High resolution
    axes[0].plot(t_hr * 1000, fid_hr, 'r-', linewidth=1.5, label='High-Res')
    axes[0].set_xlabel('Time [ms]', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'FID Generation - Sample {sample_idx}: High-Res ({len(fid_hr)} points)',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Medium resolution
    axes[1].plot(t_mid * 1000, fid_mid, 'g-', linewidth=1.5, label='Mid-Res')
    axes[1].set_xlabel('Time [ms]', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title(f'Medium-Res ({len(fid_mid)} points)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    # Low resolution
    axes[2].plot(t_low * 1000, fid_low, 'b-', linewidth=1.5, label='Low-Res')
    axes[2].set_xlabel('Time [ms]', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)
    axes[2].set_title(f'Low-Res ({len(fid_low)} points)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fid_generation_sample_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_apodization_and_noise(fid_raw, fid_with_noise, fid_noisy_windowed, t,
                                sample_idx, output_dir):
    """Plot apodization and noise addition steps."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Raw FID (no window, no noise)
    axes[0].plot(t * 1000, fid_raw, 'k-', linewidth=1.5, label='Raw FID')
    axes[0].set_xlabel('Time [ms]', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'Apodization & Noise - Sample {sample_idx}: Raw FID (no window, no noise)',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # FID with noise added (before window)
    axes[1].plot(t * 1000, fid_with_noise, 'orange', linewidth=1.5, label='After Noise Addition')
    axes[1].set_xlabel('Time [ms]', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title('After Noise Addition (simulates experimental conditions)',
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    # Final FID (noise + window applied)
    axes[2].plot(t * 1000, fid_noisy_windowed, 'purple', linewidth=1.5, label='After Kaiser Window')
    axes[2].set_xlabel('Time [ms]', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)
    axes[2].set_title('After Kaiser Window Apodization (reduces spectral leakage)',
                     fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'apodization_noise_sample_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_zero_filling_and_fft(fid, fid_padded, fft_mag, freq_axis,
                               sample_idx, output_dir):
    """Plot zero-filling and FFT transformation."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Time domain: before and after zero-filling
    t_original = np.arange(len(fid)) / 1e6
    t_padded = np.arange(len(fid_padded)) / 1e6

    axes[0].plot(t_original * 1000, fid, 'b-', linewidth=1.5, label='Original FID')
    axes[0].plot(t_padded * 1000, fid_padded, 'r--', linewidth=1.0, alpha=0.7, label='Zero-Filled FID')
    axes[0].set_xlabel('Time [ms]', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'Zero-Filling & FFT - Sample {sample_idx}: Time Domain',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Frequency domain: FFT magnitude
    axes[1].plot(freq_axis / 1000, fft_mag, 'g-', linewidth=1.5, label='FFT Magnitude')
    axes[1].set_xlabel('Frequency [kHz]', fontsize=11)
    axes[1].set_ylabel('Magnitude', fontsize=11)
    axes[1].set_title('Frequency Domain: FFT Magnitude Spectrum',
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'zero_fill_fft_sample_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_low_high_res_comparison(fft_low, fft_hr, freq_axis,
                                  sample_idx, output_dir):
    """Plot low-resolution vs high-resolution FFT comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot both on same axes
    ax.plot(freq_axis / 1000, fft_low, 'b-', linewidth=1.5, alpha=0.7, label='Low-Res')
    ax.plot(freq_axis / 1000, fft_hr, 'r-', linewidth=1.5, alpha=0.7, label='High-Res')

    ax.set_xlabel('Frequency [kHz]', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title(f'Low-Res vs High-Res Comparison - Sample {sample_idx}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'low_high_res_comparison_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_pipeline_summary(sample_data, output_dir):
    """Create a 4-panel summary showing the complete pipeline."""
    n_samples = len(sample_data)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Pipeline Summary: From FID to FFT',
                 fontsize=14, fontweight='bold')

    for i, data in enumerate(sample_data):
        fid_hr = data['fid_hr']
        fid_windowed = data['fid_windowed']
        fft_low = data['fft_low']
        fft_hr = data['fft_hr']
        t = data['t']
        freq_axis = data['freq_axis']
        sample_idx = data['sample_idx']

        # Panel 1: FID (high-res)
        ax = axes[i, 0]
        ax.plot(t * 1000, fid_hr, 'r-', linewidth=1.5)
        ax.set_xlabel('Time [ms]', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Sample {sample_idx}: FID', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Panel 2: Windowed FID
        ax = axes[i, 1]
        ax.plot(t * 1000, fid_windowed, 'orange', linewidth=1.5)
        ax.set_xlabel('Time [ms]', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title('Windowed FID', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Panel 3: Low-res FFT
        ax = axes[i, 2]
        ax.plot(freq_axis / 1000, fft_low, 'b-', linewidth=1.5)
        ax.set_xlabel('Frequency [kHz]', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.set_title('Low-Res FFT', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Panel 4: High-res FFT
        ax = axes[i, 3]
        ax.plot(freq_axis / 1000, fft_hr, 'r-', linewidth=1.5)
        ax.set_xlabel('Frequency [kHz]', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.set_title('High-Res FFT', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pipeline_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def generate_full_spectrum_data(config: SimulationConfig, batch_size: int = 1000):
    """Generate full-spectrum training data and save to LMDB."""
    print("\n" + "=" * 80)
    print("FT-ICR MS FULL-SPECTRUM DATA GENERATION (LMDB)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Compounds file: {config.COMPOUNDS_FILE}")
    print(f"  FID points: {config.N_POINTS_FID}")
    print(f"  Zero fill factor: {config.ZERO_FILL_FACTOR}")
    print(f"  Batch size: {batch_size}")
    print("=" * 80 + "\n")

    # Load compounds
    formulas, masses, _, rel_abund = load_compounds(
        config.COMPOUNDS_FILE, coverage=config.COVERAGE_PROB
    )

    frequencies = process_all_compounds(
        masses, config.AVOGADRO, config.MAGNETIC_FIELD,
        config.ION_CHARGE, config.ELECTRON_CHARGE
    )

    n_compounds = len(formulas)
    print(f"Loaded {n_compounds} compounds")

    # Time vectors
    t_full = np.arange(config.N_POINTS_FID, dtype=np.float32) / config.SAMPLING_RATE
    damp = damping_envelope(t_full, config.DAMPING_FINAL_AMP)

    # Resolution parameters
    n_high = config.N_POINTS_FID
    n_mid = n_high // config.MID_RES_FACTOR
    n_low = n_high // config.LOW_RES_FACTOR

    t_mid = np.arange(n_mid, dtype=np.float32) / config.SAMPLING_RATE
    t_low = np.arange(n_low, dtype=np.float32) / config.SAMPLING_RATE

    # Kaiser windows
    win_hr = kaiser_window(n_high, config.KAISER_BETA)
    win_mid = kaiser_window(n_mid, config.KAISER_BETA)
    win_low = kaiser_window(n_low, config.KAISER_BETA)

    # FFT parameters
    n_target = config.N_POINTS_FID * config.ZERO_FILL_FACTOR
    fft_size = n_target // 2 + 1
    sampling_interval = 1.0 / config.SAMPLING_RATE
    freq_axis = np.fft.rfftfreq(n_target, d=sampling_interval).astype(np.float32)

    print("\n" + "=" * 80)
    print("STEP 1/4: GENERATING FID SIGNALS")
    print("=" * 80)

    # Store FIDs for all compounds
    fids_hr = np.zeros((n_compounds, n_high), dtype=np.float32)
    fids_mid = np.zeros((n_compounds, n_mid), dtype=np.float32)
    fids_low = np.zeros((n_compounds, n_low), dtype=np.float32)

    # Create output directory for train_plots
    plot_dir = "figures/training_full_spectrum"
    os.makedirs(plot_dir, exist_ok=True)

    # Generate FIDs
    for i in range(n_compounds):
        fid = generate_fid(
            frequencies[i], rel_abund[i], t_full, damp,
            phase=0.0,
            max_amp=config.MAX_AMPLITUDE,
            noise_level=0.0
        )

        # Add noise to raw FID
        fid_noisy = add_noise(fid, rel_abund[i], config.MAX_AMPLITUDE, config.NOISE_LEVEL)

        # Apply windows to noisy signal
        fids_hr[i] = fid_noisy * win_hr
        fids_mid[i] = fid_noisy[:n_mid] * win_mid
        fids_low[i] = fid_noisy[:n_low] * win_low

        print(f"  FID generated: {i + 1:5d} / {n_compounds}", end="\r")

        # Plot samples at regular intervals
        if i < 3:  # Plot first 3 samples
            plot_fid_generation_sample(
                fids_hr[i], fids_mid[i], fids_low[i],
                t_full, t_mid, t_low,
                i, plot_dir
            )

    print(f"\n✅ FID signals generated for {n_compounds} compounds")

    print("\n" + "=" * 80)
    print("STEP 2/4: COMPUTING FFT MAGNITUDES")
    print("=" * 80)

    # Compute FFTs
    ffts_hr = np.zeros((n_compounds, fft_size), dtype=np.float32)
    ffts_mid = np.zeros((n_compounds, fft_size), dtype=np.float32)
    ffts_low = np.zeros((n_compounds, fft_size), dtype=np.float32)

    for i in range(n_compounds):
        _, mag_hr = compute_fft_magnitude(fids_hr[i], n_target, sampling_interval)
        _, mag_mid = compute_fft_magnitude(fids_mid[i], n_target, sampling_interval)
        _, mag_low = compute_fft_magnitude(fids_low[i], n_target, sampling_interval)

        ffts_hr[i] = mag_hr
        ffts_mid[i] = mag_mid
        ffts_low[i] = mag_low

        print(f"  FFT computed: {i + 1:5d} / {n_compounds}", end="\r")

        # Plot samples at regular intervals
        if i < 3:  # Plot first 3 samples
            # Plot apodization and noise (pipeline: FID -> add noise -> apply window)
            fid_raw = generate_fid(
                frequencies[i], rel_abund[i], t_full, damp,
                phase=0.0,
                max_amp=config.MAX_AMPLITUDE,
                noise_level=0.0
            )
            fid_with_noise = add_noise(fid_raw, rel_abund[i], config.MAX_AMPLITUDE, config.NOISE_LEVEL)
            fid_noisy_windowed = fid_with_noise * win_hr
            plot_apodization_and_noise(fid_raw, fid_with_noise, fid_noisy_windowed, t_full, i, plot_dir)

            # Plot zero-filling and FFT
            fid_padded = np.pad(fids_hr[i], (0, n_target - n_high), mode='constant')
            plot_zero_filling_and_fft(fids_hr[i], fid_padded, mag_hr, freq_axis, i, plot_dir)

            # Plot low vs high res comparison
            plot_low_high_res_comparison(mag_low, mag_hr, freq_axis, i, plot_dir)

    print(f"\n✅ FFT magnitudes computed for {n_compounds} compounds")

    print("\n" + "=" * 80)
    print("STEP 3/4: EXTRACTING PEAK SEGMENTS")
    print("=" * 80)

    # Extract peak segments
    seg_hr, seg_mid, seg_low, spans, max_len = extract_peak_segments(
        ffts_hr, ffts_mid, ffts_low,
        k_sigma=config.PEAK_K_SIGMA,
        min_floor=config.PEAK_MIN_FLOOR,
        smooth_win=config.PEAK_SMOOTH_WIN,
        pad_margin=config.PEAK_PAD_MARGIN,
        gap_merge=config.PEAK_GAP_MERGE,
        pad_value=0.0,
    )

    lengths = np.array([e - s if s is not None else 0 for s, e in spans])
    print(f"  Peak regions extracted. Max length: {max_len} bins")

    # Filter outliers
    keep_mask, drop_mask, lower, upper = iqr_outlier_filter(lengths, k=config.IQR_K)
    print(f"  IQR outlier detection: lower={lower:.1f}, upper={upper:.1f}")
    print(f"  Keep: {keep_mask.sum()}, Drop: {drop_mask.sum()}")

    # Filter segments
    seg_hr_filtered = seg_hr[keep_mask]
    seg_mid_filtered = seg_mid[keep_mask]
    seg_low_filtered = seg_low[keep_mask]
    spans_filtered = [spans[i] for i in range(len(spans)) if keep_mask[i]]

    print(f"✅ Extracted {len(seg_hr_filtered)} training segments")

    print("\n" + "=" * 80)
    print("STEP 4/4: SAVING TO LMDB")
    print("=" * 80)

    # Create LMDB databases
    lmdb_dir = "data/lmdb/training_segments"
    os.makedirs(lmdb_dir, exist_ok=True)

    dbs = create_lmdb_databases(lmdb_dir, map_size=200 * 1024**3)

    # Create train/val/test splits
    np.random.seed(42)
    n_segments = len(seg_hr_filtered)
    indices = np.random.permutation(n_segments)
    n_train = int(n_segments * 0.75)
    n_val = int(n_segments * 0.20)

    train_indices = set(indices[:n_train].tolist())
    val_indices = set(indices[n_train:n_train + n_val].tolist())
    test_indices = set(indices[n_train + n_val:].tolist())

    # Store metadata
    metadata = {
        'sampling_rate': config.SAMPLING_RATE,
        'n_points_fid': config.N_POINTS_FID,
        'zero_fill_factor': config.ZERO_FILL_FACTOR,
        'n_compounds': n_compounds,
        'n_segments': n_segments,
        'fft_size': fft_size,
        'max_segment_length': max_len,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_segments - n_train - n_val,
        'train_indices': sorted(train_indices),
        'val_indices': sorted(val_indices),
        'test_indices': sorted(test_indices)
    }

    # Write segments to LMDB
    train_counter = 0
    val_counter = 0
    test_counter = 0

    # Collect samples for pipeline summary
    pipeline_samples = []

    for idx in range(n_segments):
        sample = {
            'fft_low': seg_low_filtered[idx],
            'fft_mid': seg_mid_filtered[idx],
            'fft_hr': seg_hr_filtered[idx],
            'freq_axis': freq_axis,
            'start_idx': spans_filtered[idx][0],
            'end_idx': spans_filtered[idx][1],
            'sample_id': idx
        }

        # Write to full database
        full_key = f'sample_{idx:08d}'
        write_to_lmdb(dbs['full'], full_key, sample)

        # Write to split databases
        if idx in train_indices:
            train_key = f'sample_{train_counter:08d}'
            write_to_lmdb(dbs['train'], train_key, sample)
            train_counter += 1
        elif idx in val_indices:
            val_key = f'sample_{val_counter:08d}'
            write_to_lmdb(dbs['val'], val_key, sample)
            val_counter += 1
        elif idx in test_indices:
            test_key = f'sample_{test_counter:08d}'
            write_to_lmdb(dbs['test'], test_key, sample)
            test_counter += 1

        # Collect samples for pipeline summary (first 3 valid samples)
        if len(pipeline_samples) < 3:
            # Find which compound this segment came from
            compound_idx = idx // (n_segments // n_compounds + 1)
            compound_idx = min(compound_idx, n_compounds - 1)

            # Get segment bounds to compute local frequency axis
            start_idx, end_idx = spans_filtered[idx]

            # Skip if segment has None bounds
            if start_idx is None or end_idx is None:
                continue

            seg_len = end_idx - start_idx
            sampling_interval = 1.0 / config.SAMPLING_RATE
            # Local frequency axis for this segment: k / (n_target * d) for k = start_idx, ..., end_idx-1
            freq_axis_segment = (np.arange(seg_len) + start_idx) / (n_target * sampling_interval)

            pipeline_samples.append({
                'fid_hr': fids_hr[compound_idx],
                'fid_windowed': fids_hr[compound_idx],
                'fft_low': seg_low_filtered[idx],
                'fft_hr': seg_hr_filtered[idx],
                't': t_full,
                'freq_axis': freq_axis_segment,
                'sample_idx': idx
            })

        if (idx + 1) % batch_size == 0:
            print(f"  Processed: {idx + 1:6d} / {n_segments}", end="\r")

    print(f"\n✅ Saved {n_segments} samples to LMDB")

    # Write metadata
    print("\n" + "=" * 80)
    print("WRITING METADATA")
    print("=" * 80)

    for split in ['full', 'train', 'val', 'test']:
        write_to_lmdb(dbs[split], '__metadata__', metadata)
        write_to_lmdb(dbs[split], '__len__', {'n_samples': len(train_indices) if split == 'train' else
                                               len(val_indices) if split == 'val' else
                                               len(test_indices) if split == 'test' else
                                               n_segments})
        dbs[split].close()
        print(f"  ✅ {split}.lmdb: {metadata[f'n_{split}'] if split != 'full' else n_segments} samples")

    # Save freq_axis separately
    np.save(os.path.join(lmdb_dir, 'freq_axis.npy'), freq_axis)

    print(f"\n✅ LMDB databases created in '{lmdb_dir}'")
    print(f"   - full.lmdb: {n_segments} samples")
    print(f"   - train.lmdb: {n_train} samples")
    print(f"   - val.lmdb: {n_val} samples")
    print(f"   - test.lmdb: {n_segments - n_train - n_val} samples")

    # Generate pipeline summary
    print("\n" + "=" * 80)
    print("GENERATING PIPELINE SUMMARY")
    print("=" * 80)

    if pipeline_samples:
        plot_pipeline_summary(pipeline_samples, plot_dir)
    else:
        print("  ⚠️ No valid samples collected for pipeline summary (first segments had None bounds)")

    return lmdb_dir, freq_axis, metadata


def main():
    """Run the complete full-spectrum LMDB data generation pipeline."""
    config = SimulationConfig()

    lmdb_dir, freq_axis, metadata = generate_full_spectrum_data(config, batch_size=config.BATCH_SIZE)

    print("\n" + "=" * 80)
    print("✅ FULL-SPECTRUM LMDB DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {lmdb_dir}")
    print("Files:")
    print("  - full.lmdb")
    print("  - train.lmdb")
    print("  - val.lmdb")
    print("  - test.lmdb")
    print("  - freq_axis.npy")
    print("\nFigures:")
    print("  - figures/training_full_spectrum/fid_generation_sample_*.png")
    print("  - figures/training_full_spectrum/apodization_noise_sample_*.png")
    print("  - figures/training_full_spectrum/zero_fill_fft_sample_*.png")
    print("  - figures/training_full_spectrum/low_high_res_comparison_*.png")
    print("  - figures/training_full_spectrum/pipeline_summary.png")


if __name__ == "__main__":
    main()
