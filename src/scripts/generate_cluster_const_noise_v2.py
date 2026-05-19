#!/usr/bin/env python
"""
Isotope Cluster Super-Resolution LMDB Data Generation using prof_v2 Method.
Constant Noise + Correct Abundance Variant.

Same as generate_cluster_const_noise.py but with correct cross-cluster abundance:
- Noise is calculated ONCE per compound based on the global max amplitude
- The SAME noise realization is added to ALL clusters from that compound
- All clusters are normalized to M+0's top probability scale, so:
    M+0's top peak = 1.0 (amplitude scale)
    M+1's top peak ~ isotope_probability / m0_prob (e.g. 0.109 for M+1)
  This preserves true isotope abundance ratios across clusters.
"""
import os
import sys
import lmdb
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.config.settings import ClusterConfig
from src.signal_processing import kaiser_window
import IsoSpecPy as iso


def load_compounds_long(file_path: str):
    """Load compound formulas from long file."""
    formulas = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                formulas.append(line)
    return formulas


def get_isotope_clusters(formula: str, coverage_prob: float = 0.99):
    """
    Get isotope peaks grouped by mass clusters.

    Returns:
        clusters: dict mapping cluster_mass -> list of (mass, prob) tuples
    """
    # Generate isotope distribution
    sp = iso.IsoTotalProb(formula=formula, prob_to_cover=coverage_prob)

    # Get masses and probabilities
    l_mass = list(sp.masses)
    l_prob = list(sp.probs)

    # Convert to numpy and sort by mass
    a_mass = np.array(l_mass)
    a_prob = np.array(l_prob)

    sort_indices = np.argsort(a_mass)
    s_mass = a_mass[sort_indices]
    s_prob = a_prob[sort_indices]

    # Group by floor(mass) - this creates the m, m+1, m+2, ... clusters
    clusters = defaultdict(list)
    for mass, prob in zip(s_mass, s_prob):
        cluster_mass = int(np.floor(mass))
        clusters[cluster_mass].append((mass, prob))

    # Sort peaks within each cluster by probability (descending) so the
    # most abundant peak is first.
    for cm in clusters:
        clusters[cm].sort(key=lambda x: x[1], reverse=True)

    return dict(clusters)


def generate_cluster_fid_prof_v2_no_noise(
    cluster_mass: int,
    peaks: list,
    config: ClusterConfig,
    m0_prob: float
):
    """
    Generate FID for a single isotope cluster using prof_v2 method.
    Returns windowed but CLEAN signal (no noise added, no normalization).

    Args:
        cluster_mass: integer mass of this cluster (e.g., 506)
        peaks: list of (mass, prob) tuples for peaks in this cluster
        config: ClusterConfig with parameters
        m0_prob: top isotope probability of the M+0 (monoisotopic) cluster.
                 Used to normalize ALL clusters to the same abundance scale.
                 M+0's top peak will be ~1.0, M+1's ~m1_prob/m0_prob, etc.

    Returns:
        fid_windowed_hr: windowed high-res signal (clean)
        fid_windowed_lr: windowed low-res signal (clean)
        ref_amp_hr: max amplitude of fid_windowed_hr (for normalization)
        ref_amp_lr: max amplitude of fid_windowed_lr (for normalization)
        fft_mag_hr: FFT magnitude spectrum (clean)
        fft_mag_lr: FFT magnitude spectrum (clean)
        freq_axis: frequency axis
    """
    if len(peaks) == 0:
        return None, None, None, None, None, None, None

    # Extract masses and probabilities
    masses = np.array([p[0] for p in peaks])
    probs = np.array([p[1] for p in peaks])

    # Normalize ALL clusters to M+0's top probability scale.
    # This means the top peak of M+0 = 1.0, M+1 = m1_prob/m0_prob, etc.
    probs_norm = probs / m0_prob

    # ==========================
    # FREQUENCY CALCULATION
    # ==========================
    base_freq = 1.0  # Unit frequency, will be normalized anyway
    frequencies = base_freq * (masses[0] / masses)

    # Normalize frequencies: divide by (freq[0] * 2.0) for Nyquist
    freq_norm = frequencies / (frequencies[0] * 2.0)

    # Center frequencies
    fmin = np.min(freq_norm)
    fmax = np.max(freq_norm)
    center = (fmin + fmax) / 2.0
    freq_shifted = freq_norm - center

    # Zoom/stretch frequencies - THIS creates the fine structure visibility
    width = fmax - fmin

    # Special handling for single-peak clusters
    if len(frequencies) == 1:
        freq_zoomed = freq_norm - 0.4  # offset to get ~0.1 frequency
    else:
        center = (fmin + fmax) / 2.0
        freq_shifted = freq_norm - center
        if width == 0:
            freq_zoomed = freq_shifted
        else:
            freq_zoomed = freq_shifted / (2.5 * width)

    # Time vector - use RAW INDICES like professor
    t = np.arange(config.N_POINTS_FID, dtype=np.float32)

    # ==========================
    # GENERATE RAW FID
    # ==========================
    fid_raw = np.zeros(config.N_POINTS_FID, dtype=np.float32)
    for freq, prob_norm in zip(freq_zoomed, probs_norm):
        fid_raw += np.exp(-config.DAMPING_FINAL_AMP * t) * np.sin(2 * np.pi * freq * t) * prob_norm

    # ==========================
    # SIMULATE REDUCED ACQUISITION TIME
    # ==========================
    lr_points = config.lr_points()
    fid_raw_lr = fid_raw[:lr_points]  # Truncated acquisition
    fid_raw_hr = fid_raw  # Full acquisition

    # ==========================
    # APODIZATION (apply Kaiser window)
    # ==========================
    window_hr = kaiser_window(len(fid_raw_hr), config.KAISER_BETA)
    fid_windowed_hr = fid_raw_hr * window_hr

    window_lr = kaiser_window(len(fid_raw_lr), config.KAISER_BETA)
    fid_windowed_lr = fid_raw_lr * window_lr

    # Reference amplitudes (before noise, for normalization)
    ref_amp_hr = np.abs(fid_windowed_hr).max()
    ref_amp_lr = np.abs(fid_windowed_lr).max()

    # ==========================
    # FFT (clean, for reference)
    # ==========================
    fft_size = config.fft_size()
    fft_output_size = config.fft_output_size()

    padded_hr = np.zeros(fft_size, dtype=np.float32)
    padded_hr[:config.N_POINTS_FID] = fid_windowed_hr
    fft_complex_hr = np.fft.rfft(padded_hr)
    fft_mag_hr = np.abs(fft_complex_hr).astype(np.float32)[:fft_output_size]

    padded_lr = np.zeros(fft_size, dtype=np.float32)
    padded_lr[:len(fid_windowed_lr)] = fid_windowed_lr
    fft_complex_lr = np.fft.rfft(padded_lr)
    fft_mag_lr = np.abs(fft_complex_lr).astype(np.float32)[:fft_output_size]

    # Frequency axis
    sampling_interval = 1.0 / config.SAMPLING_RATE
    freq_axis = np.fft.rfftfreq(fft_size, d=sampling_interval).astype(np.float32)[:fft_output_size]

    return fid_windowed_hr, fid_windowed_lr, ref_amp_hr, ref_amp_lr, fft_mag_hr, fft_mag_lr, freq_axis


def apply_constant_noise(
    fid_windowed_hr: np.ndarray,
    fid_windowed_lr: np.ndarray,
    ref_amp_hr: float,
    ref_amp_lr: float,
    noise_hr: np.ndarray,
    noise_lr: np.ndarray
):
    """
    Add constant noise (same for all clusters) and normalize.
    Noise amplitude is determined by the global max amplitude of the compound.
    """
    # Add noise
    fid_noisy_hr = fid_windowed_hr + noise_hr
    fid_noisy_lr = fid_windowed_lr + noise_lr

    # Normalize using clean reference amplitudes (not noisy ones)
    if ref_amp_hr > 0:
        fid_normalized_hr = fid_noisy_hr / ref_amp_hr
    else:
        fid_normalized_hr = fid_noisy_hr

    if ref_amp_lr > 0:
        fid_normalized_lr = fid_noisy_lr / ref_amp_lr
    else:
        fid_normalized_lr = fid_noisy_lr

    return fid_normalized_hr, fid_normalized_lr


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


def plot_fid_generation(fid_hr, fid_lr, config: ClusterConfig,
                        sample_idx, output_dir):
    """Plot FID signals at high and low acquisition times."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t_hr = np.arange(len(fid_hr)) / config.SAMPLING_RATE
    t_lr = np.arange(len(fid_lr)) / config.SAMPLING_RATE

    axes[0].plot(t_hr * 1000, fid_hr, 'r-', linewidth=1.5, label='Full Acquisition')
    axes[0].set_xlabel('Time [ms]', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'Sample {sample_idx}: Full Acquisition ({len(fid_hr)} points)',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].plot(t_lr * 1000, fid_lr, 'b-', linewidth=1.5, label='Truncated Acquisition')
    axes[1].set_xlabel('Time [ms]', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title(f'Truncated Acquisition ({len(fid_lr)} points)',
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fid_generation_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_apodization_and_noise(fid_windowed, fid_noisy, t,
                               sample_idx, output_dir):
    """Plot windowed signal and final noisy signal."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(t * 1000, fid_windowed, 'k-', linewidth=1.5, label='Windowed FID (before noise)')
    axes[0].set_xlabel('Time [ms]', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'Sample {sample_idx}: Windowed FID (constant noise applied)',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].plot(t * 1000, fid_noisy, 'purple', linewidth=1.5, label='After Noise Addition')
    axes[1].set_xlabel('Time [ms]', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title('After Constant Noise Addition (same noise for all clusters)',
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'apodization_noise_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_zero_filling_and_fft(fid, fid_padded, fft_mag, freq_axis,
                               sample_idx, output_dir):
    """Plot zero-filling and FFT transformation."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t_original = np.arange(len(fid))
    t_padded = np.arange(len(fid_padded))

    axes[0].plot(t_original, fid, 'b-', linewidth=1.5, label='Original FID')
    axes[0].plot(t_padded, fid_padded, 'r--', linewidth=1.0, alpha=0.7, label='Zero-Filled FID')
    axes[0].set_xlabel('Sample Index', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'Sample {sample_idx}: Zero-Filling (Time Domain)',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].plot(freq_axis / 1000, fft_mag, 'g-', linewidth=1.5, label='FFT Magnitude')
    axes[1].set_xlabel('Frequency [kHz]', fontsize=11)
    axes[1].set_ylabel('Magnitude', fontsize=11)
    axes[1].set_title('Frequency Domain: FFT Magnitude Spectrum',
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'zero_fill_fft_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_low_high_res_comparison(fft_low, fft_hr, freq_axis,
                                  sample_idx, output_dir):
    """Plot low-resolution vs high-resolution FFT comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    fft_low_norm = (fft_low - fft_low.min()) / (fft_low.max() - fft_low.min() + 1e-8)
    fft_hr_norm = (fft_hr - fft_hr.min()) / (fft_hr.max() - fft_hr.min() + 1e-8)

    ax.plot(freq_axis / 1000, fft_low_norm, 'b-', linewidth=1.5, alpha=0.7, label='Low-Res (truncated)')
    ax.plot(freq_axis / 1000, fft_hr_norm, 'r-', linewidth=1.5, alpha=0.7, label='High-Res (full)')

    ax.set_xlabel('Frequency [kHz]', fontsize=12)
    ax.set_ylabel('Magnitude (normalized)', fontsize=12)
    ax.set_title(f'Sample {sample_idx}: Low-Res vs High-Res Comparison',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'low_high_res_comparison_{sample_idx:04d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pipeline_summary(sample_data, output_dir):
    """Create a 4-panel summary showing the complete pipeline."""
    n_samples = len(sample_data)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Pipeline Summary: Constant Noise + Correct Abundance (prof_v2)',
                 fontsize=14, fontweight='bold')

    for i, data in enumerate(sample_data):
        fid_hr = data['fid_hr']
        fid_lr = data['fid_lr']
        fft_low = data['fft_lr']
        fft_hr = data['fft_hr']
        freq_axis = data['freq_axis']
        sample_idx = data['sample_id']

        t_hr = np.arange(len(fid_hr))

        ax = axes[i, 0]
        ax.plot(t_hr, fid_hr, 'r-', linewidth=1.5)
        ax.set_xlabel('Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Sample {sample_idx}: FID (Full)', fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        t_lr = np.arange(len(fid_lr))
        ax.plot(t_lr, fid_lr, 'b-', linewidth=1.5)
        ax.set_xlabel('Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title('FID (Truncated)', fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 2]
        fft_low_norm = (fft_low - fft_low.min()) / (fft_low.max() - fft_low.min() + 1e-8)
        ax.plot(freq_axis / 1000, fft_low_norm, 'b-', linewidth=1.5)
        ax.set_xlabel('Frequency [kHz]', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.set_title('Low-Res FFT', fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 3]
        fft_hr_norm = (fft_hr - fft_hr.min()) / (fft_hr.max() - fft_hr.min() + 1e-8)
        ax.plot(freq_axis / 1000, fft_hr_norm, 'r-', linewidth=1.5)
        ax.set_xlabel('Frequency [kHz]', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.set_title('High-Res FFT', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pipeline_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_cluster_lmdb_data(config: ClusterConfig, max_compounds: int = None):
    """
    Generate isotope cluster LMDB data using prof_v2 method with:
    - Constant noise per compound
    - Cross-cluster abundance normalization (all normalized to M+0 scale)
    """
    print("\n" + "=" * 80)
    print("ISOTOPE CLUSTER LMDB DATA GENERATION (prof_v2 - CONSTANT NOISE + CORRECT ABUNDANCE)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Compounds file: {config.COMPOUNDS_FILE}")
    print(f"  Full acquisition points: {config.N_POINTS_FID}")
    print(f"  Truncated acquisition points: {config.lr_points()}")
    print(f"  FFT size (zero-filled): {config.fft_size()}")
    print(f"  Output FFT points: {config.fft_output_size()}")
    print(f"  Damping: {config.DAMPING_FINAL_AMP}")
    print(f"  Kaiser beta: {config.KAISER_BETA}")
    print(f"  Noise level: {config.NOISE_LEVEL}")
    print("=" * 80 + "\n")

    # Load compounds
    formulas = load_compounds_long(config.COMPOUNDS_FILE)
    if max_compounds:
        formulas = formulas[:max_compounds]
    print(f"Loaded {len(formulas)} compounds")

    # Create output directories
    os.makedirs(config.LMDB_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    # Create LMDB databases
    dbs = create_lmdb_databases(config.LMDB_DIR, map_size=200 * 1024**3)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Collect all cluster data
    all_clusters = []
    compound_assignments = []
    cluster_counter = 0

    # Samples for sanity plots
    plot_samples = []

    print("\n" + "=" * 80)
    print("GENERATING ISOTOPE CLUSTERS WITH CONSTANT NOISE + CORRECT ABUNDANCE")
    print("=" * 80)

    for compound_idx, formula in enumerate(formulas):
        # Get isotope clusters for this compound
        clusters = get_isotope_clusters(formula, coverage_prob=config.COVERAGE_PROB)

        # Sort clusters by mass
        cluster_masses = sorted(clusters.keys())

        # M+0 (minimum cluster_mass) is the monoisotopic cluster
        min_cluster_mass = cluster_masses[0]
        m0_top_prob = clusters[min_cluster_mass][0][1]  # top probability of M+0

        # ============================================================
        # STEP 1: Generate windowed signals for ALL clusters (no noise yet)
        # All normalized to M+0's top probability scale
        # ============================================================
        cluster_windows = {}

        global_max_amp = 0.0

        for cluster_idx, cluster_mass in enumerate(cluster_masses):
            peaks = clusters[cluster_mass]

            result = generate_cluster_fid_prof_v2_no_noise(
                cluster_mass, peaks, config, m0_prob=m0_top_prob
            )

            fid_windowed_hr, fid_windowed_lr, ref_amp_hr, ref_amp_lr, \
                fft_mag_hr, fft_mag_lr, freq_axis = result

            if fid_windowed_hr is None:
                continue

            # Track global max amplitude across all clusters
            global_max_amp = max(global_max_amp, ref_amp_hr)

            cluster_windows[cluster_mass] = {
                'fid_windowed_hr': fid_windowed_hr,
                'fid_windowed_lr': fid_windowed_lr,
                'ref_amp_hr': ref_amp_hr,
                'ref_amp_lr': ref_amp_lr,
                'fft_mag_hr': fft_mag_hr,
                'fft_mag_lr': fft_mag_lr,
                'freq_axis': freq_axis,
                'peaks': peaks,
                'fft_mag_hr_raw': fft_mag_hr,
                'fft_mag_lr_raw': fft_mag_lr,
            }

        # ============================================================
        # STEP 2: Generate noise based on global max amplitude
        # Noise is the same for ALL clusters from this compound
        # ============================================================
        noise_amplitude = config.NOISE_LEVEL * global_max_amp
        noise_hr = np.random.randn(config.N_POINTS_FID).astype(np.float32) * noise_amplitude
        noise_lr = np.random.randn(config.lr_points()).astype(np.float32) * noise_amplitude

        # ============================================================
        # STEP 3: Add noise and normalize for each cluster
        # ============================================================
        for cluster_idx, cluster_mass in enumerate(cluster_masses):
            if cluster_mass not in cluster_windows:
                continue

            cw = cluster_windows[cluster_mass]

            fid_normalized_hr, fid_normalized_lr = apply_constant_noise(
                cw['fid_windowed_hr'],
                cw['fid_windowed_lr'],
                cw['ref_amp_hr'],
                cw['ref_amp_lr'],
                noise_hr,
                noise_lr
            )

            # Compute FFT from normalized noisy signals
            fft_size = config.fft_size()
            fft_output_size = config.fft_output_size()

            padded_hr = np.zeros(fft_size, dtype=np.float32)
            padded_hr[:config.N_POINTS_FID] = fid_normalized_hr
            fft_complex_hr = np.fft.rfft(padded_hr)
            fft_mag_hr_final = np.abs(fft_complex_hr).astype(np.float32)[:fft_output_size]

            padded_lr = np.zeros(fft_size, dtype=np.float32)
            padded_lr[:len(fid_normalized_lr)] = fid_normalized_lr
            fft_complex_lr = np.fft.rfft(padded_lr)
            fft_mag_lr_final = np.abs(fft_complex_lr).astype(np.float32)[:fft_output_size]

            # Store cluster data
            sample = {
                'fid_hr': fid_normalized_hr,
                'fid_lr': fid_normalized_lr,
                'fft_hr': fft_mag_hr_final,
                'fft_lr': fft_mag_lr_final,
                'fft_hr_raw': cw['fft_mag_hr_raw'],
                'fft_lr_raw': cw['fft_mag_lr_raw'],
                'freq_axis': cw['freq_axis'],
                'cluster_mass': cluster_mass,
                'compound_formula': formula,
                'n_peaks': len(cw['peaks']),
                'sample_id': cluster_counter,
                'noise_amplitude': noise_amplitude,
                'peaks': cw['peaks'],
                'm0_prob': m0_top_prob,  # store for reference
            }

            all_clusters.append(sample)
            compound_assignments.append((compound_idx, formula))
            cluster_counter += 1

            # Collect samples for sanity plots (first 3 clusters from first compound)
            if len(plot_samples) < 3 and compound_idx == 0:
                plot_samples.append(sample)

            m_plus = cluster_mass - min_cluster_mass
            print(f"  Cluster generated: {cluster_counter:6d} "
                  f"(Formula: {formula}, M+{m_plus}, m0_prob={m0_top_prob:.6f})", end="\r")

    print(f"\nGenerated {len(all_clusters)} isotope clusters")

    # Create train/val/test splits
    print("\n" + "=" * 80)
    print("CREATING TRAIN/VAL/TEST SPLITS (STRATIFIED BY COMPOUND)")
    print("=" * 80)

    n_clusters = len(all_clusters)
    n_train = int(n_clusters * config.TRAIN_RATIO)
    n_val = int(n_clusters * config.VAL_RATIO)

    # Group cluster indices by compound
    compound_to_indices = defaultdict(list)
    for i, (c_idx, _) in enumerate(compound_assignments):
        compound_to_indices[c_idx].append(i)

    # Sort compounds by their first cluster index for deterministic ordering
    sorted_compound_ids = sorted(compound_to_indices.keys())
    ordered_compound_indices = []
    for c_idx in sorted_compound_ids:
        ordered_compound_indices.extend(compound_to_indices[c_idx])

    train_set = set(ordered_compound_indices[:n_train])
    val_set   = set(ordered_compound_indices[n_train:n_train + n_val])
    test_set  = set(ordered_compound_indices[n_train + n_val:])

    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # Store metadata
    fft_size = config.fft_size()
    fft_output_size = config.fft_output_size()
    lr_points = config.lr_points()

    metadata = {
        'sampling_rate': config.SAMPLING_RATE,
        'n_points_fid': config.N_POINTS_FID,
        'lr_points': lr_points,
        'fft_size': fft_size,
        'fft_output_size': fft_output_size,
        'zero_fill_factor': config.ZERO_FILL_FACTOR,
        'acquisition_reduction_factor': config.ACQUISITION_REDUCTION_FACTOR,
        'damping': config.DAMPING_FINAL_AMP,
        'kaiser_beta': config.KAISER_BETA,
        'noise_level': config.NOISE_LEVEL,
        'n_clusters': n_clusters,
        'n_train': len(train_set),
        'n_val': len(val_set),
        'n_test': len(test_set),
        'train_indices': sorted(train_set),
        'val_indices': sorted(val_set),
        'test_indices': sorted(test_set),
        'noise_mode': 'constant_per_compound',
        'abundance_mode': 'normalized_to_m0',
    }

    # Write to LMDB
    print("\n" + "=" * 80)
    print("WRITING TO LMDB")
    print("=" * 80)

    train_counter = 0
    val_counter = 0
    test_counter = 0

    for idx, sample in enumerate(all_clusters):
        full_key = f'sample_{idx:08d}'
        write_to_lmdb(dbs['full'], full_key, sample)

        if idx in train_set:
            train_key = f'sample_{train_counter:08d}'
            write_to_lmdb(dbs['train'], train_key, sample)
            train_counter += 1
        elif idx in val_set:
            val_key = f'sample_{val_counter:08d}'
            write_to_lmdb(dbs['val'], val_key, sample)
            val_counter += 1
        elif idx in test_set:
            test_key = f'sample_{test_counter:08d}'
            write_to_lmdb(dbs['test'], test_key, sample)
            test_counter += 1

        if (idx + 1) % config.BATCH_SIZE == 0:
            print(f"  Processed: {idx + 1:6d} / {n_clusters}", end="\r")

    print(f"\nSaved {n_clusters} samples to LMDB")

    # Write metadata
    print("\n" + "=" * 80)
    print("WRITING METADATA")
    print("=" * 80)

    for split in ['full', 'train', 'val', 'test']:
        write_to_lmdb(dbs[split], '__metadata__', metadata)
        write_to_lmdb(dbs[split], '__len__', {'n_samples': len(train_set) if split == 'train' else
                                               len(val_set) if split == 'val' else
                                               len(test_set) if split == 'test' else
                                               n_clusters})
        dbs[split].close()
        print(f"  {split}.lmdb: {metadata[f'n_{split}'] if split != 'full' else n_clusters} samples")

    # Save freq_axis separately
    np.save(os.path.join(config.LMDB_DIR, 'freq_axis.npy'), plot_samples[0]['freq_axis'])

    print(f"\nLMDB databases created in '{config.LMDB_DIR}'")

    # Generate sanity plots
    print("\n" + "=" * 80)
    print("GENERATING SANITY PLOTS")
    print("=" * 80)

    for i, sample in enumerate(plot_samples):
        min_cmass = cluster_masses[0]
        m_plus = sample['cluster_mass'] - min_cmass
        print(f"\nSample {i}: {sample['compound_formula']}, M+{m_plus} "
              f"({sample['n_peaks']} peaks, m0_prob={sample['m0_prob']:.6f})")

        plot_fid_generation(sample['fid_hr'], sample['fid_lr'], config, i, config.PLOT_DIR)

        t_hr = np.arange(len(sample['fid_hr'])) / config.SAMPLING_RATE
        plot_apodization_and_noise(sample['fid_hr'], sample['fid_hr'], t_hr, i, config.PLOT_DIR)

        fft_size = config.fft_size()
        fid_padded = np.zeros(fft_size, dtype=np.float32)
        fid_padded[:len(sample['fid_hr'])] = sample['fid_hr']
        plot_zero_filling_and_fft(sample['fid_hr'], fid_padded, sample['fft_hr'],
                                  sample['freq_axis'], i, config.PLOT_DIR)

        plot_low_high_res_comparison(sample['fft_lr'], sample['fft_hr'],
                                     sample['freq_axis'], i, config.PLOT_DIR)

    plot_pipeline_summary(plot_samples, config.PLOT_DIR)

    print("\n" + "=" * 80)
    print("DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {config.LMDB_DIR}")
    print("Files:")
    print("  - full.lmdb")
    print("  - train.lmdb")
    print("  - val.lmdb")
    print("  - test.lmdb")
    print("  - freq_axis.npy")

    return config.LMDB_DIR, metadata


def main():
    """Run the complete cluster LMDB data generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate isotope cluster LMDB data (prof_v2 - constant noise + correct abundance)"
    )
    parser.add_argument("--max-compounds", type=int, default=None,
                        help="Limit number of compounds to process (default: all)")
    args = parser.parse_args()

    config = ClusterConfig()

    # Generate data
    lmdb_dir, metadata = generate_cluster_lmdb_data(config, max_compounds=args.max_compounds)

    print(f"\nTotal clusters generated: {metadata['n_clusters']}")
    print("Done!")


if __name__ == "__main__":
    main()
