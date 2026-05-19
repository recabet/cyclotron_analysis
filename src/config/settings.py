"""
Global simulation parameters.
All configuration in one place - modify these values to change the simulation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation parameters."""
    # ----------------------------------------
    # Physical constants
    # ----------------------------------------
    AVOGADRO: float = 6.02214076e23  # mol⁻¹
    ELECTRON_CHARGE: float = 1.602176634e-19  # C
    ION_CHARGE: int = 1  # elementary charges
    MAGNETIC_FIELD: float = 8.0  # Tesla

    # ----------------------------------------
    # Isotope distribution (IsoSpec)
    # ----------------------------------------
    COVERAGE_PROB: float = 0.999  # Fraction of probability space to cover
    COMPOUNDS_FILE: str = "data/compounds/compounds_660000hz_at_10T_long.txt"

    # ----------------------------------------
    # Signal acquisition parameters
    # ----------------------------------------
    SAMPLING_RATE: float = 1e6  # Hz
    N_POINTS_FID: int = 65536  # 8192 points
    ZERO_FILL_FACTOR: int = 2  # Zero-padding for FFT (2 = double)
    DAMPING_FINAL_AMP: float = 0.01  # Relative amplitude at end of FID

    # ----------------------------------------
    # Resolution reduction
    # ----------------------------------------
    MID_RES_FACTOR: int = 2  # Divide N_POINTS_FID by this
    LOW_RES_FACTOR: int = 8  # Divide N_POINTS_FID by this

    # ----------------------------------------
    # Windowing
    # ----------------------------------------
    KAISER_BETA: float = 10.0  # Kaiser window shape parameter

    # ----------------------------------------
    # Noise and amplitude
    # ----------------------------------------
    MAX_AMPLITUDE: float = 1.0  # Global scaling for peak heights
    NOISE_LEVEL: float = 0.1  # Noise std = NOISE_LEVEL * max(amplitude)

    # ----------------------------------------
    # Batch processing (memory management)
    # ----------------------------------------
    BATCH_SIZE: int = 4000  # Number of compounds to process at once

    # ----------------------------------------
    # HDF5 compression
    # ----------------------------------------
    COMPRESSION: str = "lzf"  # For backward compatibility (mirrors COMPRESSION_ALGORITHM)
    COMPRESSION_ALGORITHM: str = "lzf"  # 'lzf' (faster) or 'gzip' (smaller)
    COMPRESSION_LEVEL: int = 0  # Not used for lzf

    # ----------------------------------------
    # Narrowband (frequency-shift) approach
    # ----------------------------------------
    CLUSTER_TOLERANCE_HZ: float = 250.0  # Max gap to keep peaks in same cluster [Hz]
                                         # Peaks separated by >250 Hz are separate samples
                                         # (Isotope peaks are ~246 Hz apart, fine structure ~0.1-1 Hz)
                                         # Set to ~250 Hz to preserve complete isotopic clusters
    FREQUENCY_RESOLUTION_TARGET: float = 2.0  # Target resolution per bin [Hz] (was 1.0, changed to 4x reduce file size)
    CLUSTER_BATCH_SIZE: int = 5000  # Batch size for narrowband cluster writing (was 1000)

    # ----------------------------------------
    # Output file names
    # ----------------------------------------
    FID_H5: str = "data/waves/fid/fid_levels.h5"
    FFT_H5: str = "data/waves/fft/fft_full_spectra.h5"
    SEGMENTS_H5: str = "data/waves/segments/training_segments_{n_high}.h5"
    NARROWBAND_H5: str = "data/waves/narrowband/narrowband_clusters_v2_optimized.h5"  # new: 33k samples, 264MB

    # LMDB directories
    LMDB_DIR: str = "data/lmdb/narrowband"  # LMDB database directory for narrowband data
    TRAIN_LMDB_DIR: str = "data/lmdb/training_segments"  # LMDB database directory for full-spectrum training data

    # ----------------------------------------
    # Peak segmentation parameters
    # ----------------------------------------
    PEAK_K_SIGMA: float = 3.0  # Threshold = baseline + k_sigma * noise_std
    PEAK_MIN_FLOOR: float = 0.01  # Minimum absolute threshold
    PEAK_SMOOTH_WIN: int = 7  # Moving average window for noise estimation
    PEAK_PAD_MARGIN: int = 10  # Extra bins to keep on each side
    PEAK_GAP_MERGE: int = 5  # Merge peaks separated by ≤ this many bins

    # ----------------------------------------
    # Outlier filtering (IQR)
    # ----------------------------------------
    IQR_K: float = 1.5  # IQR multiplier for outlier detection


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for isotope cluster super-resolution using prof_v2 method."""
    # ----------------------------------------
    # Signal acquisition - high res (full acquisition)
    # ----------------------------------------
    N_POINTS_FID: int = 2048  # Full acquisition time points
    N_POINTS_FID_V2: int = 512  # Full acquisition time points
    N_POINTS_FID_LR:int=64
    SAMPLING_RATE: float = 1e6  # Hz

    # ----------------------------------------
    # Signal acquisition - low res (truncated acquisition)
    # ----------------------------------------
    ACQUISITION_REDUCTION_FACTOR: int = 16  # Low-res uses 1/N of full points (8x = 256 pts instead of 2048)

    # ----------------------------------------
    # Zero-filling and FFT
    # ----------------------------------------
    ZERO_FILL_FACTOR: int = 2  # Zero-pad to FFT_SIZE * 2

    # Computed values (access as properties or methods)
    def fft_size(self) -> int:
        return self.N_POINTS_FID * self.ZERO_FILL_FACTOR

    def fft_output_size(self) -> int:
        return self.N_POINTS_FID  # 2048 output points

    def lr_points(self) -> int:
        return self.N_POINTS_FID // self.ACQUISITION_REDUCTION_FACTOR

    # ----------------------------------------
    # Windowing and damping (prof_v2 defaults)
    # ----------------------------------------
    KAISER_BETA: float = 5.0  # From professor's original
    DAMPING_FINAL_AMP: float = 0.005  # Light damping preserves fine structure

    # ----------------------------------------
    # Noise
    # ----------------------------------------
    NOISE_LEVEL: float = 0.01  # Noise std relative to max amplitude

    # ----------------------------------------
    # Isotope distribution (IsoSpec)
    # ----------------------------------------
    COVERAGE_PROB: float = 0.999  # Fraction of probability space to cover
    COMPOUNDS_FILE: str = "data/compounds/compounds_660000hz_at_10T_long.txt"

    # ----------------------------------------
    # Output
    # ----------------------------------------
    LMDB_DIR: str = "data/lmdb/cluster_prof_v2_fix"
    PLOT_DIR: str = "figures/cluster_prof_v2"
    MAX_AMPLITUDE: float = 1.0  # Global scaling for peak heights

    # ----------------------------------------
    # Train/val/test split
    # ----------------------------------------
    TRAIN_RATIO: float = 0.75
    VAL_RATIO: float = 0.20
    # TEST_RATIO: 1 - TRAIN_RATIO - VAL_RATIO

    # ----------------------------------------
    # Batch processing
    # ----------------------------------------
    BATCH_SIZE: int = 20000  # Compounds per batch


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and settings for super-resolution training."""
    SEED: int = 42
    BATCH_SIZE: int = 32  # Increased for better GPU utilization
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 200
    PATIENCE: int = 10  # Increased to avoid premature stopping

    # Model architecture
    ENC_HIDDEN: int = 128
    ENC_LAYERS: int = 2
    DEC_HIDDEN: Optional[int] = None  # if None, set to ENC_HIDDEN
    DEC_LAYERS: int = 2
    DROPOUT: float = 0.1
    BIDIRECTIONAL: bool = True
    USE_ATTN_BRIDGE: bool = True
    ATTN_HEADS: int = 8  # Reduced to increase head_dim from 8 to 16
    ATTN_LAYERS: int = 2

    # Loss function
    LOSS: str = "huber"  # "mse", "mae", "huber"
    HUBER_DELTA: float = 0.15  # Increased for normalized spectra

    # Optimizer
    OPTIMIZER: str = "adamw"
    WEIGHT_DECAY: float = 1e-4
    BETAS: tuple = (0.9, 0.98)
    CLIP_NORM: float = 1.0

    # Training optimization
    USE_AMP: bool = True  # Enable automatic mixed precision
    NUM_WORKERS: int = 4  # Number of data loading workers
    PIN_MEMORY: bool = True  # Enable pinned memory for faster GPU transfer
    PRELOAD_DATA: bool = True  # Preload H5 data into memory to avoid I/O bottleneck

    # Scheduler / Warmup
    WARMUP_EPOCHS: int = 10  # Number of warmup epochs

    # Visualization
    PLOT_EVERY: int = 1
    N_PREVIEW_SAMPLES: int = 3  # Number of samples to preview

    # Data
    TRAIN_RATIO: float = 0.75
    VAL_RATIO: float = 0.20
    # TEST_RATIO is 1 - TRAIN_RATIO - VAL_RATIO

    # Paths
    H5_PATH: str = "data/waves/segments/training_segments_65536.h5"
    TEST_H5_PATH: str = "data/waves/test/training_segments_65536_test.h5"
    X_KEY: str = "fft_low"
    Y_KEY: str = "fft_hr"

    # Output
    MODEL_SAVE_PATH: str = "weights/best_lstm_zoomed_seq2seq_65536_hr.pt"


@dataclass(frozen=True)
class NarrowbandConfig(TrainingConfig):
    """Configuration for narrowband (frequency-shifted cluster) super-resolution training."""

    # Override data paths for narrowband HDF5 file
    H5_PATH: str = "data/waves/narrowband/narrowband_clusters_v2_optimized.h5"
    TEST_H5_PATH: str = "data/waves/narrowband/narrowband_clusters_test.h5"
    X_KEY: str = "fft_half"  # Low-resolution input
    Y_KEY: str = "fft_full"  # High-resolution target

    # LMDB path
    LMDB_DIR: str = "data/lmdb/narrowband"

    # Narrowband-specific: window size for training
    # Frequency resolution ~1.92 Hz/bin (at 2.0 Hz target), so 1024 pts ≈ 1966 Hz range
    # Captures full isotope cluster (typically < 700 Hz spread) with margin
    INTERVAL_SIZE: int = 1024

    # Batch size - increased for better GPU utilization
    BATCH_SIZE: int = 32  # Increased from 16 for 12GB GPU

    # Update model save path
    MODEL_SAVE_PATH: str = "weights/best_narrowband_lstm_seq2seq.pt"


@dataclass(frozen=True)
class ClusterTrainingConfig(TrainingConfig):
    """Configuration for cluster prof_v2 super-resolution training."""

    # Override data paths for cluster prof_v2 LMDB file
    H5_PATH: str = "data/lmdb/cluster_prof_v2_fix"
    LMDB_DIR: str = "data/lmdb/cluster_prof_v2_fix"

    # Input/Output keys
    X_KEY: str = "fft_lr"  # Low-resolution from truncated acquisition
    Y_KEY: str = "fft_hr"  # High-resolution from full acquisition

    # Model save path
    MODEL_SAVE_PATH: str = "weights/best_cluster_prof_v2_lstm_seq2seq.pt"

    # Batch size
    BATCH_SIZE: int = 64  # Larger batch for cluster data