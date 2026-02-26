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
    COVERAGE_PROB: float = 0.99  # Fraction of probability space to cover
    COMPOUNDS_FILE: str = "../data/compounds/compounds_660000hz_at_10T_long.txt"

    # ----------------------------------------
    # Signal acquisition parameters
    # ----------------------------------------
    SAMPLING_RATE: float = 1e6  # Hz
    N_POINTS_FID: int = 8192  # 8192 points
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
    BATCH_SIZE: int = 1000  # Number of compounds to process at once

    # ----------------------------------------
    # HDF5 compression
    # ----------------------------------------
    COMPRESSION: str = "lzf"  # 'lzf' (faster) or 'gzip' (smaller)

    # ----------------------------------------
    # Output file names
    # ----------------------------------------
    FID_H5: str = "data/waves/fid/fid_levels.h5"
    FFT_H5: str = "data/waves/fft/fft_full_spectra.h5"
    SEGMENTS_H5: str = "data/waves/segments/training_segments_{n_high}.h5"

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
class TrainingConfig:
    """Hyperparameters and settings for super-resolution training."""
    SEED: int = 42
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 100
    PATIENCE: int = 5

    # Model architecture
    ENC_HIDDEN: int = 128
    ENC_LAYERS: int = 2
    DEC_HIDDEN: Optional[int] = None  # if None, set to ENC_HIDDEN
    DEC_LAYERS: int = 2
    DROPOUT: float = 0.1
    BIDIRECTIONAL: bool = True
    USE_ATTN_BRIDGE: bool = True
    ATTN_HEADS: int = 16
    ATTN_LAYERS: int = 2

    # Loss function
    LOSS: str = "huber"  # "mse", "mae", "huber"
    HUBER_DELTA: float = 0.05

    # Optimizer
    OPTIMIZER: str = "adamw"
    WEIGHT_DECAY: float = 1e-4
    BETAS: tuple = (0.9, 0.98)
    CLIP_NORM: float = 1.0

    # Data
    TRAIN_RATIO: float = 0.75
    VAL_RATIO: float = 0.20
    # TEST_RATIO is 1 - TRAIN_RATIO - VAL_RATIO

    # Paths
    H5_PATH: str = "data/waves/segments/training_segments_8192.h5"
    X_KEY: str = "fft_low"
    Y_KEY: str = "fft_hr"

    # Output
    MODEL_SAVE_PATH: str = "weights/best_lstm_seq2seq.pt"