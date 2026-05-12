"""
FID (Free Induction Decay) signal generation.
"""
import numpy as np


def damping_envelope (t: np.ndarray, final_amplitude: float = 0.01) -> np.ndarray:
    """
    Exponential damping: amplitude decays to `final_amplitude` at t[-1].

    Args:
        t: Time vector [s]
        final_amplitude: Relative amplitude at the end of the FID (0-1)

    Returns:
        Damping envelope (same length as t), float32
    """
    decay = -np.log(final_amplitude) / t[-1]
    return np.exp(-decay * t).astype(np.float32)


def kaiser_window (n: int, beta: float) -> np.ndarray:
    """
    Create a Kaiser window for apodization.

    Args:
        n: Window length
        beta: Shape parameter (higher = more aggressive suppression)

    Returns:
        Kaiser window, float32
    """
    return np.kaiser(n, beta).astype(np.float32)


def add_noise (signal: np.ndarray, amplitudes: np.ndarray, max_amp: float, noise_level: float) -> np.ndarray:
    """
    Add Gaussian noise to a signal.

    Args:
        signal: Input FID signal
        amplitudes: Amplitudes used to scale noise (noise = noise_level * max(amplitude) * max_amp)
        max_amp: Global scaling factor for amplitudes
        noise_level: Noise std = noise_level * max(amplitude) * max_amp

    Returns:
        Signal with added noise
    """
    # amps = amplitudes * max_amp
    noise_std = noise_level * max_amp * np.sqrt(len(signal))
    noise = np.random.normal(0.0, noise_std, size=len(signal)).astype(np.float32)
    return signal + noise


def generate_fid (
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        t: np.ndarray,
        damping: np.ndarray,
        phase: float = 0.0,
        max_amp: float = 1.0,
        noise_level: float = 0.1,
        add_noise_after_window=None,
) -> np.ndarray:
    """
    Generate a single FID signal with damping.

    Args:
        frequencies: 1D array of cyclotron frequencies [Hz]
        amplitudes: 1D array of relative amplitudes (same length as frequencies)
        t: 1D time vector [s]
        damping: Pre‑computed damping envelope (same length as t)
        phase: Initial phase [rad]
        max_amp: Global scaling factor for amplitudes
        noise_level: Noise std = noise_level * max(amplitude) (kept for backward compatibility)
        add_noise_after_window: DEPRECATED - use add_noise() function instead

    Returns:
        FID signal (float32) of length len(t)
    """
    # Scale amplitudes
    amps = amplitudes * max_amp
    
    # Phase argument for each isotope: 2πft + φ
    # Shape: (n_isotopes, n_time_points)
    angular = 2 * np.pi * frequencies[:, None] * t[None, :] + phase
    
    # Coherent sum of sine waves
    signal = np.sum(amps[:, None] * np.sin(angular), axis=0, dtype=np.float32)
    
    # Apply exponential damping
    signal *= damping
    
    return signal
