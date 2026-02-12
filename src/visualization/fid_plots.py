"""
FID visualization with matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from src.io.hdf5_readers import load_fid_compound


def plot_fid_example(
        fid_file: str,
        formulas: List[str],
        index: int = 1,
        sampling_rate: float = 1e6,
        zoom_us: float = 1000.0,
        save_path: Optional[str] = None,
) -> None:
    """
    Plot FID signals at all three resolutions for one compound.

    Args:
        fid_file: Path to FID HDF5 file
        formulas: List of compound formulas (for title)
        index: Compound index to plot
        sampling_rate: Sampling rate in Hz
        zoom_us: Zoom window in microseconds
        save_path: If provided, save figure to this path
    """
    hr, mid, low = load_fid_compound(fid_file, index)

    # Time axes
    t_full = np.arange(len(hr)) / sampling_rate
    t_mid = np.arange(len(mid)) / sampling_rate
    t_low = np.arange(len(low)) / sampling_rate

    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=False)

    # Low resolution
    axs[0].plot(t_low, low, linewidth=0.7, color='blue')
    axs[0].set_title(f"Low Resolution FID (N={len(low)}) – {formulas[index]}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, alpha=0.3)

    # Medium resolution
    axs[1].plot(t_mid, mid, linewidth=0.7, color='green')
    axs[1].set_title(f"Medium Resolution FID (N={len(mid)}) – {formulas[index]}")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True, alpha=0.3)

    # High resolution (full)
    axs[2].plot(t_full, hr, linewidth=0.5, color='red')
    axs[2].set_title(f"High Resolution FID (N={len(hr)}) – {formulas[index]}")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Amplitude")
    axs[2].grid(True, alpha=0.3)

    # Zoom into first N microseconds
    zoom_limit = zoom_us * 1e-6
    idx_zoom = np.searchsorted(t_full, zoom_limit)
    axs[3].plot(t_full[:idx_zoom] * 1e6, hr[:idx_zoom], linewidth=0.8, color='purple')
    axs[3].set_title(f"FID – First {zoom_us:.0f} µs")
    axs[3].set_xlabel("Time (µs)")
    axs[3].set_ylabel("Amplitude")
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()