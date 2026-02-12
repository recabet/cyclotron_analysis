"""
Mass spectrum visualization with matplotlib (static plots).
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from src.io.hdf5_readers import load_spectrum_compound


def plot_fft_comparison(
        fft_file: str,
        formulas: List[str],
        index: int = 1,
        theoretical_freq: Optional[np.ndarray] = None,
        theoretical_amp: Optional[np.ndarray] = None,
        max_points: int = 20000,
        save_path: Optional[str] = None,
) -> None:
    """
    Static matplotlib plot of FFT magnitude for three resolutions plus theoretical peaks.

    Args:
        fft_file: Path to FFT HDF5 file
        formulas: List of compound formulas
        index: Compound index to plot
        theoretical_freq: Theoretical peak frequencies [Hz]
        theoretical_amp: Theoretical peak amplitudes
        max_points: Maximum points to display (downsampled for performance)
        save_path: Path to save PNG figure (default: ../figures/fft_comparison_{index}.png)
    """
    # Load spectrum data
    freq, hr, mid, low = load_spectrum_compound(fft_file, index)

    # Downsample if needed for performance
    if len(freq) > max_points:
        step = len(freq) // max_points
        freq = freq[::step]
        hr = hr[::step]
        mid = mid[::step]
        low = low[::step]
        print(f"   Downsampled to {len(freq):,} points for plotting")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot three resolutions
    ax.plot(freq, hr, linewidth=1.5, color='red', alpha=0.85, label='High resolution')
    ax.plot(freq, mid, linewidth=1.5, color='green', alpha=0.85, label='Medium resolution')
    ax.plot(freq, low, linewidth=1.5, color='blue', alpha=0.85, label='Low resolution')

    # Theoretical peaks as markers with stems
    if theoretical_freq is not None and theoretical_amp is not None:
        # Scatter plot for peaks
        ax.scatter(
            theoretical_freq,
            theoretical_amp,
            marker='*',
            s=250,
            color='gold',
            edgecolors='black',
            linewidths=1.5,
            label='Theoretical peaks',
            zorder=5
        )

        # Vertical lines (stems) from baseline to peaks
        for xv, yv in zip(theoretical_freq, theoretical_amp):
            ax.vlines(
                xv,
                0,
                yv,
                colors='gray',
                linestyles='dotted',
                linewidth=1.2,
                alpha=0.5
            )

    # Styling
    ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Magnitude', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Mass Spectrum Comparison – {formulas[index]}',
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc='upper right',
        framealpha=0.95,
        edgecolor='gray',
        fontsize=11,
        shadow=True
    )

    # Y-axis limits
    ax.set_ylim(-0.05, 1.1)

    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', length=3)

    # Tight layout
    plt.tight_layout()

    # Save figure
    if save_path is None:
        save_path = f"../figures/fft_comparison_{index}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   Figure saved to {save_path}")

    # Close to free memory
    plt.close(fig)