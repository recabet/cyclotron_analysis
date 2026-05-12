"""
Diagnostic train_plots for quality control of generated data.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_segment_lengths_boxplot(
        lengths: np.ndarray,
        lower: float,
        upper: float,
        keep_mask: np.ndarray,
        save_path: Optional[str] = None,
) -> None:
    """
    Boxplot of segment lengths with IQR thresholds and outliers highlighted.

    Args:
        lengths: Array of segment lengths (bins)
        lower: Lower IQR threshold
        upper: Upper IQR threshold
        keep_mask: Boolean array, True for non-outliers
        save_path: If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 3))

    # Boxplot without fliers (we'll add them separately)
    bp = ax.boxplot(
        [lengths],
        vert=False,
        showfliers=False,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#7aa6f0", alpha=0.35, edgecolor="#3b5ba7"),
        medianprops=dict(color="#0f1311", linewidth=1.5),
        whiskerprops=dict(color="#3b5ba7", linewidth=1),
        capprops=dict(color="#3b5ba7", linewidth=1),
        showmeans=True,
        meanline=True,
        meanprops=dict(color="#2e7d32", linewidth=1.5, linestyle='--'),
    )

    # Outliers as scatter points with jitter
    outliers = lengths[~keep_mask]
    if outliers.size > 0:
        ax.scatter(
            outliers,
            np.ones_like(outliers) * 1.0,  # y-position
            s=30,
            facecolors="#d81b1b40",
            edgecolors="#8b0000",
            linewidths=0.5,
            zorder=3,
            label=f'Outliers (n={outliers.size})',
        )

    # Threshold lines (with labels for legend)
    ax.axvline(
        lower,
        ls="--",
        lw=1.2,
        color="#030303",
        alpha=0.7,
        label='IQR Bounds'
    )
    ax.axvline(
        upper,
        ls="--",
        lw=1.2,
        color="#030303",
        alpha=0.7
    )

    # Annotations
    ax.text(
        lower, 1.35, f"Lower: {int(lower)}",
        ha="center", va="bottom", fontsize=10, color="#080808"
    )
    ax.text(
        upper, 1.35, f"Upper: {int(upper)}",
        ha="center", va="bottom", fontsize=10, color="#080808"
    )

    # Summary statistics
    n_keep = keep_mask.sum()
    n_total = len(lengths)
    mean_kept = lengths[keep_mask].mean()
    median_kept = np.median(lengths[keep_mask])

    ax.set_title(
        f"Segment Length Distribution\n"
        f"Kept: {n_keep}/{n_total} ({n_keep / n_total:.1%}) | "
        f"Mean: {mean_kept:.1f} bins | Median: {median_kept:.1f} bins",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel("Segment Length (frequency bins)", fontsize=11)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)

    # Only show legend if there are outliers
    if outliers.size > 0:
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_sample_segments(
        segments: np.ndarray,
        compounds: List[str],
        n_samples: int = 5,
        title: str = "Cropped Spectrum Segments",
        save_path: Optional[str] = None,
) -> None:
    """
    Plot a few examples of cropped spectrum segments.

    Args:
        segments: 2D array of cropped spectra (n_compounds, segment_length)
        compounds: List of compound formulas
        n_samples: Number of random samples to plot
        title: Plot title
        save_path: If provided, save figure to this path
    """
    n_available = min(n_samples, len(segments))
    indices = np.random.choice(len(segments), n_available, replace=False)

    fig, axes = plt.subplots(n_available, 1, figsize=(10, 3 * n_available))
    if n_available == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.plot(segments[idx], linewidth=1.0, color='#1f77b4')
        ax.set_title(f"{compounds[idx]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Bins (re-indexed from peak start)", fontsize=10)
        ax.set_ylabel("Intensity", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Add segment statistics as text
        segment = segments[idx]
        max_val = segment.max()
        max_pos = segment.argmax()
        ax.text(
            0.98, 0.95,
            f'Max: {max_val:.3f} @ bin {max_pos}\nLength: {len(segment)} bins',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()