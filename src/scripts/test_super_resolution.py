#!/usr/bin/env python
"""
Leakage-safe inference script.
Detects all peaks and renders a per-sample plot + metric table.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

from src.config import NarrowbandConfig
from src.models import LSTMSeq2Seq
from src.training.dataset_lmdb import LMDBDataset


# ------------------------------------------------------------
# Metrics Logic
# ------------------------------------------------------------
def get_peak_metrics(signal):
    """
    Finds all peaks using the professor's method:
    1. Measure FWHM of the strongest peak
    2. Use half-FWHM as minimum width constraint
    3. Apply height and prominence thresholds (1% of global max)

    This naturally filters out narrow noise spikes while detecting real peaks.
    """
    signal = np.asarray(signal, dtype=np.float64)
    sig_max = signal.max()
    if sig_max <= 0:
        return [], []

    # Find the strongest peak and measure its FWHM
    global_peak = int(np.argmax(signal))
    w, _, _, _ = peak_widths(signal, [global_peak], rel_height=0.5)
    min_width = float(w[0] / 2.0)  # half-FWHM as minimum width

    # Use width + height + prominence thresholds to find all real peaks
    peaks, properties = find_peaks(
        signal,
        height=sig_max / 100.0,
        width=min_width,
        rel_height=0.5,
        prominence=sig_max / 100.0,
    )

    if len(peaks) == 0:
        return [], []

    widths, _, _, _ = peak_widths(signal, peaks, rel_height=0.5)
    return peaks.tolist(), widths.tolist()


# ------------------------------------------------------------
# Plot and Table Rendering (Separate Files)
# ------------------------------------------------------------
def save_waveform_plot(x, y, pred, freq_axis, save_path, window_size=None):
    """
    Save waveform plot to file. Plots against physical frequency axis when available.
    If window_size is provided, centers the plot around the tallest peak.

    Args:
        window_size: If specified, shows this many bins centered on the main peak
                    (e.g., 128 = 64 left + 64 right of peak)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Trim x to the length of y/pred (they may be padded)
    n = len(y)
    x_plot = x[:n] if len(x) >= n else x

    # Use frequency axis if provided, otherwise bin indices
    if freq_axis is not None and len(freq_axis) == len(y):
        xaxis = freq_axis
        xlabel = "Frequency [Hz]"
    else:
        xaxis = np.arange(len(y))
        xlabel = "Bin"

    # Apply windowing if requested
    if window_size is not None and len(y) > 0:
        # Find the tallest peak in ground truth
        center_idx = np.argmax(y)
        half_window = window_size // 2

        # Calculate bounds
        lo = max(0, center_idx - half_window)
        hi = min(len(y), center_idx + half_window)

        # Slice data
        xaxis = xaxis[lo:hi]
        x_plot = x_plot[lo:hi]
        y = y[lo:hi]
        pred = pred[lo:hi]

    ax.plot(xaxis, x_plot, label="Input (Low-Res)", linewidth=1,
            alpha=0.6, color='gray')
    ax.plot(xaxis, y,    label="Ground Truth (High-Res)", linewidth=1.2,
            color='blue')
    ax.plot(xaxis, pred, label="Prediction", linewidth=1.2,
            color='red', linestyle='--')

    if len(y) > 0:
        ax.axvline(xaxis[np.argmax(y)], linestyle=":", alpha=0.3,
                   color='black')

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Spectral Analysis: {os.path.basename(save_path)}")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_peaks_table(gt_peaks, gt_fwhms, pr_peaks, pr_fwhms, freq_axis, save_path):
    """
    Save peaks table to file.
    """
    fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * max(len(gt_peaks), len(pr_peaks)))))
    ax.axis('off')

    num_peaks = max(len(gt_peaks), len(pr_peaks))
    table_data = []

    if num_peaks == 0:
        table_data.append(["N/A", "No Peaks Detected", "-", "-", "-"])
    else:
        for r in range(num_peaks):
            # Convert bin index → frequency when possible
            def bin_to_freq(bin_idx):
                if freq_axis is not None and bin_idx < len(freq_axis):
                    return f"{freq_axis[bin_idx]:.2f} Hz"
                return str(bin_idx)

            g_p = bin_to_freq(gt_peaks[r]) if r < len(gt_peaks) else "-"
            p_p = bin_to_freq(pr_peaks[r]) if r < len(pr_peaks) else "-"
            g_w = f"{gt_fwhms[r]:.2f}" if r < len(gt_fwhms) else "-"
            p_w = f"{pr_fwhms[r]:.2f}" if r < len(pr_fwhms) else "-"
            table_data.append([f"Peak {r + 1}", g_p, p_p, g_w, p_w])

    columns = ("ID", "GT Position", "Pred Position", "GT FWHM (bins)",
               "Pred FWHM (bins)")

    the_table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test super-resolution model on LMDB dataset"
    )
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Window size for plotting (e.g., 128 = 64 left + 64 right of peak). "
             "If not specified, train_plots full spectrum."
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of samples to process (default: 50)"
    )
    args = parser.parse_args()

    config = NarrowbandConfig()
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test split directly from the pre-split LMDB database
    test_ds = LMDBDataset(
        lmdb_path=config.LMDB_DIR,
        split='test',
        normalize=True,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0)

    # Load Model
    model = LSTMSeq2Seq(
        in_dim=1,
        enc_hidden=config.ENC_HIDDEN,
        enc_layers=config.ENC_LAYERS,
        dec_hidden=config.DEC_HIDDEN,
        dec_layers=config.DEC_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        use_attn_bridge=config.USE_ATTN_BRIDGE,
        attn_heads=config.ATTN_HEADS,
        attn_layers=config.ATTN_LAYERS,
    ).to(device)

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Inference Loop
    os.makedirs("test_plots/hr/waveforms", exist_ok=True)
    os.makedirs("test_plots/hr/tables", exist_ok=True)
    n_samples = args.n_samples

    print(f"\nProcessing {n_samples} samples and generating train_plots with tables...")
    if args.window_size is not None:
        print(f"  Window size: {args.window_size} bins (centered on main peak)")

    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            if i >= n_samples:
                break

            xb = xb.to(device).float()
            yb = yb.to(device).float()
            pred = model(xb)

            x_np   = xb.cpu().numpy()[0, :, 0]
            y_np   = yb.cpu().numpy()[0, :, 0]
            pred_np = pred.cpu().numpy()[0, :, 0]

            # Retrieve frequency axis for this sample from LMDB
            freq_axis = None
            try:
                with test_ds.env.begin() as txn:
                    key = f'sample_{i:08d}'.encode('ascii')
                    raw = txn.get(key)
                    if raw is not None:
                        sample_data = pickle.loads(raw)
                        fa = sample_data.get('freq_axis', None)
                        if fa is not None:
                            freq_axis = np.asarray(fa, dtype=np.float32)
            except Exception:
                pass

            # Extract metrics per-peak
            gt_peaks, gt_fwhms = get_peak_metrics(y_np)
            pr_peaks, pr_fwhms = get_peak_metrics(pred_np)

            # Save waveform plot and table separately
            wave_path = f"test_plots/hr/waveforms/sample_{i:03d}.png"
            table_path = f"test_plots/hr/tables/sample_{i:03d}.png"
            save_waveform_plot(x_np, y_np, pred_np, freq_axis, wave_path,
                               window_size=args.window_size)
            save_peaks_table(gt_peaks, gt_fwhms, pr_peaks, pr_fwhms, freq_axis,
                             table_path)

            if i % 10 == 0:
                print(f"  > Progress: {i}/{n_samples}")

    print(f"\n✅ Done. Check 'test_plots/hr/waveforms/' for train_plots and 'test_plots/hr/tables/' for peak tables.")


if __name__ == "__main__":
    main()
