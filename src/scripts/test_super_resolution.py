#!/usr/bin/env python
"""
Leakage-safe inference script.
Detects all peaks and renders a per-sample plot + metric table.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

from src.config import TrainingConfig
from src.models import LSTMSeq2Seq
from src.training import H5SpectraDataset


# ------------------------------------------------------------
# Metrics Logic
# ------------------------------------------------------------
def get_peak_metrics (signal):
    """
    Finds all peaks above threshold and calculates their positions and FWHM.
    """
    peaks, _ = find_peaks(signal, height=0.1, distance=10)
    if len(peaks) == 0:
        return [], []
    
    widths, _, _, _ = peak_widths(signal, peaks, rel_height=0.5)
    return peaks.tolist(), widths.tolist()


# ------------------------------------------------------------
# Combined Plot and Table Rendering
# ------------------------------------------------------------
def save_sample_results (x, y, pred, gt_peaks, gt_fwhms, pr_peaks, pr_fwhms, save_path):
    """
    Creates a figure with the spectrum plot on top and a data table on the bottom.
    """
    fig, (ax_plot, ax_table) = plt.subplots(2, 1, figsize=(12, 8),
                                            gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Plot the Spectra
    ax_plot.plot(x, label="Input (Low-Res)", linewidth=1, alpha=0.6, color='gray')
    ax_plot.plot(y, label="Ground Truth (High-Res)", linewidth=1.2, color='blue')
    ax_plot.plot(pred, label="Prediction", linewidth=1.2, color='red', linestyle='--')
    
    # Highlight the main peak for reference
    if len(y) > 0:
        ax_plot.axvline(np.argmax(y), linestyle=":", alpha=0.3, color='black')
    
    ax_plot.set_title(f"Spectral Analysis: {os.path.basename(save_path)}")
    ax_plot.legend(loc='upper right')
    ax_plot.grid(alpha=0.2)
    
    # 2. Prepare and Render Table
    ax_table.axis('off')
    
    num_peaks = max(len(gt_peaks), len(pr_peaks))
    table_data = []
    
    if num_peaks == 0:
        table_data.append(["N/A", "No Peaks Detected", "-", "-", "-"])
    else:
        for r in range(num_peaks):
            g_p = gt_peaks[r] if r < len(gt_peaks) else "-"
            p_p = pr_peaks[r] if r < len(pr_peaks) else "-"
            g_w = f"{gt_fwhms[r]:.2f}" if r < len(gt_fwhms) else "-"
            p_w = f"{pr_fwhms[r]:.2f}" if r < len(pr_fwhms) else "-"
            table_data.append([f"Peak {r + 1}", g_p, p_p, g_w, p_w])
    
    columns = ("ID", "GT Position", "Pred Position", "GT FWHM", "Pred FWHM")
    
    the_table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main ():
    config = TrainingConfig()
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Splitting Logic (Matches Training)
    with h5py.File(config.TEST_H5_PATH, "r") as f:
        N = len(f[config.X_KEY])
    
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_train, n_val = int(N * config.TRAIN_RATIO), int(N * config.VAL_RATIO)
    test_idx = all_idx[n_train + n_val:]
    
    test_ds = H5SpectraDataset(
        config.TEST_H5_PATH, config.X_KEY, config.Y_KEY,
        indices=test_idx, centered=True, normalize=True, interval_size=512,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Load Model
    model = LSTMSeq2Seq(
        in_dim=1, enc_hidden=config.ENC_HIDDEN, enc_layers=config.ENC_LAYERS,
        dec_hidden=config.DEC_HIDDEN, dec_layers=config.DEC_LAYERS,
        dropout=config.DROPOUT, bidirectional=config.BIDIRECTIONAL,
        use_attn_bridge=config.USE_ATTN_BRIDGE, attn_heads=config.ATTN_HEADS,
        attn_layers=config.ATTN_LAYERS,
    ).to(device)
    
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # Inference Loop
    os.makedirs("test_plots/hr", exist_ok=True)
    n_samples = 50
    
    print(f"\nProcessing {n_samples} samples and generating plots with tables...")
    
    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            if i >= n_samples:
                break
            
            xb, yb = xb.to(device).float(), yb.to(device).float()
            pred = model(xb)
            
            x_np = xb.cpu().numpy()[0, :, 0]
            y_np = yb.cpu().numpy()[0, :, 0]
            pred_np = pred.cpu().numpy()[0, :, 0]
            
            # Extract metrics
            gt_peaks, gt_fwhms = get_peak_metrics(y_np)
            pr_peaks, pr_fwhms = get_peak_metrics(pred_np)
            
            # Save combined Plot + Table
            save_path = f"test_plots/hr/sample_{i:03d}.png"
            save_sample_results(x_np, y_np, pred_np, gt_peaks, gt_fwhms, pr_peaks, pr_fwhms, save_path)
            
            if i % 10 == 0:
                print(f"  > Progress: {i}/{n_samples}")
    
    print(f"\n✅ Done. Check 'test_plots/opt/' for the combined PNG files.")


if __name__ == "__main__":
    main()