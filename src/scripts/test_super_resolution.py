#!/usr/bin/env python
"""
Leakage-safe inference script.

Uses EXACT same dataset split logic as training script.
Only runs on the held-out test set.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import TrainingConfig
from src.models import LSTMSeq2Seq
from src.training import H5SpectraDataset


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def save_overlay_plot(x, y, pred, save_path, zoom_ratio=1.0):
    seq_len = len(y)

    # 🔥 Find actual peak location
    peak_idx = np.argmax(y)

    zoom_size = int(seq_len * zoom_ratio)

    start = max(peak_idx - zoom_size // 2, 0)
    end = min(peak_idx + zoom_size // 2, seq_len)

    plt.figure(figsize=(12, 5))
    plt.plot(x, label="Input (Low-Res)", linewidth=1, alpha=0.8)
    plt.plot(y, label="Ground Truth (High-Res)", linewidth=1)
    plt.plot(pred, label="Prediction", linewidth=1)

    plt.xlim(start, end)

    # mark peak
    plt.axvline(peak_idx, linestyle="--", alpha=0.5)

    plt.legend()
    plt.title("Input vs Ground Truth vs Prediction (Peak Zoom)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    config = TrainingConfig()

    # Same seed as training (CRITICAL)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Recreate EXACT SAME SPLIT as training
    # ------------------------------------------------------------
    print("\nRecreating dataset split (no leakage)...")

    with h5py.File(config.TEST_H5_PATH, "r") as f:
        N = len(f[config.X_KEY])

    all_idx = np.arange(N)
    np.random.shuffle(all_idx)

    n_train = int(N * config.TRAIN_RATIO)
    n_val = int(N * config.VAL_RATIO)

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # ------------------------------------------------------------
    # Create TEST dataset ONLY
    # ------------------------------------------------------------
    interval_size = 256   # must match training

    test_ds = H5SpectraDataset(
        config.TEST_H5_PATH,
        config.X_KEY,
        config.Y_KEY,
        indices=test_idx,   # ✅ CRITICAL FIX
        centered=True,
        normalize=True,
        interval_size=interval_size,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(test_ds)}")
    print(f"Interval size: {interval_size}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
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

    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from: {config.MODEL_SAVE_PATH}")

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    os.makedirs("test_plots", exist_ok=True)
    os.makedirs("test_plots/quarter",exist_ok=True)
    n_samples = 50

    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            if i >= n_samples:
                break

            xb = xb.to(device).float()
            yb = yb.to(device).float()

            pred = model(xb)

            x_np = xb.cpu().numpy()[0, :, 0]
            y_np = yb.cpu().numpy()[0, :, 0]
            pred_np = pred.cpu().numpy()[0, :, 0]

            save_path = f"test_plots/quarter/sample_quarter{i:03d}.png"

            save_overlay_plot(x_np, y_np, pred_np, save_path)

            print(f"Saved: {save_path}")

    print("\n✅ Inference complete. No data leakage.")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()