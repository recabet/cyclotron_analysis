import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from src.models import LSTMSeq2Seq
from src.config import settings
from src.training import H5SpectraDataset
from src.training import metrics_np


config = settings.TrainingConfig()

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


# =========================================================
# TEST (FULL INFERENCE)
# =========================================================
def test(interval_size=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with h5py.File(config.H5_PATH, 'r') as f:
        N = len(f[config.X_KEY])
        seq_len = f[config.X_KEY].shape[1]

    print(f"Total samples: {N}")
    print(f"Sequence length: {seq_len}")
    print(f"Interval mode: {interval_size}")

    # Split
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_train = int(N * config.TRAIN_RATIO)
    test_idx = all_idx[n_train:]

    test_ds = H5SpectraDataset(
        config.H5_PATH,
        config.X_KEY,
        config.Y_KEY,
        interval_size=interval_size,
        indices=test_idx,
        normalize=False
    )

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
        attn_layers=config.ATTN_LAYERS
    ).to(device)

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            y_hat = model(xb)

            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)[:, :, 0]
    y_pred = np.concatenate(y_pred_list, axis=0)[:, :, 0]

    metrics = metrics_np(y_true, y_pred)

    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6e}" if isinstance(v, float) else f"{k}: {v}")

    # Save inference plots
    save_dir = "../figures/test_infer"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(y_true)):
        true_spec = y_true[i]
        pred_spec = y_pred[i]

        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axs[0].plot(true_spec)
        axs[0].set_title("Ground Truth")

        axs[1].plot(pred_spec)
        axs[1].set_title("Prediction")

        axs[2].plot(true_spec, label="True")
        axs[2].plot(pred_spec, label="Pred")
        axs[2].legend()
        axs[2].set_title("Overlay")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"infer_{i:04d}.png"), dpi=150)
        plt.close()

    print("✅ Inference plots saved.")


# =========================================================
# FULL VS INTERVAL VISUAL COMPARISON
# =========================================================
def compare_full_vs_interval(interval_size=1024):

    print("\n🔎 Comparing Full vs Interval")

    with h5py.File(config.H5_PATH, 'r') as f:
        N = len(f[config.X_KEY])

    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_train = int(N * config.TRAIN_RATIO)
    test_idx = all_idx[n_train:]

    # Full dataset
    full_ds = H5SpectraDataset(
        config.H5_PATH,
        config.X_KEY,
        config.Y_KEY,
        interval_size=None,
        indices=test_idx,
        normalize=False
    )

    # Interval dataset
    interval_ds = H5SpectraDataset(
        config.H5_PATH,
        config.X_KEY,
        config.Y_KEY,
        interval_size=interval_size,
        indices=test_idx,
        normalize=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 6))

    for idx in range(2):

        # -------- FULL --------
        x_full, y_full = full_ds[idx]
        y_full = y_full.numpy().squeeze()

        axes[0, idx].plot(y_full)
        axes[0, idx].set_title(f"Sample {idx} - Full (8192)")

        # -------- INTERVAL --------
        x_int, y_int = interval_ds[idx]
        y_int = y_int.numpy().squeeze()

        peak_idx = np.argmax(y_int)
        expected = len(y_int) // 2

        print(
            f"Sample {idx} interval peak index: {peak_idx} | expected: {expected}"
        )

        axes[1, idx].plot(y_int)
        axes[1, idx].axvline(expected, linestyle="--")
        axes[1, idx].set_title(
            f"Sample {idx} - Interval ({interval_size})"
        )

    plt.tight_layout()
    plt.savefig("spectra_comparison.png", dpi=150)
    plt.show()

    print("✅ Comparison plot saved as spectra_comparison.png")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # 1️⃣ Compare full 8192 vs centered interval
    compare_full_vs_interval(interval_size=256)

    # 2️⃣ Run full inference (change interval_size if needed)
    # test(interval_size=None)        # Full spectrum
    # test(interval_size=1024)        # Interval mode