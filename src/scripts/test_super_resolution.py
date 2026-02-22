
import h5py
import numpy as np
import torch
from  torch.utils.data import DataLoader
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

def test():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with h5py.File(config.H5_PATH, 'r') as f:
        N = len(f[config.X_KEY])
        seq_len = f[config.X_KEY].shape[1]

    print(f"Total samples: {N}")
    print(f"Sequence length: {seq_len}")

    # Create indices
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_train = int(N * config.TRAIN_RATIO)
    train_idx = all_idx[:n_train]
    test_idx = all_idx[n_train:]

    test_ds = H5SpectraDataset(config.H5_PATH,
                              config.X_KEY,
                              config.Y_KEY,
                              indices=test_idx,
                              normalize=False)

    model=LSTMSeq2Seq(in_dim=1,
                      enc_hidden=config.ENC_HIDDEN,
                      enc_layers=config.ENC_LAYERS,
                      dec_hidden=config.DEC_HIDDEN,
                      dec_layers=config.DEC_LAYERS,
                      dropout=config.DROPOUT,
                      bidirectional=config.BIDIRECTIONAL,
                      use_attn_bridge=config.USE_ATTN_BRIDGE,
                      attn_heads=config.ATTN_HEADS,
                      attn_layers=config.ATTN_LAYERS).to(device)

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()

    test_loader = DataLoader(test_ds,
                             batch_size=config.BATCH_SIZE,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            yb = yb.to(non_blocking=True).float()
            xb = xb.to(device,non_blocking=True).float()
            y_hat = model(xb)
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)[:, :, 0]
    y_pred = np.concatenate(y_pred_list, axis=0)[:, :, 0]

    metrics = metrics_np(y_true, y_pred)
    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")



    save_dir = "../figures/test_infer"
    os.makedirs(save_dir, exist_ok=True)

    num_samples = y_true.shape[0]

    print(f"Saving {num_samples} inference plots to {save_dir}")

    for i in range(num_samples):
        true_spec = y_true[i]
        pred_spec = y_pred[i]

        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        # ---- True ----
        axs[0].plot(true_spec)
        axs[0].set_title("Ground Truth")
        axs[0].set_ylabel("Magnitude")

        # ---- Pred ----
        axs[1].plot(pred_spec)
        axs[1].set_title("Prediction")
        axs[1].set_ylabel("Magnitude")

        # ---- Overlay ----
        axs[2].plot(true_spec, label="True", alpha=0.8)
        axs[2].plot(pred_spec, label="Predicted", alpha=0.8)
        axs[2].set_title("Overlay")
        axs[2].set_xlabel("Frequency Bin")
        axs[2].set_ylabel("Magnitude")
        axs[2].legend()

        plt.tight_layout()

        out_path = os.path.join(save_dir, f"infer_{i:04d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    print("✅ All inference figures saved.")

def plot_xb_yb(xb, yb, sample_idx=0):
    """
    Plot xb (input) and yb (target) for a single sample from a batch
    on separate axes.

    xb, yb: torch tensors of shape (B, T, 1)
    sample_idx: which sample in batch to plot
    """

    x = xb[sample_idx].detach().cpu().numpy()[:, 0]
    y = yb[sample_idx].detach().cpu().numpy()[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # xb
    axes[0].plot(x)
    axes[0].set_title("xb (input)")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True)

    # yb
    axes[1].plot(y)
    axes[1].set_title("yb (target)")
    axes[1].set_xlabel("Sequence index")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"xb_yb_{sample_idx}.png", dpi=300)
    plt.show()



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataset
    with h5py.File(config.H5_PATH, 'r') as f:
        N = len(f[config.X_KEY])

    all_idx = np.arange(N)
    np.random.shuffle(all_idx)

    n_train = int(N * config.TRAIN_RATIO)
    test_idx = all_idx[n_train:]

    test_ds = H5SpectraDataset(
        config.H5_PATH,
        config.X_KEY,
        config.Y_KEY,
        indices=test_idx,
        normalize=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Grab ONE batch and plot
    for xb, yb in test_loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        eps = 1e-2  # adjust as needed

        xb_nz = torch.sum(torch.abs(xb) > eps).item()
        yb_nz = torch.sum(torch.abs(yb) > eps).item()

        print("xb nz:", xb_nz)
        print("yb nz:", yb_nz)
        print(f"xb nonzeros: {xb_nz}")
        print(f"yb nonzeros: {yb_nz}")

        plot_xb_yb(xb, yb)
        break

