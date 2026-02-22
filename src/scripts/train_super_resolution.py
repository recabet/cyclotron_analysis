#!/usr/bin/env python
"""
Super-resolution training for FT-ICR MS spectra using LSTM Seq2Seq.
Headless / SSH-compatible version — no Tkinter / $DISPLAY required.
Multi-GPU support via torch.nn.DataParallel.

Outputs:
  - Console progress bar each epoch
  - plots/loss_curve.png          — updated every epoch
  - plots/spectra_epochNNNN.png   — 3-panel spectra grid every PLOT_EVERY epochs
  - training_history.npz
  - best model checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py

import matplotlib
matplotlib.use("Agg")

from src.config import TrainingConfig
from src.models import LSTMSeq2Seq
from src.training import H5SpectraDataset, fit, metrics_np
from src.visualization import (
    HeadlessSpectraPlotter,
    PLOTS_DIR,
    HeadlessMonitor,
)
# torch.backends.cudnn.enabled = False


class SpectraPlotCallback:
    """
    Passed to fit() as epoch_end_callback.
    Signature: callback(epoch, model, device)

    Unwraps DataParallel automatically so inference always runs on GPU 0
    with a small fixed batch (no need to split across GPUs for preview).
    """

    def __init__(self, val_loader: DataLoader,
                 plotter: HeadlessSpectraPlotter,
                 device, plot_every: int = 5):
        self.plot_every = plot_every
        self.plotter = plotter
        self.device = device

        xb_ref, yb_ref = next(iter(val_loader))
        n = min(plotter.n_samples, xb_ref.shape[0])
        self._xb = xb_ref[:n].float()
        self._yb = yb_ref[:n].float()

    def __call__(self, epoch: int, model: nn.Module, device):
        if epoch % self.plot_every != 0:
            return

        raw_model = model.module if isinstance(model, nn.DataParallel) else model

        raw_model.eval()
        with torch.no_grad():
            pred = raw_model(self._xb.to(device)).cpu().numpy()
        raw_model.train()

        self.plotter.save(epoch, self._xb.numpy(), self._yb.numpy(), pred)


def main():
    config = TrainingConfig()

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    if n_gpus > 1:
        print(f"🚀  {n_gpus} GPUs detected — enabling DataParallel across all of them.")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name}  ({props.total_memory // 1024 ** 2} MB)")
    elif n_gpus == 1:
        print("    1 GPU detected.")
    else:
        print("⚠️   No GPU — running on CPU.")

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    with h5py.File(config.H5_PATH, "r") as f:
        N = len(f[config.X_KEY])
        seq_len = f[config.X_KEY].shape[1]
    print(f"Total samples: {N}  |  Sequence length: {seq_len}")

    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_train = int(N * config.TRAIN_RATIO)
    n_val = int(N * config.VAL_RATIO)
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    train_ds = H5SpectraDataset(config.H5_PATH,
                                config.X_KEY,
                                config.Y_KEY,
                                indices=train_idx,
                                normalize=False)
    val_ds = H5SpectraDataset(config.H5_PATH,
                              config.X_KEY,
                              config.Y_KEY,
                              indices=val_idx,
                              normalize=False)
    test_ds = H5SpectraDataset(config.H5_PATH,
                               config.X_KEY,
                               config.Y_KEY,
                               indices=test_idx,
                               normalize=False)

    sample_x, sample_y = train_ds[0]
    print(f"x shape: {sample_x.shape}  y shape: {sample_y.shape}")

    effective_batch = config.BATCH_SIZE
    print(f"Per-GPU batch size:   {config.BATCH_SIZE}")
    print(f"Effective batch size: {effective_batch}  (×{max(n_gpus, 1)} GPUs)")

    train_loader = DataLoader(train_ds,
                              batch_size=effective_batch,
                              shuffle=False,
                              num_workers=0)
    val_loader = DataLoader(val_ds,
                            batch_size=effective_batch,
                            shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds,
                             batch_size=effective_batch,
                             shuffle=False,
                             num_workers=0)

    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

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

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"✅  Model wrapped with DataParallel (GPUs: {list(range(n_gpus))})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("CONFIGURING TRAINING")
    print("=" * 60)

    if config.LOSS.lower() == "mse":
        criterion = nn.MSELoss()
    elif config.LOSS.lower() == "mae":
        criterion = nn.L1Loss()
    elif config.LOSS.lower() == "huber":
        criterion = nn.HuberLoss(delta=config.HUBER_DELTA)
    else:
        raise ValueError(f"Unsupported loss: {config.LOSS}")

    if config.OPTIMIZER.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Loss: {config.LOSS}  Optimizer: {config.OPTIMIZER}")
    print(f"LR: {config.LEARNING_RATE}  Epochs: {config.EPOCHS}  Patience: {config.PATIENCE}")

    plot_every = getattr(config, "PLOT_EVERY", 1)
    n_preview_samples = getattr(config, "N_PREVIEW_SAMPLES", 3)

    monitor = HeadlessMonitor(total_epochs=config.EPOCHS, model_name="LSTM Seq2Seq")
    plotter = HeadlessSpectraPlotter(plot_every=plot_every, n_samples=n_preview_samples)
    spectra_cb = SpectraPlotCallback(
        val_loader=val_loader,
        plotter=plotter,
        device=device,
        plot_every=plot_every,
    )

    print(f"\n📊  Loss curve  → {PLOTS_DIR}/loss_curve.png  (updated every epoch)")
    print(f"📸  Spectra PNG → {PLOTS_DIR}/spectra_epochNNNN.png  (every {plot_every} epochs)")
    print(f"    Previewing {n_preview_samples} validation samples per update.\n")

    monitor.set_status("Training starting…")

    try:
        history, best_val_loss = fit(
            model, train_loader, val_loader, criterion, optimizer, device,
            epochs=config.EPOCHS,
            patience=config.PATIENCE,
            model_save_path=config.MODEL_SAVE_PATH,
            clip_norm=config.CLIP_NORM,
            gui=monitor,
            epoch_end_callback=spectra_cb,
        )
        print(f"\n✅ Training complete. Best val loss: {best_val_loss:.6e}")

        # Final spectra snapshot
        final_epoch = len(history["train"])
        spectra_cb(final_epoch, model, device)

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise

    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()
            y_hat = model(xb)
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)[:, :, 0]
    y_pred = np.concatenate(y_pred_list, axis=0)[:, :, 0]

    metrics = metrics_np(y_true, y_pred)
    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(raw_model.state_dict(), config.MODEL_SAVE_PATH)

    history_path = "training_history.npz"
    np.savez(history_path, train_loss=history["train"], val_loss=history["val"])
    print(f"\n📊 Training history → {history_path}")
    print(f"💾 Model checkpoint → {config.MODEL_SAVE_PATH}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
