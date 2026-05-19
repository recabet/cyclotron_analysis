#!/usr/bin/env python
"""
Isotope Cluster Super-Resolution Training using prof_v2 LMDB datasets.

This script trains a model on isotope clusters using LMDB for fast,
efficient data loading with multiprocessing.

Key differences from narrowband training:
- Uses truncated acquisition time simulation (before apodization)
- Both outputs are 2048 points (but low-res has broader peaks)
- Trains on individual isotope clusters instead of full spectra

Headless / SSH-compatible version — no Tkinter / $DISPLAY required.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")

from src.config.settings import ClusterConfig, ClusterTrainingConfig
from src.models import LSTMSeq2Seq
from src.training import fit, metrics_np
from src.training.dataset_cluster_lmdb import ClusterLMDBDataModule
from src.visualization import (
    HeadlessSpectraPlotter,
    PLOTS_DIR,
    HeadlessMonitor,
)

torch.set_float32_matmul_precision('high')


class SpectraPlotCallback:
    """Callback for plotting spectra previews during training."""

    def __init__(self, val_loader, plotter: HeadlessSpectraPlotter, device, plot_every: int = 5):
        self.plot_every = plot_every
        self.plotter = plotter
        self.device = device
        self.val_loader = val_loader
        self._xb = None
        self._yb = None

    def _load_reference_batch(self):
        """Load reference batch from validation set on first call."""
        if self._xb is None:
            xb_ref, yb_ref = next(iter(self.val_loader))
            n = min(self.plotter.n_samples, xb_ref.shape[0])
            self._xb = xb_ref[:n].float()
            self._yb = yb_ref[:n].float()

    def __call__(self, epoch: int, model: nn.Module, device):
        if epoch % self.plot_every != 0:
            return

        self._load_reference_batch()

        raw_model = model.module if isinstance(model, nn.DataParallel) else model

        raw_model.eval()
        with torch.no_grad():
            pred = raw_model(self._xb.to(device)).cpu().numpy()
        raw_model.train()

        self.plotter.save(epoch, self._xb.numpy(), self._yb.numpy(), pred)


def main():
    data_config = ClusterConfig()
    config = ClusterTrainingConfig()

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    if n_gpus > 1:
        print(f"🚀  {n_gpus} GPUs detected — enabling DataParallel.")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name}  ({props.total_memory // 1024 ** 2} MB)")
    elif n_gpus == 1:
        props = torch.cuda.get_device_properties(0)
        print(f"    1 GPU detected: {props.name}  ({props.total_memory // 1024 ** 2} MB)")
    else:
        print("⚠️   No GPU — running on CPU.")

    amp_active = config.USE_AMP and device.type == "cuda"
    print(f"Mixed precision (AMP):  {'✅ enabled' if amp_active else '❌ disabled'}")
    print(f"DataLoader workers:     {config.NUM_WORKERS}")
    print(f"Batch size:             {config.BATCH_SIZE}")

    print("\n" + "=" * 60)
    print("LOADING LMDB DATASETS")
    print("=" * 60)

    # Create data module
    data_module = ClusterLMDBDataModule(
        lmdb_path=config.LMDB_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        normalize=True
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Get sample to verify shapes
    xb, yb = next(iter(train_loader))
    print(f"x shape: {xb.shape}  y shape: {yb.shape}")

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
        print(f"✅  Model wrapped with DataParallel")

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

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    print(f"Loss:      {config.LOSS} (delta={config.HUBER_DELTA})")
    print(f"Optimizer: {config.OPTIMIZER}")
    print(f"LR:        {config.LEARNING_RATE}  Epochs: {config.EPOCHS}")
    print(f"Scheduler: OneCycleLR with 10% warmup")

    plot_every = config.PLOT_EVERY
    n_preview_samples = config.N_PREVIEW_SAMPLES

    monitor = HeadlessMonitor(total_epochs=config.EPOCHS, model_name="Cluster prof_v2 LMDB")
    plotter = HeadlessSpectraPlotter(plot_every=plot_every, n_samples=n_preview_samples)
    spectra_cb = SpectraPlotCallback(
        val_loader=val_loader,
        plotter=plotter,
        device=device,
        plot_every=plot_every,
    )

    print(f"\n📊  Loss curve  → {PLOTS_DIR}/loss_curve.png")
    print(f"📸  Spectra PNG → {PLOTS_DIR}/spectra_epochNNNN.png")

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
            scheduler=scheduler,
            use_amp=amp_active,
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

    test_loader = data_module.test_dataloader()
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

    history_path = "training_history_cluster_prof_v2.npz"
    np.savez(history_path, train_loss=history["train"], val_loss=history["val"])
    print(f"\n📊 Training history → {history_path}")
    print(f"💾 Model checkpoint → {config.MODEL_SAVE_PATH}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
