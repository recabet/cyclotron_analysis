#!/usr/bin/env python
"""
Super-resolution training for FT-ICR MS spectra using LSTM Seq2Seq.
Loads preprocessed HDF5 dataset (from generate_training_data.py) and trains a model.
Includes real-time training visualization with Tkinter GUI.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
import h5py
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.config.settings import TrainingConfig
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.training.dataset import H5SpectraDataset
from src.training.trainer import fit
from src.training.metrics import metrics_np


torch.backends.cudnn.enabled = False


class TrainingMonitorGUI:
    """
    Real-time training monitor with Tkinter GUI.
    Shows loss curves, current metrics, and training progress.
    """

    def __init__(self, total_epochs, model_name="LSTM Seq2Seq"):

        self.root = tk.Tk()
        self.root.title(f"Training Monitor - {model_name}")
        self.root.geometry("1000x700")

        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        self._setup_ui()

    def _setup_ui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ----- Top: Info Panel -----
        info_frame = ttk.LabelFrame(main_frame, text="Training Status", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            info_frame,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        self.progress_bar.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))

        # Epoch counter
        self.epoch_label = ttk.Label(info_frame, text="Epoch: 0 / 0", font=("Arial", 12, "bold"))
        self.epoch_label.grid(row=1, column=0, sticky=tk.W)

        # Best validation loss
        self.best_loss_label = ttk.Label(info_frame, text="Best Val Loss: N/A", font=("Arial", 11))
        self.best_loss_label.grid(row=1, column=1, sticky=tk.E)

        # Current losses
        self.train_loss_label = ttk.Label(info_frame, text="Train Loss: N/A", font=("Arial", 10))
        self.train_loss_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        self.val_loss_label = ttk.Label(info_frame, text="Val Loss: N/A", font=("Arial", 10))
        self.val_loss_label.grid(row=2, column=1, sticky=tk.E, pady=(5, 0))

        # Status message
        self.status_label = ttk.Label(info_frame, text="Status: Initializing...", foreground="blue")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(5, 0))

        # ----- Bottom: Plot -----
        plot_frame = ttk.LabelFrame(main_frame, text="Loss Curves", padding="10")
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create matplotlib figure
        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Epoch', fontsize=11)
        self.ax.set_ylabel('Loss', fontsize=11)
        self.ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        # Initialize empty plots
        self.train_line, = self.ax.plot([], [], 'o-', color='#1f77b4', linewidth=2,
                                        markersize=5, label='Train Loss')
        self.val_line, = self.ax.plot([], [], 's-', color='#ff7f0e', linewidth=2,
                                      markersize=5, label='Val Loss')
        self.ax.legend(loc='upper right', fontsize=10)

        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update(self, epoch, train_loss, val_loss, status="Training..."):
        """Update the GUI with new training metrics."""
        self.current_epoch = epoch
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Update progress bar
        progress = (epoch / self.total_epochs) * 100
        self.progress_var.set(progress)

        # Update labels
        self.epoch_label.config(text=f"Epoch: {epoch} / {self.total_epochs}")
        self.best_loss_label.config(text=f"Best Val Loss: {self.best_val_loss:.6e}")
        self.train_loss_label.config(text=f"Train Loss: {train_loss:.6e}")
        self.val_loss_label.config(text=f"Val Loss: {val_loss:.6e}")
        self.status_label.config(text=f"Status: {status}")

        # Update plot
        epochs = list(range(1, len(self.train_losses) + 1))
        self.train_line.set_data(epochs, self.train_losses)
        self.val_line.set_data(epochs, self.val_losses)

        # Auto-scale axes
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw
        self.canvas.draw()
        self.root.update()

    def set_status(self, message, color="blue"):
        """Set status message with color."""
        self.status_label.config(text=f"Status: {message}", foreground=color)
        self.root.update()

    def close(self):
        """Close the GUI window."""
        self.root.quit()
        self.root.destroy()



def main():
    config = TrainingConfig()

    # Set seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # 1. Load dataset and split indices
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    with h5py.File(config.H5_PATH, 'r') as f:
        N = len(f[config.X_KEY])
        seq_len = f[config.X_KEY].shape[1]
    print(f"Total samples: {N}")
    print(f"Sequence length: {seq_len}")

    # Create indices
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    n_train = int(N * config.TRAIN_RATIO)
    n_val = int(N * config.VAL_RATIO)
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")

    # ------------------------------------------------------------
    # 2. Create datasets and loaders
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    train_ds = H5SpectraDataset(config.H5_PATH, config.X_KEY, config.Y_KEY,
                                indices=train_idx, normalize=True)
    sample_x, sample_y = train_ds[0]  # get one sample
    print(f"\nSample shapes:")
    print(f"  x shape: {sample_x.shape}")  # should be (seq_len, 1)
    print(f"  y shape: {sample_y.shape}")

    val_ds = H5SpectraDataset(config.H5_PATH, config.X_KEY, config.Y_KEY,
                              indices=val_idx, normalize=True)

    test_ds = H5SpectraDataset(config.H5_PATH, config.X_KEY, config.Y_KEY,
                               indices=test_idx, normalize=True)

    train_loader = DataLoader(train_ds,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)

    val_loader = DataLoader(val_ds,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    test_loader = DataLoader(test_ds,
                             batch_size=config.BATCH_SIZE,
                             shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------
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
        attn_layers=config.ATTN_LAYERS
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"TOTAL parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # seq_len_proxy = 2048  # smaller than full FID to avoid OOM
    # x = torch.randn(1, seq_len_proxy, 1, device=device)
    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
    #     with torch.no_grad():  # no grad needed for memory estimate
    #         model(x)
    #
    # print("\n=== CUDA Memory Usage Estimate (per op) ===\n")
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_memory_usage",
    #     row_limit=20
    # ))

    # # Print total params and batch-size estimate
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"\nTotal parameters: {total_params:,} (~{total_params * 4 / 1024 ** 2:.2f} MB float32)")
    #
    # # Rough batch memory estimate (forward pass only)
    # batch_mem_MB = (x.numel() * 4 + sum(p.numel() for p in model.parameters()) * 4) / 1024 ** 2
    # print(f"Rough memory for batch_size=1, seq_len={seq_len_proxy}: {batch_mem_MB:.2f} MB")

    # ------------------------------------------------------------
    # 4. Loss and optimizer
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONFIGURING TRAINING")
    print("=" * 60)

    if config.LOSS.lower() == 'mse':
        criterion = nn.MSELoss()
    elif config.LOSS.lower() == 'mae':
        criterion = nn.L1Loss()
    elif config.LOSS.lower() == 'huber':
        criterion = nn.HuberLoss(delta=config.HUBER_DELTA)
    else:
        raise ValueError(f"Unsupported loss: {config.LOSS}")

    print(f"Loss function: {config.LOSS}")

    if config.OPTIMIZER.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS
        )
    else:
        # fallback to Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Optimizer: {config.OPTIMIZER}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Patience: {config.PATIENCE}")

    # ------------------------------------------------------------
    # 5. Initialize GUI and Train
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STARTING TRAINING WITH GUI")
    print("=" * 60)
    print("\n🖥️  Launching training monitor GUI...")

    gui = TrainingMonitorGUI(total_epochs=config.EPOCHS, model_name="LSTM Seq2Seq")
    gui.set_status("Training starting...", "blue")

    try:
        history, best_val_loss = fit(
            model, train_loader, val_loader, criterion, optimizer, device,
            epochs=config.EPOCHS,
            patience=config.PATIENCE,
            model_save_path=config.MODEL_SAVE_PATH,
            clip_norm=config.CLIP_NORM,
            gui=gui
        )
        print(f"\n✅ Training completed. Best validation loss: {best_val_loss:.6e}")

        # Keep GUI open for a moment
        print("\nGUI will close in 5 seconds...")
        gui.root.after(5000, gui.close)
        gui.root.mainloop()

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        gui.set_status(f"Error: {e}", "red")
        gui.root.after(3000, gui.close)
        gui.root.mainloop()
        raise

    # ------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    model.eval()
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

    # ------------------------------------------------------------
    # 7. Save history
    # ------------------------------------------------------------
    history_path = "training_history.npz"
    np.savez(history_path, train_loss=history["train"], val_loss=history["val"])
    print(f"\n📊 Training history saved to {history_path}")

    print(f"\n💾 Model saved to {config.MODEL_SAVE_PATH}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()