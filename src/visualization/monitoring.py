import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os

PLOTS_DIR = "plots/high_res"
os.makedirs(PLOTS_DIR, exist_ok=True,)


class HeadlessMonitor:
    """
    Drop-in replacement for TrainingMonitorGUI that works without a display.
    Prints a progress bar to stdout and saves plots/loss_curve.png each epoch.
    """

    def __init__(self, total_epochs, model_name="LSTM Seq2Seq"):
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        plt.style.use("dark_background")

    def update(self, epoch, train_loss, val_loss, status="Training..."):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        bar_len = 30
        filled = int(bar_len * epoch / self.total_epochs)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = epoch / self.total_epochs * 100

        print(
            f"\r[{bar}] {pct:5.1f}%  "
            f"epoch {epoch:04d}/{self.total_epochs}  "
            f"train {train_loss:.4e}  val {val_loss:.4e}  "
            f"best {self.best_val_loss:.4e}  | {status}",
            flush=True,
        )
        self._save_loss_curve()

    def set_status(self, message, color=None):
        print(f"\n[STATUS] {message}")

    def close(self):
        pass

    def _save_loss_curve(self):
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0e1117")
        ax.set_facecolor("#161b22")

        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, "o-", color="#4fc3f7",
                linewidth=1.5, markersize=4, label="Train Loss")
        ax.plot(epochs, self.val_losses, "s-", color="#ffb74d",
                linewidth=1.5, markersize=4, label="Val Loss")

        ax.set_xlabel("Epoch", color="#8b949e")
        ax.set_ylabel("Loss", color="#8b949e")
        ax.set_title(f"Training & Validation Loss — {self.model_name}",
                     color="#e6edf3", fontweight="bold")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, alpha=0.2, color="#30363d")
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=120)
        plt.close(fig)



class HeadlessSpectraPlotter:
    """
    Saves plots/spectra_epochNNNN.png every `plot_every` epochs.

    Columns: Low-res X  |  High-res Y  |  Prediction Ŷ
    Rows:    one per sample
    """

    COL_TITLES = [
        "Low-resolution Input  (X)",
        "High-resolution Ground Truth  (Y)",
        "Model Prediction  (Ŷ)",
    ]
    COL_COLORS = ["#4fc3f7", "#a5d6a7", "#ffb74d"]

    def __init__(self, plot_every: int = 5, n_samples: int = 3):
        self.plot_every = plot_every
        self.n_samples = n_samples

    @staticmethod
    def _squeeze(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[:, :, 0]
        return arr

    def save(self, epoch: int, x_np: np.ndarray,
             y_np: np.ndarray, pred_np: np.ndarray):
        x_np = self._squeeze(x_np)
        y_np = self._squeeze(y_np)
        pred_np = self._squeeze(pred_np)

        n_show = min(self.n_samples, x_np.shape[0])
        data_cols = [x_np, y_np, pred_np]

        fig = plt.figure(figsize=(15, 3.2 * n_show), facecolor="#0e1117")
        fig.suptitle(
            f"Spectra Preview — Epoch {epoch}",
            color="#e6edf3", fontsize=13, fontweight="bold", y=1.01,
        )

        gs = gridspec.GridSpec(n_show, 3, figure=fig, hspace=0.55, wspace=0.35)

        for row in range(n_show):
            for col, (data, color, title) in enumerate(
                    zip(data_cols, self.COL_COLORS, self.COL_TITLES)):

                ax = fig.add_subplot(gs[row, col])
                ax.set_facecolor("#161b22")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")
                ax.tick_params(colors="#8b949e", labelsize=7)
                ax.set_xlabel("Point index", color="#8b949e", fontsize=7)

                if col == 0:
                    ax.set_ylabel(f"Sample {row + 1}",
                                  color="#8b949e", fontsize=8, fontweight="bold")
                else:
                    ax.set_ylabel("Amplitude", color="#8b949e", fontsize=7)

                if row == 0:
                    ax.set_title(title, color=color,
                                 fontsize=9, fontweight="bold", pad=6)

                signal = data[row]
                xs = np.arange(len(signal))
                ax.plot(xs, signal, color=color, linewidth=0.8, alpha=0.9)
                ax.fill_between(xs, signal, alpha=0.12, color=color)
                ax.set_xlim(0, len(signal) - 1)

        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"spectra_epoch{epoch:04d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  📸  Spectra plot saved → {path}")
