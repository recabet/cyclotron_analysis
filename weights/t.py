#!/usr/bin/env python
"""
Plot training & validation loss from training_history_cluster_prof_v2.npz
using the colour scheme defined in the main test script.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless safe; comment out if you want interactive
import matplotlib.pyplot as plt

# ── Palette (identical to test script) ────────────────────────────────────
TARGET_COLOR = '#1f77b4'    # blue
INPUT_COLOR  = '#ff7f0e'    # orange
PRED_COLOR   = '#d62728'    # red
FIG_BG       = '#f0f2f5'
PANEL_BG     = '#ffffff'
GRID_COLOR   = '#d0d5dd'
SPINE_COLOR  = '#c0c5cc'
TITLE_COLOR  = '#1a1a2e'
LABEL_COLOR  = '#444455'
TICK_COLOR   = '#666677'

# ── Load data ─────────────────────────────────────────────────────────────
history = np.load("training_history_cluster_prof_v2.npz")
train_loss = history["train_loss"]
val_loss   = history["val_loss"]
print(min(val_loss))
print(min(train_loss))
epochs = np.arange(1, len(train_loss) + 1)

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 5.5), facecolor=FIG_BG)
ax = fig.add_subplot(111, facecolor=PANEL_BG)

# Lines
ax.plot(epochs, train_loss, label="Training loss",
        color=INPUT_COLOR, linewidth=1.8, alpha=0.95)
ax.plot(epochs, val_loss,   label="Validation loss",
        color=TARGET_COLOR, linewidth=1.8, alpha=0.95)

# Appearance
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
    spine.set_linewidth(0.8)
ax.tick_params(colors=TICK_COLOR, labelsize=10)
ax.set_xlabel("Epoch", fontsize=12, color=LABEL_COLOR)
ax.set_ylabel("Huber Loss", fontsize=12, color=LABEL_COLOR)
ax.grid(True, linestyle='--', linewidth=0.6, color=GRID_COLOR, alpha=0.5)

legend = ax.legend(fontsize=11, framealpha=0.9, edgecolor=SPINE_COLOR,
                   fancybox=False, handlelength=2.2)
legend.get_frame().set_linewidth(0.7)

# Title
ax.set_title("Training and Validation Loss — Cluster prof_v2",
             fontsize=14, fontweight='bold', color=TITLE_COLOR, pad=12)

# ── Save ───────────────────────────────────────────────────────────────────
output_path = "loss_curve.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
print(f"Loss curve saved → {output_path}")
plt.close(fig)