#!/usr/bin/env python
"""
Test script for Cluster prof_v2 model.
Loads model trained by train_cluster_prof_v2.py, computes spectral metrics
on EVERY test sample (all replicates), and generates plots for a subset
(3 lightest masses per compound).

Metrics are evaluated on the normalised [0,1] arrays that appear in the plots.

Output structure:
  test_plots/cluster_const_compound_v2/
  ├── metrics_summary.json          ← global spectral metrics (all 2122 samples)
  └── <FormulaA>/                   ← plots for selected compounds
      ├── M+0.png
      ├── M+1.png
      └── M+2.png
"""
import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, Counter

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.config.settings import ClusterConfig, ClusterTrainingConfig
from src.models import LSTMSeq2Seq
from src.training.dataset_cluster_lmdb import ClusterLMDBDataModule
from src.evaluation.spectral_metrics import (
    SpectrumEvalConfig,
    evaluate_spectrum,
)

try:
    import IsoSpecPy as iso
    HAS_ISOSPEC = True
except ImportError:
    HAS_ISOSPEC = False


# ── palette ─────────────────────────────────────────────────────────────────
TARGET_COLOR = '#1f77b4'
INPUT_COLOR  = '#ff7f0e'
PRED_COLOR   = '#d62728'
FIG_BG       = '#f0f2f5'
PANEL_BG     = '#ffffff'
GRID_COLOR   = '#d0d5dd'
SPINE_COLOR  = '#c0c5cc'
TITLE_COLOR  = '#1a1a2e'
LABEL_COLOR  = '#444455'
TICK_COLOR   = '#666677'

FREQ_RES_HZ_PER_BIN = 244.140625

# ── Number of isotopes to plot (metrics are for ALL samples) ────────────────
MAX_PLOT_ISOTOPES = 3   # M+0, M+1, M+2 for plots

# ── Spectral evaluation config ──────────────────────────────────────────────
EVAL_CONFIG = SpectrumEvalConfig(
    peak_height=0.65,
    peak_prominence=0.25,
    peak_distance=20,
    match_tolerance=5,
    envelope_window=31,
    freq_hz_per_bin=FREQ_RES_HZ_PER_BIN,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def compute_mol_weight(formula: str) -> float:
    if not HAS_ISOSPEC:
        return 0.0
    try:
        sp = iso.IsoTotalProb(formula=formula, prob_to_cover=1.00)
        s, total_p = 0.0, 0.0
        for m, p in zip(sp.masses, sp.probs):
            s += m * p
            total_p += p
        return s / total_p
    except Exception:
        return 0.0


def norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def _infer_one_sample(idx: int, test_dataset, model, device):
    """Return (x_raw, y_noisy, pred_norm) for one sample index."""
    with test_dataset.env.begin() as txn:
        key = f'sample_{idx:08d}'.encode('ascii')
        data = pickle.loads(txn.get(key))
        x_raw = data['fft_lr'].astype(np.float32)
        y_noisy = data['fft_hr'].astype(np.float32)

    x_norm = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min() + 1e-12)
    x_t = torch.from_numpy(x_norm).reshape(1, -1, 1).float().to(device)
    with torch.no_grad():
        pred_norm_t = model(x_t)
    pred_norm_np = pred_norm_t.detach().cpu().numpy()[0, :, 0]
    return x_raw, y_noisy, pred_norm_np


# ------------------------------------------------------------
# Plot a single cluster → one PNG
# ------------------------------------------------------------
def plot_single_cluster(cd, formula, mol_weight, compound_dir):
    fig = plt.figure(figsize=(13, 4.2), facecolor=FIG_BG)
    gs = gridspec.GridSpec(1, 1, figure=fig,
                           left=0.08, right=0.97, top=0.82, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
        spine.set_linewidth(0.8)

    xp = cd['xp']
    x_p = norm(cd['x_p'])
    y_p = norm(cd['y_p'])
    pr_p = norm(cd['pr_p'])

    ax.fill_between(xp, x_p, alpha=0.18, color=INPUT_COLOR)
    ax.plot(xp, x_p, label="Input (Low-Res)",
            linewidth=1.4, alpha=0.85, color=INPUT_COLOR)
    ax.fill_between(xp, y_p, alpha=0.12, color=TARGET_COLOR)
    ax.plot(xp, y_p, label="Target (High-Res, noisy)",
            linewidth=1.6, alpha=0.90, color=TARGET_COLOR)
    ax.plot(xp, pr_p, label="Prediction",
            linewidth=1.6, alpha=0.90, color=PRED_COLOR,
            linestyle='--', dashes=(5, 2))

    ax.set_ylabel("Normalized Amplitude", fontsize=10, color=LABEL_COLOR, labelpad=6)
    ax.set_xlabel("FFT Bin Index", fontsize=10, color=LABEL_COLOR, labelpad=6)
    ax.tick_params(axis='both', labelsize=9, colors=TICK_COLOR)
    ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.6, color=GRID_COLOR)
    ax.set_ylim(-0.05, 1.18)

    peak_idx = int(np.argmax(y_p))
    peak_x = xp[peak_idx]
    peak_y = y_p[peak_idx]
    clean_peak = cd.get('clean_peak', 0)
    offset = max(5, int(len(xp) * 0.03))
    ax.annotate(
        f"peak = {clean_peak:.2f}",
        xy=(peak_x, peak_y),
        xytext=(peak_x + offset, peak_y * 0.86),
        fontsize=8.5, color=TARGET_COLOR, ha='left', va='top',
        arrowprops=dict(arrowstyle='->', color=TARGET_COLOR,
                        lw=1.0, connectionstyle='arc3,rad=0.0'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=TARGET_COLOR, alpha=0.88, linewidth=0.8),
    )
    leg = ax.legend(loc='upper right', fontsize=9, framealpha=0.90,
                    edgecolor=SPINE_COLOR, fancybox=False, handlelength=2.2)
    leg.get_frame().set_linewidth(0.7)

    weight_str = f"{mol_weight:.2f}" if mol_weight > 0 else "?"
    fig.suptitle(
        f"{formula}   MW = {weight_str}   |   "
        f"{cd['m_plus_label']}   rel_amp = {cd['rel_amp']:.4f}",
        fontsize=13, fontweight='bold', color=TITLE_COLOR, y=0.97,
    )

    safe_label = cd['m_plus_label'].replace('+', 'plus').replace('-', 'minus')
    fig_path = os.path.join(compound_dir, f"{safe_label}.png")
    fig.savefig(fig_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test Cluster prof_v2 model")
    parser.add_argument("--compounds", type=str, default=None,
                        help="Comma-separated list of compounds to plot [default: auto-select]")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top compounds to plot by cluster count [default: 10]")
    args = parser.parse_args()

    data_config = ClusterConfig()
    config = ClusterTrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
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

    if os.path.exists(config.MODEL_SAVE_PATH):
        state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v
                          for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"Loaded model from {config.MODEL_SAVE_PATH}")
    else:
        print(f"WARNING: Model not found at {config.MODEL_SAVE_PATH}, using untrained model")

    if isinstance(model, nn.DataParallel):
        model = model.module

    data_module = ClusterLMDBDataModule(
        lmdb_path=data_config.LMDB_DIR,
        batch_size=1,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        normalize=True,
    )
    test_dataset = data_module.test_dataset
    n_test = len(test_dataset)

    root_dir = "test_plots/cluster_const_compound_v2"
    os.makedirs(root_dir, exist_ok=True)
    print(f"\nRoot output directory: {root_dir}/")

    # ════════════════════════════════════════════════════════════════════════
    # PASS 1 — compute spectral metrics on EVERY test sample (all 2122)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n[PASS 1] Evaluating metrics on all {n_test} test samples …")

    formula_counter = Counter()
    metric_accumulator = defaultdict(list)

    for idx in range(n_test):
        # read metadata for formula counting (used later for plotting)
        meta = test_dataset.get_metadata(idx)
        formula = meta['compound_formula']
        formula_counter[formula] += 1

        # inference
        _, y_noisy, pred_norm = _infer_one_sample(idx, test_dataset, model, device)
        y_norm = (y_noisy - y_noisy.min()) / (y_noisy.max() - y_noisy.min() + 1e-12)

        # compute spectral metrics
        m = evaluate_spectrum(pred_norm, y_norm, EVAL_CONFIG)
        for key, val in m.to_dict().items():
            if val is not None:
                metric_accumulator[key].append(float(val))

    print(f"  Total compounds: {len(formula_counter)}")
    print(f"  Total samples evaluated: {len(metric_accumulator.get('rmse', []))}")

    # ── Print and save global metric summary ──────────────────────────────
    display_keys = [
        ("rmse",                  "RMSE"),
        ("mae",                   "MAE"),
        ("pearson_r",             "Pearson r"),
        ("snr",                   "SNR (dB)"),
        ("sam",                   "SAM (rad)"),
        ("max_xcorr",             "Max XCorr"),
        ("precision",             "Precision"),
        ("recall",                "Recall"),
        ("f1",                    "F1"),
        ("mean_pos_error",        "Peak pos err (bins)"),
        ("fwhm_gt_bins",          "FWHM GT (bins)"),
        ("fwhm_pred_bins",        "FWHM Pred (bins)"),
        ("fwhm_ratio",            "FWHM ratio"),
        ("mean_fwhm_improvement", "FWHM improvement"),
        ("fwhm_gt_hz",            "FWHM GT (Hz)"),
        ("fwhm_pred_hz",          "FWHM Pred (Hz)"),
        ("envelope_l2_error",     "Envelope L2"),
        ("envelope_kl_div",       "Envelope KL"),
    ]

    total_evals = len(metric_accumulator.get('rmse', []))
    print("\n" + "=" * 68)
    print(f"  SPECTRAL METRICS (all samples, n = {total_evals})")
    print("=" * 68)
    print(f"  {'Metric':<28}  {'Mean':>12}  {'Std':>12}  {'n':>6}")
    print("-" * 68)
    for key, display in display_keys:
        vals = metric_accumulator.get(key, [])
        if vals:
            print(f"  {display:<28}  {np.mean(vals):>12.5f}  {np.std(vals):>12.5f}  {len(vals):>6}")
    print("=" * 68)

    summary = {
        key: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for key, vals in metric_accumulator.items()
    }
    summary_path = os.path.join(root_dir, "metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Metrics saved → {summary_path}")

    # ════════════════════════════════════════════════════════════════════════
    # Build a dedup map (first-seen per formula+mass) ONLY for plotting
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n[PASS 2] Building dedup map for plotting …")
    # formula -> dict of cluster_mass -> (sample_index, max_amplitude)
    compound_isotopes = defaultdict(dict)

    for idx in range(n_test):
        meta = test_dataset.get_metadata(idx)
        formula = meta['compound_formula']
        cmass = meta['cluster_mass']

        if cmass not in compound_isotopes[formula]:
            with test_dataset.env.begin() as txn:
                key = f'sample_{idx:08d}'.encode('ascii')
                data = pickle.loads(txn.get(key))
                amp = float(np.max(data['fft_hr']))
            compound_isotopes[formula][cmass] = (idx, amp)

    # ════════════════════════════════════════════════════════════════════════
    # PASS 3 — plots for selected compounds (3 lightest masses only)
    # ════════════════════════════════════════════════════════════════════════
    if args.compounds:
        plot_formulas = [f.strip() for f in args.compounds.split(',')]
        print(f"\n[PASS 3] Plotting user‑specified compounds: {plot_formulas}")
    else:
        top_pairs = formula_counter.most_common(args.top_n)
        plot_formulas = [f for f, _ in top_pairs]
        print(f"\n[PASS 3] Plotting top-{args.top_n} compounds by cluster count:")
        for f, c in top_pairs:
            print(f"  {f}: {c} clusters")

    for formula in plot_formulas:
        if formula not in compound_isotopes:
            print(f"  WARNING: {formula} not found in test set — skipping")
            continue

        cmass_dict = compound_isotopes[formula]
        # Plot only the 3 lightest isotopes
        top_cmasses = sorted(cmass_dict.keys())[:MAX_PLOT_ISOTOPES]
        if not top_cmasses:
            continue

        min_cmass = top_cmasses[0]   # M+0
        mol_w = compute_mol_weight(formula)
        safe_formula = re.sub(r'[^a-zA-Z0-9]', '', formula)
        compound_dir = os.path.join(root_dir, safe_formula)
        os.makedirs(compound_dir, exist_ok=True)

        ref_amp = cmass_dict[min_cmass][1]

        print(f"\n  {formula}  →  {compound_dir}/")
        print(f"    MW={mol_w:.2f}  |  {len(top_cmasses)} clusters plotted  |  "
              f"ref_abundance(M+0)={ref_amp:.6f}")

        for cmass in top_cmasses:
            idx, amp = cmass_dict[cmass]
            x_raw, y_noisy, pred_norm = _infer_one_sample(idx, test_dataset, model, device)

            xp = np.arange(len(y_noisy))
            rel_amp = amp / ref_amp if ref_amp > 0 else 0.0
            m_plus_label = f"M+{cmass - min_cmass}"

            cd = dict(
                m_plus_label=m_plus_label,
                rel_amp=rel_amp,
                xp=xp,
                x_p=x_raw.copy(),
                y_p=y_noisy.copy(),
                pr_p=pred_norm.copy(),
                cluster_mass=cmass,
                clean_peak=amp,
            )
            fig_path = plot_single_cluster(cd, formula, mol_w, compound_dir)
            print(f"      {m_plus_label}  rel_amp={rel_amp:.4f}  →  {fig_path}")

    print(f"\nDone. Output layout:")
    print(f"  {root_dir}/")
    print(f"    metrics_summary.json")
    for formula in plot_formulas:
        safe = re.sub(r'[^a-zA-Z0-9]', '', formula)
        if safe in os.listdir(root_dir):
            print(f"    {safe}/")
    print()


if __name__ == "__main__":
    main()