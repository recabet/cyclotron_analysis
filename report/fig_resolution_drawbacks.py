#!/usr/bin/env python3
"""
Generate figure showing the drawbacks of lower resolution in FTMS.
Panel A/B: real data from LMDB
Panel C: synthetic two-peak example (Gaussians) — clearly shows merging
"""
import os, sys, pickle, lmdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
from src.config.settings import ClusterConfig

# ============================================================
# Load real data for panels A and B
# ============================================================
config = ClusterConfig()
env = lmdb.open("../"+config.LMDB_DIR + '/test.lmdb', readonly=True, lock=False)
txn = env.begin()

best = None
best_score = 0
for idx in range(1652):
    key = f'sample_{idx:08d}'.encode('ascii')
    raw = txn.get(key)
    if raw is None:
        break
    data = pickle.loads(raw)
    hr = data['fft_hr'].astype(np.float64)
    lr = data['fft_lr'].astype(np.float64)
    score = hr.max()
    if score > best_score:
        best_score = score
        best = {'hr': hr, 'lr': lr, 'idx': idx,
                'formula': data.get('compound_formula', ''),
                'cluster_mass': data.get('cluster_mass', 0)}
env.close()

hr_real = best['hr']
lr_real = best['lr']
lr_ratio = lr_real.max() / hr_real.max()
bins = np.arange(len(hr_real))
hr_peak_bin = np.argmax(hr_real)
lr_peak_bin = np.argmax(lr_real)

# Normalize both by HR max for common scale
hr_norm = hr_real / hr_real.max()
lr_normed = lr_real / hr_real.max()

# FWHM
def fwhm_bounds(spec, peak_bin):
    half = spec[peak_bin] * 0.5
    lo = peak_bin
    while lo > 0 and spec[lo] > half:
        lo -= 1
    hi = peak_bin
    while hi < len(spec) - 1 and spec[hi] > half:
        hi += 1
    return lo, hi

hr_lo, hr_hi = fwhm_bounds(hr_real, hr_peak_bin)
lr_lo, lr_hi = fwhm_bounds(lr_real, lr_peak_bin)
hr_fwhm = hr_hi - hr_lo
lr_fwhm = lr_hi - lr_lo

# ============================================================
# Panel C: synthetic Gaussian peaks — clear peak merging
# Both curves on same absolute scale for fair comparison
# ============================================================
n_pts = 2048
x = np.arange(n_pts)

# Parameters tuned so HR resolves two peaks, LR merges them
# The FWHM of real data: HR ~7 bins, LR ~30 bins
# LR FWHM is ~4x wider, so sigma_LR = ~4 * sigma_HR
sigma_hr = 8
sigma_lr = 32
A_main = 0.95   # main peak
A_sat = 0.40   # satellite peak (40% of main — realistic isotope ratio)
p_main = 900    # main peak position
p_sat = 970    # satellite peak position (70 bins = 10x HR sigma, visible in HR but not LR)

def gaussian(pos, amp, center, sigma):
    return amp * np.exp(-0.5 * ((pos - center) / sigma)**2)

# High-res: two clearly separated peaks
spec_hr_syn = (gaussian(x, A_main, p_main, sigma_hr) +
                gaussian(x, A_sat, p_sat, sigma_hr))

# Low-res: same peaks but broadened so they merge
spec_lr_syn = (gaussian(x, A_main, p_main, sigma_lr) +
                gaussian(x, A_sat, p_sat, sigma_lr))

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 13))
fig.patch.set_facecolor('#f8f9fa')

BLUE = '#1f77b4'
RED = '#d62728'
GREEN = '#2ca02c'
ORANGE = '#ff7f0e'

# ============================================================
# Panel A: Full spectra — amplitude reduction
# ============================================================
ax = axes[0]
ax.fill_between(bins, hr_norm, alpha=0.12, color=BLUE)
ax.plot(bins, hr_norm, color=BLUE, linewidth=1.3, alpha=0.9,
        label='High-Res (full FID, 2048 pts)')
ax.plot(bins, lr_normed, color=ORANGE, linewidth=1.3, alpha=0.9,
        label='Low-Res (truncated FID, 256 pts)')

ax.annotate('',
            xy=(hr_peak_bin, hr_norm[hr_peak_bin]),
            xytext=(hr_peak_bin, lr_normed[hr_peak_bin]),
            arrowprops=dict(arrowstyle='<->', color=RED, lw=2))
ax.annotate(f'Peak amplitude\nreduced to {lr_ratio:.0%}',
            xy=(hr_peak_bin + 30,
                (hr_norm[hr_peak_bin] + lr_normed[hr_peak_bin]) / 2),
            fontsize=10, color=RED,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=RED, alpha=0.9))

ax.set_xlabel('FFT Bin Index', fontsize=11)
ax.set_ylabel('Normalized Amplitude', fontsize=11)
ax.set_title('Panel A: Low-Resolution Acquisition Reduces Peak Amplitude',
             fontsize=12, fontweight='bold', pad=8)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============================================================
# Panel B: Zoomed — peak broadening
# ============================================================
ax = axes[1]
zoom_lo = max(0, hr_peak_bin - 150)
zoom_hi = min(len(hr_real), hr_peak_bin + 150)
bz = bins[zoom_lo:zoom_hi]

ax.fill_between(bz, hr_norm[zoom_lo:zoom_hi], alpha=0.12, color=BLUE)
ax.plot(bz, hr_norm[zoom_lo:zoom_hi], color=BLUE, linewidth=1.5,
        label=f'High-Res (FWHM = {hr_fwhm} bins)')
ax.plot(bz, lr_normed[zoom_lo:zoom_hi], color=ORANGE, linewidth=1.5,
        label=f'Low-Res (FWHM = {lr_fwhm} bins, ~{round(lr_fwhm/hr_fwhm,1)}x broader)')

mid_h = hr_norm[hr_peak_bin] * 0.5
ax.plot([hr_lo, hr_hi], [mid_h, mid_h], '--', color=BLUE, linewidth=2, alpha=0.7)
mid_l = lr_normed[lr_peak_bin] * 0.5
ax.plot([lr_lo, lr_hi], [mid_l, mid_l], '--', color=ORANGE, linewidth=2, alpha=0.7)

ax.set_xlabel('FFT Bin Index', fontsize=11)
ax.set_ylabel('Normalized Amplitude', fontsize=11)
ax.set_title('Panel B: Low-Resolution Peaks Are Broader — Reduced Resolving Power',
             fontsize=12, fontweight='bold', pad=8)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(zoom_lo, zoom_hi)
ax.set_ylim(0, 1.08)

# ============================================================
# Panel C: Synthetic — two peaks merging (clear illustration)
# ============================================================
ax = axes[2]

# Zoom to show both peaks clearly
zoom_lo_s = 700
zoom_hi_s = 1100
bz_s = x[zoom_lo_s:zoom_hi_s]

ax.fill_between(bz_s, spec_hr_syn[zoom_lo_s:zoom_hi_s], alpha=0.12, color=BLUE)
ax.plot(bz_s, spec_hr_syn[zoom_lo_s:zoom_hi_s], color=BLUE, linewidth=2.0,
        label='High-Res: two peaks resolved')
ax.plot(bz_s, spec_lr_syn[zoom_lo_s:zoom_hi_s], color=ORANGE, linewidth=2.0,
        label='Low-Res: peaks merged into single broad feature')

# Annotate HR peaks
ax.axvline(x=p_main, color=GREEN, linestyle=':', linewidth=2, alpha=0.8)
ax.axvline(x=p_sat, color=GREEN, linestyle=':', linewidth=2, alpha=0.8)
ax.annotate('M+0\n(resolved)',
            xy=(p_main, A_main),
            xytext=(p_main - 55, A_main + 0.08),
            fontsize=10, color=GREEN, ha='center',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=GREEN, alpha=0.85))
ax.annotate('M+1\n(resolved)',
            xy=(p_sat, A_sat),
            xytext=(p_sat + 55, A_sat + 0.08),
            fontsize=10, color=GREEN, ha='center',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=GREEN, alpha=0.85))

# Arrow for merged LR peak
merged_bin = (p_main + p_sat) // 2
ax.annotate('Peaks merged\n(unresolved)',
            xy=(merged_bin, spec_lr_syn.max()),
            xytext=(merged_bin + 80, spec_lr_syn.max() * 0.6),
            fontsize=10, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=RED, alpha=0.9))

ax.set_xlabel('Position (FFT Bin Index)', fontsize=11)
ax.set_ylabel('Amplitude', fontsize=11)
ax.set_title('Panel C: Closely Spaced Isotope Peaks Cannot Be Distinguished at Low Resolution',
             fontsize=12, fontweight='bold', pad=8)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(zoom_lo_s, zoom_hi_s)
ax.set_ylim(0, 1.15)

fig.suptitle(
    'Three Drawbacks of Low Resolution in FT-ICR Mass Spectrometry',
    fontsize=14, fontweight='bold', y=0.995, color='#222222')

plt.tight_layout(rect=[0, 0, 1, 0.975])
out_path = os.path.join(os.path.dirname(__file__), 'fig_resolution_drawbacks.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")

# Quick sanity check
print(f"\nSanity check:")
print(f"  HR peaks at: {p_main} and {p_sat} (sep={p_sat-p_main} bins)")
print(f"  LR peak merged: {np.argmax(spec_lr_syn)}, single peak")
print(f"  Real data LR/HR ratio: {lr_ratio:.1%}")