#!/usr/bin/env python
"""
Inspect and visualise all generated FT-ICR MS data.

Run this AFTER `generate_training_data.py` to verify:
  - FID signals (raw, windowed, damped)
  - Full FFT spectra with theoretical peak markers
  - Cropped training segments and their length distribution
"""
import os
import sys
import numpy as np

from src.config.settings import SimulationConfig, TrainingConfig
from src.isotope.loader import load_compounds
from src.isotope.converter import process_all_compounds
from src.io.hdf5_readers import load_training_data
from src.visualization.spectrum_plots import plot_fft_comparison
from src.visualization.fid_plots import plot_fid_example
from src.visualization.diagnostics import (
    plot_segment_lengths_boxplot,
    plot_sample_segments,
)


def main():
    # ------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------
    sim_cfg = SimulationConfig()
    train_cfg = TrainingConfig()

    print("\n" + "=" * 60)
    print("FT-ICR MS DATA INSPECTION")
    print("=" * 60)

    # Ensure output directory for figures
    os.makedirs("../figures", exist_ok=True)

    # ------------------------------------------------------------
    # 2. Load compounds for theoretical peak positions
    # ------------------------------------------------------------
    print("\n--- Loading compound list ---")
    formulas, masses, _, rel_abund = load_compounds(
        sim_cfg.COMPOUNDS_FILE, coverage=sim_cfg.COVERAGE_PROB
    )

    frequencies = process_all_compounds(
        masses,
        sim_cfg.AVOGADRO,
        sim_cfg.MAGNETIC_FIELD,
        sim_cfg.ION_CHARGE,
        sim_cfg.ELECTRON_CHARGE,
    )

    print(f"   Loaded {len(formulas)} compounds")

    # ------------------------------------------------------------
    # 3. Inspect FID signals
    # ------------------------------------------------------------
    print("\n--- Plotting FID example (compound #1) ---")
    try:
        plot_fid_example(
            sim_cfg.FID_H5,
            formulas,
            index=1,
            sampling_rate=sim_cfg.SAMPLING_RATE,
            zoom_us=1000.0,
            save_path="../figures/fid_example.png",
        )
    except FileNotFoundError:
        print(f"⚠️  FID file not found: {sim_cfg.FID_H5}")
        print("   Run generate_training_data.py first.")
    except Exception as e:
        print(f"⚠️  Error plotting FID: {e}")

    # ------------------------------------------------------------
    # 4. Inspect full FFT spectra (static plot)
    # ------------------------------------------------------------
    print("\n--- Plotting FFT spectrum comparison (compound #1) ---")
    try:
        idx_example = 1
        plot_fft_comparison(
            sim_cfg.FFT_H5,
            formulas,
            index=idx_example,
            theoretical_freq=frequencies[idx_example],
            theoretical_amp=rel_abund[idx_example] * sim_cfg.MAX_AMPLITUDE,
            max_points=20000,
            save_path=f"../figures/fft_comparison_{idx_example}.png",
        )
    except FileNotFoundError:
        print(f"⚠️  FFT file not found: {sim_cfg.FFT_H5}")
    except Exception as e:
        print(f"⚠️  Error plotting FFT: {e}")

    # ------------------------------------------------------------
    # 5. Load training segments and run diagnostics
    # ------------------------------------------------------------
    print("\n--- Loading training segments ---")
    try:
        data = load_training_data(train_cfg.H5_PATH)

        print(f"   Training segments shape: {data['fft_hr'].shape}")
        print(f"   Kept compounds: {len(data['compounds'])}")
        print(f"   Segment length: {data['fft_hr'].shape[1]} bins")

        # ---- Segment length distribution (needs original spans) ----
        # We stored the effective spans, we can recompute lengths
        spans_eff = data["spans_eff"]  # (N, 2)
        lengths = spans_eff[:, 1] - spans_eff[:, 0]

        # Apply IQR outlier filter to get statistics
        print("\n--- Analyzing segment length distribution ---")
        try:
            from src.processing.filtering import iqr_outlier_filter
            keep_mask, _, lower, upper = iqr_outlier_filter(lengths, k=1.5)

            print(f"   Segment length range: {lengths.min():.0f} - {lengths.max():.0f} bins")
            print(f"   Mean length: {lengths.mean():.1f} bins")
            print(f"   Median length: {np.median(lengths[keep_mask]):.1f} bins")
            print(f"   IQR bounds: [{lower:.0f}, {upper:.0f}]")
            print(f"   Outliers detected: {(~keep_mask).sum()} / {len(lengths)}")

            # Boxplot
            print("\n--- Plotting segment length distribution ---")
            plot_segment_lengths_boxplot(
                lengths,
                lower,
                upper,
                keep_mask,
                save_path="../figures/segment_lengths_boxplot.png",
            )
        except ImportError:
            print("⚠️  Could not import iqr_outlier_filter from src.processing.filtering")
            print("   Skipping segment length analysis")
        except Exception as e:
            print(f"⚠️  Error in segment length analysis: {e}")

        # Sample segments
        print("\n--- Plotting sample segments ---")
        try:
            plot_sample_segments(
                data["fft_hr"],
                data["compounds"],
                n_samples=4,
                title="Cropped High‑Resolution Spectrum Segments",
                save_path="../figures/sample_segments.png",
            )
        except Exception as e:
            print(f"⚠️  Error plotting sample segments: {e}")

    except FileNotFoundError:
        print(f"⚠️  Training segments file not found: {train_cfg.H5_PATH}")
        print("   Run generate_training_data.py first.")
    except Exception as e:
        print(f"⚠️  Error loading training data: {e}")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ INSPECTION COMPLETE")
    print("=" * 60)
    print("\n📁 Generated figures in ../figures/:")

    expected_files = [
        "fid_example.png",
        "fft_comparison_1.png",
        "segment_lengths_boxplot.png",
        "sample_segments.png",
    ]

    for filename in expected_files:
        filepath = os.path.join("../figures", filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   ✓ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"   ✗ {filename} (not created)")

    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)