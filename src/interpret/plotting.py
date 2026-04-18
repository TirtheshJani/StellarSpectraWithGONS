"""Matplotlib figures for the interpretability audit.

Two figures are produced by ``scripts/make_figure.py``:

  1. ``importance_overlay.pdf``
     Summary: representative normalized spectrum on top, permutation
     importance and per-class mean |SHAP| on the bottom axis, vertical
     dashed lines at MK_LINES entries in the feature window, shaded
     vertical bands at each LINE_SETS ablation window.

  2. ``importance_overlay_per_class.pdf``
     One panel per surviving class. Each panel shows that class's
     mean |SHAP| trace and vertical dashed lines only for MK_LINES
     entries whose ``diag_for`` tuple contains that class. Proves the
     physics-aware-per-class story.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.interpret.lines import LINE_SETS, lines_for_class, lines_in_window


def _normalise_trace(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def plot_summary_overlay(
    wave_centers: np.ndarray,
    importance_mean: np.ndarray,
    shap_per_class: np.ndarray,
    class_labels: list[str],
    representative_spectrum: np.ndarray,
    out_path: Path,
) -> None:
    """Render and save the summary overlay figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_spec, ax_imp) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(10.0, 5.5),
        gridspec_kw={"height_ratios": [1.0, 1.3], "hspace": 0.05},
    )
    ax_spec.plot(wave_centers, representative_spectrum, color="black", lw=0.8)
    ax_spec.set_ylabel("normalized flux")
    ax_spec.set_ylim(
        np.nanpercentile(representative_spectrum, 2) - 0.05,
        np.nanpercentile(representative_spectrum, 99) + 0.05,
    )

    ax_imp.plot(
        wave_centers, _normalise_trace(importance_mean),
        color="black", lw=1.2, label="permutation importance",
    )
    for c, lab in enumerate(class_labels):
        ax_imp.plot(
            wave_centers, _normalise_trace(shap_per_class[c]),
            lw=0.9, alpha=0.55, label=f"|SHAP| ({lab})",
        )
    ax_imp.set_ylabel("importance (normalized)")
    ax_imp.set_xlabel(r"wavelength (\AA)")
    ax_imp.set_ylim(0.0, 1.02)

    wmin, wmax = float(wave_centers.min()), float(wave_centers.max())
    for line in lines_in_window(wmin, wmax):
        for ax in (ax_spec, ax_imp):
            ax.axvline(line.wavelength_aa, color="grey", ls="--", lw=0.5, alpha=0.7)
        ax_imp.text(
            line.wavelength_aa, 1.02, line.name.replace("_", " "),
            rotation=90, ha="right", va="bottom", fontsize=7, color="grey",
        )
    for set_name, windows in LINE_SETS.items():
        for lo, hi in windows:
            if hi < wmin or lo > wmax:
                continue
            ax_imp.axvspan(max(lo, wmin), min(hi, wmax), color="C0", alpha=0.08)

    ax_imp.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_overlay(
    wave_centers: np.ndarray,
    shap_per_class: np.ndarray,
    class_labels: list[str],
    out_path: Path,
) -> None:
    """Render and save the per-class overlay figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_labels)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, sharex=True, figsize=(10.0, 2.0 * n + 0.5),
        gridspec_kw={"hspace": 0.1},
    )
    if n == 1:
        axes = [axes]

    wmin, wmax = float(wave_centers.min()), float(wave_centers.max())
    for c, (ax, label) in enumerate(zip(axes, class_labels)):
        ax.plot(
            wave_centers, _normalise_trace(shap_per_class[c]),
            color=f"C{c}", lw=1.1,
        )
        ax.set_ylabel(f"|SHAP|\n({label})")
        ax.set_ylim(0.0, 1.02)
        class_lines = [
            line for line in lines_for_class(label)
            if wmin <= line.wavelength_aa <= wmax
        ]
        for line in class_lines:
            ax.axvline(line.wavelength_aa, color="grey", ls="--", lw=0.6, alpha=0.8)
            ax.text(
                line.wavelength_aa, 1.02, line.name.replace("_", " "),
                rotation=90, ha="right", va="bottom", fontsize=7, color="grey",
            )
    axes[-1].set_xlabel(r"wavelength (\AA)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
