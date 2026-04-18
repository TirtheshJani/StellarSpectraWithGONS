#!/usr/bin/env python
"""Render the two interpretability figures from artifact files."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.interpret.lines import ALLOWED_MK_CLASSES
from src.interpret.plotting import plot_per_class_overlay, plot_summary_overlay


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--importance", required=True, type=Path)
    p.add_argument("--shap", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--representative-class", default="G")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    feats = np.load(args.features, allow_pickle=False)
    X = feats["X"]
    y = feats["y"].astype(np.int64)
    wc = feats["wave_centers"]

    imp = np.load(args.importance, allow_pickle=False)["importance_mean"]
    shap_npz = np.load(args.shap, allow_pickle=False)
    mean_abs_per_class = shap_npz["mean_abs_per_class"]

    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]

    rep_cls = args.representative_class
    if rep_cls in class_labels:
        rep_int = present[class_labels.index(rep_cls)]
        rep = np.nanmedian(X[y == rep_int], axis=0)
    else:
        rep = np.nanmedian(X, axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_summary_overlay(
        wc, imp, mean_abs_per_class, class_labels, rep,
        args.out_dir / "importance_overlay.pdf",
    )
    plot_per_class_overlay(
        wc, mean_abs_per_class, class_labels,
        args.out_dir / "importance_overlay_per_class.pdf",
    )
    print(f"wrote overlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
