#!/usr/bin/env python
"""Run permutation importance, SHAP, and occlusion; report 3-way triangulation."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np

from src.interpret.classifier import load_model
from src.interpret.importance import compute_permutation_importance, save_importance
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS
from src.interpret.occlusion import masked_line_ablation, sliding_window_occlusion
from src.interpret.shap_explain import (
    bootstrap_topk_stability,
    compute_shap_values,
    mean_abs_shap_per_class,
    save_shap,
    save_stability,
    stratified_subsample,
)


def _topk(values: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(values)[-k:].tolist())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--shap-max-samples", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    wc = payload["wave_centers"]
    val_idx = payload["val_idx"]
    test_idx = payload["test_idx"]

    model = load_model(args.model)
    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]

    # --- permutation importance on the validation set ---
    imp_mean, imp_std = compute_permutation_importance(
        model, X[val_idx], y[val_idx], n_repeats=10, seed=args.seed,
    )
    save_importance(args.out_dir / "perm_importance.npz", wc, imp_mean, imp_std)

    # --- SHAP on a stratified val subsample ---
    X_shap, y_shap, sub_idx = stratified_subsample(
        X[val_idx], y[val_idx], max_n=args.shap_max_samples, seed=args.seed,
    )
    shap_vals = compute_shap_values(model, X_shap)
    save_shap(args.out_dir / "shap_values.npz", shap_vals, wc, sub_idx, y_shap)
    stability = bootstrap_topk_stability(
        shap_vals, top_k=args.top_k, n_bootstrap=100, seed=args.seed,
    )
    save_stability(args.out_dir / "shap_stability.json", stability, class_labels)
    mabs = mean_abs_shap_per_class(shap_vals)

    # --- sliding-window occlusion on the test set ---
    occ_centers, occ_delta = sliding_window_occlusion(
        model, X[test_idx], y[test_idx], wc, window_aa=50.0, stride_aa=25.0,
    )
    np.savez_compressed(
        args.out_dir / "occlusion.npz",
        window_centers=occ_centers, delta_acc=occ_delta, wave_centers=wc,
    )

    # --- 3-way triangulation (global top-K) ---
    perm_top = _topk(imp_mean, args.top_k)
    shap_top = _topk(mabs.mean(axis=0), args.top_k)
    occ_score = -np.interp(wc, occ_centers, occ_delta)  # deeper drop -> larger score
    occ_top = _topk(occ_score, args.top_k)

    pairs = {
        "perm_vs_shap": _jaccard(perm_top, shap_top),
        "perm_vs_occlusion": _jaccard(perm_top, occ_top),
        "shap_vs_occlusion": _jaccard(shap_top, occ_top),
    }
    three_way = perm_top & shap_top & occ_top
    red_flags = [k for k, v in pairs.items() if v < 0.5]

    triangulation = {
        "top_k": args.top_k,
        "pairwise_jaccard": pairs,
        "three_way_intersection_size": len(three_way),
        "three_way_intersection_bins": sorted(three_way),
        "red_flags": red_flags,
        "class_labels": class_labels,
        "line_sets": list(LINE_SETS.keys()),
    }
    with (args.out_dir / "triangulation_report.json").open("w") as f:
        json.dump(triangulation, f, indent=2)

    # Optional: masked-line ablation driver lives in scripts/ablation.py;
    # run_interpret only produces the triangulation + importance + SHAP + occlusion.

    print("triangulation:", json.dumps(pairs, indent=2))
    if red_flags:
        print("RED FLAGS (pairwise Jaccard < 0.5):", red_flags)


if __name__ == "__main__":
    main()
