"""SHAP TreeExplainer for the MK LightGBM classifier, with bootstrap ranking stability.

The bootstrap resamples the SHAP sample with replacement 100 times and
recomputes top-K rankings per class. Mean pairwise Jaccard of the top-K
sets is the stability metric reported in artifacts/shap_stability.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_n: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_sub, y_sub, idx) with at most ``max_n`` rows, class-balanced."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_n // len(classes))
    take: list[int] = []
    for c in classes:
        ids = np.flatnonzero(y == c)
        if len(ids) > per_class:
            take.extend(rng.choice(ids, size=per_class, replace=False).tolist())
        else:
            take.extend(ids.tolist())
    idx = np.array(sorted(take))
    return X[idx], y[idx], idx


def compute_shap_values(
    model: Any, X_sample: np.ndarray
) -> np.ndarray:
    """Return SHAP values shaped (n_classes, n_samples, n_features).

    TreeExplainer returns a list for multiclass; we stack it to a 3-D array.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        arr = np.stack(sv, axis=0)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3 and arr.shape[0] == X_sample.shape[0]:
            # newer shap: (n_samples, n_features, n_classes) -> transpose
            arr = np.transpose(arr, (2, 0, 1))
    logger.info("shap values shape: %s", arr.shape)
    return arr.astype(np.float32)


def mean_abs_shap_per_class(shap_values: np.ndarray) -> np.ndarray:
    """Collapse samples -> mean |SHAP|. Shape: (n_classes, n_features)."""
    return np.mean(np.abs(shap_values), axis=1).astype(np.float32)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def bootstrap_topk_stability(
    shap_values: np.ndarray,
    top_k: int = 20,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict[int, float]:
    """Mean pairwise Jaccard of top-K bins per class across bootstrap resamples."""
    rng = np.random.default_rng(seed)
    n_classes, n_samples, _n_feat = shap_values.shape
    out: dict[int, float] = {}
    for c in range(n_classes):
        topk_sets: list[set[int]] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_samples, size=n_samples)
            mabs = np.mean(np.abs(shap_values[c, idx, :]), axis=0)
            top = set(np.argsort(mabs)[-top_k:].tolist())
            topk_sets.append(top)
        pairs = [
            jaccard(topk_sets[i], topk_sets[j])
            for i in range(len(topk_sets))
            for j in range(i + 1, len(topk_sets))
        ]
        out[c] = float(np.mean(pairs)) if pairs else 1.0
    return out


def save_shap(
    out_path: Path,
    shap_values: np.ndarray,
    wave_centers: np.ndarray,
    sample_idx: np.ndarray,
    y_sample: np.ndarray,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        shap_values=shap_values.astype(np.float32),
        mean_abs_per_class=mean_abs_shap_per_class(shap_values),
        wave_centers=wave_centers.astype(np.float32),
        sample_idx=sample_idx.astype(np.int64),
        y_sample=y_sample.astype(np.int8),
    )
    logger.info("wrote SHAP values -> %s", out_path)


def save_stability(out_path: Path, stability: dict[int, float], class_labels: list[str]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "top_k": 20,
        "metric": "mean_pairwise_jaccard",
        "per_class": {class_labels[c]: v for c, v in stability.items()},
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote SHAP stability -> %s", out_path)
