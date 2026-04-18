"""Permutation importance on the validation set."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (importance_mean, importance_std) of shape (n_features,).

    Uses ``sklearn.inspection.permutation_importance`` with accuracy scoring.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs,
        scoring="accuracy",
    )
    logger.info(
        "permutation importance: max=%.4f mean=%.4f (n_features=%d, n_repeats=%d)",
        float(result.importances_mean.max()),
        float(result.importances_mean.mean()),
        X.shape[1],
        n_repeats,
    )
    return (
        result.importances_mean.astype(np.float32),
        result.importances_std.astype(np.float32),
    )


def save_importance(
    out_path: Path,
    wave_centers: np.ndarray,
    importance_mean: np.ndarray,
    importance_std: np.ndarray,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        wave_centers=wave_centers.astype(np.float32),
        importance_mean=importance_mean.astype(np.float32),
        importance_std=importance_std.astype(np.float32),
    )
    logger.info("wrote permutation importance -> %s", out_path)
