"""Sliding-window occlusion trace and masked-line ablation with null controls.

Masking replaces flux inside a wavelength window with 1.0 (the
continuum-normalized level), not with zero. This avoids injecting a
synthetic absorption feature where the line used to be.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CONTINUUM_FILL: float = 1.0


def _window_mask(wave_centers: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(wave_centers.shape, dtype=bool)
    for lo, hi in windows:
        m |= (wave_centers >= lo) & (wave_centers <= hi)
    return m


def _apply_mask(X: np.ndarray, mask: np.ndarray, fill: float = CONTINUUM_FILL) -> np.ndarray:
    X_out = X.copy()
    X_out[:, mask] = fill
    return X_out


def sliding_window_occlusion(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    wave_centers: np.ndarray,
    window_aa: float = 50.0,
    stride_aa: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a single-window mask across ``wave_centers`` and report delta_acc.

    Returns (window_centers, delta_acc).  delta_acc is negative when masking
    the window hurts accuracy.
    """
    baseline_acc = float(np.mean(model.predict(X) == y))
    wmin = float(wave_centers.min())
    wmax = float(wave_centers.max())
    centers = np.arange(wmin + window_aa / 2, wmax - window_aa / 2 + 1e-6, stride_aa)
    delta = np.empty(len(centers), dtype=np.float32)
    for i, c in enumerate(centers):
        lo = c - window_aa / 2
        hi = c + window_aa / 2
        mask = (wave_centers >= lo) & (wave_centers <= hi)
        X_masked = _apply_mask(X, mask)
        masked_acc = float(np.mean(model.predict(X_masked) == y))
        delta[i] = masked_acc - baseline_acc
    logger.info(
        "sliding occlusion: %d windows, baseline_acc=%.4f, min_delta=%.4f",
        len(centers), baseline_acc, float(delta.min()),
    )
    return centers.astype(np.float32), delta


@dataclass
class AblationRow:
    line_set: str
    mk_class: str
    n_test: int
    baseline_acc: float
    masked_acc_mean: float
    delta_acc_mean: float
    delta_acc_ci_low: float
    delta_acc_ci_high: float
    p_value_vs_random: float


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    mask = y_true == c
    if not mask.any():
        return np.nan
    return float(np.mean(y_pred[mask] == c))


def _sample_random_windows(
    rng: np.random.Generator,
    wave_centers: np.ndarray,
    total_width_aa: float,
    n_segments: int,
    forbidden_mask: np.ndarray,
) -> list[tuple[float, float]]:
    """Draw ``n_segments`` random windows whose total width matches ``total_width_aa``
    and avoid any bin flagged in ``forbidden_mask``.

    All-or-nothing: returns ``[]`` if the full ``n_segments`` cannot be drawn,
    so a partial sample never enters the null distribution with mismatched
    geometry (Physicist convention: fail-closed on geometry mismatch).
    """
    seg_width = total_width_aa / n_segments
    wmin = float(wave_centers.min())
    wmax = float(wave_centers.max())
    picks: list[tuple[float, float]] = []
    for _ in range(n_segments * 20):
        lo = rng.uniform(wmin, wmax - seg_width)
        hi = lo + seg_width
        m = (wave_centers >= lo) & (wave_centers <= hi)
        if m.any() and not forbidden_mask[m].any():
            picks.append((lo, hi))
            if len(picks) == n_segments:
                return picks
    return []


def masked_line_ablation(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    wave_centers: np.ndarray,
    line_sets: dict[str, list[tuple[float, float]]],
    class_labels: list[str],
    per_class: bool = True,
    n_bootstrap: int = 500,
    n_random_controls: int = 100,
    seed: int = 42,
) -> list[AblationRow]:
    """Bootstrap the accuracy drop from masking each ``line_set``.

    ``p_value_vs_random`` is the fraction of random-window controls whose
    delta is at least as negative as the observed line-set delta
    (one-sided: smaller p -> line set is more damaging than random).
    """
    rng = np.random.default_rng(seed)
    y_pred_base = model.predict(X_test)
    baseline_acc = float(np.mean(y_pred_base == y_test))
    rows: list[AblationRow] = []

    all_line_bins = np.zeros(wave_centers.shape, dtype=bool)
    for windows in line_sets.values():
        all_line_bins |= _window_mask(wave_centers, windows)

    for set_name, windows in line_sets.items():
        line_mask = _window_mask(wave_centers, windows)
        if not line_mask.any():
            logger.warning("line set %s has no overlap with wave_centers - skipping", set_name)
            continue
        total_width = sum(hi - lo for lo, hi in windows)
        n_segments = len(windows)

        X_masked = _apply_mask(X_test, line_mask)
        y_pred_masked = model.predict(X_masked)

        # Random-null distribution: draw n_random_controls non-line windows
        # with matched width/segment-count, outside any MK line window.
        null_deltas: list[float] = []
        for _ in range(n_random_controls):
            picks = _sample_random_windows(
                rng, wave_centers, total_width, n_segments, all_line_bins
            )
            if not picks:
                continue
            m = _window_mask(wave_centers, picks)
            y_pred_null = model.predict(_apply_mask(X_test, m))
            null_deltas.append(
                float(np.mean(y_pred_null == y_test)) - baseline_acc
            )
        null_deltas_arr = np.array(null_deltas, dtype=np.float32)

        iter_classes: list[int] = (
            sorted(np.unique(y_test).tolist()) if per_class else [-1]
        )
        for c in iter_classes:
            if c == -1:
                base_metric = baseline_acc
                masked_metric = float(np.mean(y_pred_masked == y_test))
                label = "ALL"
                n = len(y_test)
            else:
                base_metric = _per_class_recall(y_test, y_pred_base, c)
                masked_metric = _per_class_recall(y_test, y_pred_masked, c)
                label = class_labels[c] if c < len(class_labels) else str(c)
                n = int(np.sum(y_test == c))
            delta = masked_metric - base_metric

            # Bootstrap CI on delta (resample test rows with replacement).
            boot_deltas = np.empty(n_bootstrap, dtype=np.float32)
            for b in range(n_bootstrap):
                idx = rng.integers(0, len(y_test), size=len(y_test))
                if c == -1:
                    bb = float(np.mean(y_pred_base[idx] == y_test[idx]))
                    bm = float(np.mean(y_pred_masked[idx] == y_test[idx]))
                else:
                    yi = y_test[idx]
                    bb = _per_class_recall(yi, y_pred_base[idx], c)
                    bm = _per_class_recall(yi, y_pred_masked[idx], c)
                boot_deltas[b] = (bm - bb) if np.isfinite(bm) and np.isfinite(bb) else np.nan
            boot_valid = boot_deltas[np.isfinite(boot_deltas)]
            if len(boot_valid) == 0:
                ci_lo = ci_hi = np.nan
            else:
                ci_lo = float(np.percentile(boot_valid, 2.5))
                ci_hi = float(np.percentile(boot_valid, 97.5))

            p_val = (
                float(np.mean(null_deltas_arr <= delta))
                if len(null_deltas_arr) else np.nan
            )
            rows.append(AblationRow(
                line_set=set_name,
                mk_class=label,
                n_test=n,
                baseline_acc=float(base_metric),
                masked_acc_mean=float(masked_metric),
                delta_acc_mean=float(delta),
                delta_acc_ci_low=ci_lo,
                delta_acc_ci_high=ci_hi,
                p_value_vs_random=p_val,
            ))
    return rows
