"""Peak detection on permutation importance and matching against MK_LINES.

The audit pattern:
    1. Detect peaks in the 1-D importance trace.
    2. Match each peak to the nearest registry line within ``tolerance_aa``.
    3. Report precision, recall, and Jaccard vs the in-window MK_LINES.

We sweep tolerance over {1, 2, 5, 10} A so downstream figures can show how
the match rate scales with tolerance -- a classifier that learns real lines
should match at 2 A already; one that relies on broad continuum shape only
starts matching at 10 A by coincidence.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from src.interpret.lines import MK_LINES, Line, lines_in_window

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCES_AA: Final[tuple[float, ...]] = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Peak:
    wavelength_aa: float
    height: float


@dataclass(frozen=True)
class MatchMetrics:
    tolerance_aa: float
    n_peaks: int
    n_lines_in_window: int
    n_matched_lines: int
    precision: float
    recall: float
    jaccard: float


def detect_peaks(
    importance: np.ndarray,
    wave_centers: np.ndarray,
    prominence: float | None = None,
    min_distance_aa: float = 2.0,
) -> list[Peak]:
    """Return peaks of the importance trace, sorted descending by height.

    ``prominence`` defaults to 1 median absolute deviation above the median.
    """
    from scipy.signal import find_peaks

    if prominence is None:
        med = float(np.median(importance))
        mad = float(np.median(np.abs(importance - med)))
        prominence = med + mad
    dx = float(np.median(np.diff(wave_centers)))
    distance_bins = max(1, int(round(min_distance_aa / max(dx, 1e-6))))

    idx, props = find_peaks(importance, prominence=prominence, distance=distance_bins)
    order = np.argsort(importance[idx])[::-1]
    peaks: list[Peak] = [
        Peak(
            wavelength_aa=float(wave_centers[i]),
            height=float(importance[i]),
        )
        for i in idx[order]
    ]
    logger.info(
        "detected %d peaks (prominence=%.4g, min_distance=%.1f A)",
        len(peaks), prominence, min_distance_aa,
    )
    return peaks


def match_peaks_to_lines(
    peaks: list[Peak], lines: list[Line], tolerance_aa: float,
) -> dict[str, list]:
    """Greedy nearest-neighbour matching of peaks to lines within tolerance.

    Each line and each peak can be matched at most once. Returns a dict with
    lists of matched pairs, unmatched peaks, and unmatched lines.
    """
    if not peaks or not lines:
        return {
            "matched": [],
            "unmatched_peaks": list(peaks),
            "unmatched_lines": list(lines),
        }
    peak_waves = np.array([p.wavelength_aa for p in peaks])
    line_waves = np.array([l.wavelength_aa for l in lines])
    peak_used = np.zeros(len(peaks), dtype=bool)
    line_used = np.zeros(len(lines), dtype=bool)
    matched: list[tuple[Line, Peak, float]] = []
    # Iterate peaks by descending height (already sorted by detect_peaks).
    for pi, pw in enumerate(peak_waves):
        if peak_used[pi]:
            continue
        dists = np.abs(line_waves - pw)
        dists[line_used] = np.inf
        li = int(np.argmin(dists))
        if dists[li] <= tolerance_aa:
            matched.append((lines[li], peaks[pi], float(dists[li])))
            line_used[li] = True
            peak_used[pi] = True
    return {
        "matched": matched,
        "unmatched_peaks": [peaks[i] for i in np.flatnonzero(~peak_used)],
        "unmatched_lines": [lines[i] for i in np.flatnonzero(~line_used)],
    }


def compute_match_metrics(
    matches: dict[str, list], n_peaks: int, n_lines: int, tolerance_aa: float,
) -> MatchMetrics:
    n_matched = len(matches["matched"])
    precision = n_matched / n_peaks if n_peaks else 0.0
    recall = n_matched / n_lines if n_lines else 0.0
    union = n_peaks + n_lines - n_matched
    jaccard = n_matched / union if union else 0.0
    return MatchMetrics(
        tolerance_aa=tolerance_aa,
        n_peaks=n_peaks,
        n_lines_in_window=n_lines,
        n_matched_lines=n_matched,
        precision=precision,
        recall=recall,
        jaccard=jaccard,
    )


def sweep_tolerances(
    importance: np.ndarray,
    wave_centers: np.ndarray,
    tolerances_aa: tuple[float, ...] = DEFAULT_TOLERANCES_AA,
    prominence: float | None = None,
) -> list[MatchMetrics]:
    """Detect peaks once, then score at each tolerance."""
    peaks = detect_peaks(importance, wave_centers, prominence=prominence)
    lines_in = lines_in_window(float(wave_centers.min()), float(wave_centers.max()))
    results: list[MatchMetrics] = []
    for tol in tolerances_aa:
        m = match_peaks_to_lines(peaks, lines_in, tolerance_aa=tol)
        results.append(compute_match_metrics(m, len(peaks), len(lines_in), tol))
    return results


def save_sweep(
    csv_path: Path, json_path: Path, results: list[MatchMetrics],
) -> None:
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "tolerance_aa", "n_peaks", "n_lines_in_window", "n_matched_lines",
        "precision", "recall", "jaccard",
    ]
    lines_out = [",".join(header)]
    for r in results:
        lines_out.append(
            f"{r.tolerance_aa},{r.n_peaks},{r.n_lines_in_window},"
            f"{r.n_matched_lines},{r.precision:.4f},{r.recall:.4f},{r.jaccard:.4f}"
        )
    csv_path.write_text("\n".join(lines_out) + "\n")

    summary = {
        "tolerances_aa": [r.tolerance_aa for r in results],
        "recall": {r.tolerance_aa: r.recall for r in results},
        "precision": {r.tolerance_aa: r.precision for r in results},
        "jaccard": {r.tolerance_aa: r.jaccard for r in results},
        "n_lines_in_window": results[0].n_lines_in_window if results else 0,
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote line-match sweep -> %s, %s", csv_path, json_path)
