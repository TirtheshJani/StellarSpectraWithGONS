"""Build the LightGBM feature matrix from the regridded HDF5.

The feature window is 4800-6800 A and contains Hbeta (4861), Mg b triplet
(5167-5184), Na D doublet (5889-5896), Ca I 6162 and 6439, and Halpha
(6563) (Gray & Corbally 2009). Flux is rebinned from ~0.55 A native
pixels by ``rebin_factor`` (default 5) to ~2.75 A bins (~1000 features).

Imputation uses the TRAIN-set per-bin median (no label leak), following
the conservative convention of The Cannon (Ness+ 2015) and StarNet
(Fabbro+ 2018).

The 90% coverage threshold accommodates the UVES 580 inter-chip gap at
5769-5834 A (~3.25% of the window; Dekker+ 2000, Sacco+ 2014). The S/N
floor of 20 follows the GES DR5.1 QC recommendation (Gilmore+ 2022).

Splits are group-stratified on ``group_col`` when available to prevent
cluster-level leakage across folds (Blanco-Cuaresma 2019).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WAVE_MIN: Final[float] = 4800.0
DEFAULT_WAVE_MAX: Final[float] = 6800.0
DEFAULT_REBIN: Final[int] = 5
DEFAULT_MIN_SNR: Final[float] = 20.0
DEFAULT_MAX_NAN_FRAC: Final[float] = 0.10
DEFAULT_COVERAGE_THRESHOLD: Final[float] = 0.90
DEFAULT_MIN_COVERED_SPECTRA: Final[int] = 2000


@dataclass
class CoverageReport:
    n_total: int
    n_ges: int
    n_covered: int
    threshold: float
    wave_min: float
    wave_max: float


def coverage_probe(
    h5_path: Path,
    wave_min: float = DEFAULT_WAVE_MIN,
    wave_max: float = DEFAULT_WAVE_MAX,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    min_spectra: int = DEFAULT_MIN_COVERED_SPECTRA,
    survey: str = "ges",
) -> CoverageReport:
    """Count spectra meeting the fractional-coverage threshold over the window.

    Raises RuntimeError if fewer than ``min_spectra`` pass the threshold.
    Zero-flux pixels are treated as missing (consistent with the pipeline's
    mask convention).
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        wave = h5["spectra/wavelength"][:]
        flux = h5["spectra/flux"]
        surveys = np.array(h5["metadata/survey"][:], dtype=object)
        surveys = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in surveys],
            dtype=object,
        )

        window_mask = (wave >= wave_min) & (wave <= wave_max)
        if not window_mask.any():
            raise RuntimeError(
                f"no grid points in [{wave_min}, {wave_max}] - "
                "check build_hdf5 resolution settings"
            )

        survey_mask = surveys == survey
        covered = 0
        for i in np.flatnonzero(survey_mask):
            row = flux[i, window_mask]
            valid = np.isfinite(row) & (row != 0.0)
            if float(valid.mean()) >= coverage_threshold:
                covered += 1

    report = CoverageReport(
        n_total=int(len(surveys)),
        n_ges=int(survey_mask.sum()),
        n_covered=covered,
        threshold=coverage_threshold,
        wave_min=wave_min,
        wave_max=wave_max,
    )
    logger.info(
        "coverage probe: %d/%d %s spectra >= %.0f%% coverage of [%g, %g]",
        covered, int(survey_mask.sum()), survey,
        100 * coverage_threshold, wave_min, wave_max,
    )
    if covered < min_spectra:
        raise RuntimeError(
            f"only {covered} {survey} spectra meet coverage >= "
            f"{coverage_threshold:.2f} of [{wave_min}, {wave_max}] "
            f"(need {min_spectra}); abort."
        )
    return report


def rebin_flux(
    flux_2d: np.ndarray,
    wave_native: np.ndarray,
    wave_min: float = DEFAULT_WAVE_MIN,
    wave_max: float = DEFAULT_WAVE_MAX,
    rebin_factor: int = DEFAULT_REBIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice to [wave_min, wave_max] and rebin by averaging ``rebin_factor`` pixels.

    A rebinned bin is set to NaN if at least half its native pixels are
    non-finite. ``wave_centers`` is the mean native wavelength per block.
    Zero-valued pixels are not treated as missing here -- use the coverage
    probe or the ``max_nan_frac`` cut in ``build_features`` for masking.
    """
    if rebin_factor < 1:
        raise ValueError(f"rebin_factor must be >= 1, got {rebin_factor}")
    window = (wave_native >= wave_min) & (wave_native <= wave_max)
    w = wave_native[window]
    f = flux_2d[:, window]
    n_pix = f.shape[1]
    n_bins = n_pix // rebin_factor
    if n_bins == 0:
        raise RuntimeError(
            f"rebin produces zero bins: {n_pix} native pixels / {rebin_factor}"
        )
    truncated = n_bins * rebin_factor
    w = w[:truncated]
    f = f[:, :truncated]

    f_blocks = f.reshape(f.shape[0], n_bins, rebin_factor)
    w_blocks = w.reshape(n_bins, rebin_factor)

    finite = np.isfinite(f_blocks)
    n_finite = finite.sum(axis=2)
    f_safe = np.where(finite, f_blocks, 0.0)
    summed = f_safe.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_finite > 0, summed / n_finite, np.nan)
    half_or_fewer_finite = n_finite < (rebin_factor // 2 + 1)
    mean[half_or_fewer_finite] = np.nan

    wave_centers = w_blocks.mean(axis=1)
    return mean.astype(np.float32), wave_centers.astype(np.float32)


def fit_median_imputer(X_train: np.ndarray) -> np.ndarray:
    """Per-column median of ``X_train`` ignoring NaN. Empty columns fall back to 1.0."""
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(X_train, axis=0)
    med[~np.isfinite(med)] = 1.0
    return med.astype(np.float32)


def apply_median_imputer(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Replace NaN entries of ``X`` with the corresponding ``med[column]``."""
    mask = ~np.isfinite(X)
    if not mask.any():
        return X
    X_out = X.copy()
    col_idx = np.where(mask)[1]
    X_out[mask] = med[col_idx]
    return X_out


def _split_indices(
    y: np.ndarray,
    groups: np.ndarray | None,
    frac_train: float,
    frac_val: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

    n = len(y)
    idx_all = np.arange(n)
    if groups is not None:
        gss1 = GroupShuffleSplit(
            n_splits=1, test_size=1.0 - frac_train, random_state=seed
        )
        train_idx, rest_idx = next(gss1.split(idx_all, y, groups=groups))
        rel = frac_val / (1.0 - frac_train)
        gss2 = GroupShuffleSplit(
            n_splits=1, test_size=1.0 - rel, random_state=seed + 1
        )
        val_rel, test_rel = next(
            gss2.split(rest_idx, y[rest_idx], groups=groups[rest_idx])
        )
        val_idx = rest_idx[val_rel]
        test_idx = rest_idx[test_rel]
    else:
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - frac_train, random_state=seed
        )
        train_idx, rest_idx = next(sss1.split(idx_all, y))
        rel = frac_val / (1.0 - frac_train)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - rel, random_state=seed + 1
        )
        val_rel, test_rel = next(sss2.split(rest_idx, y[rest_idx]))
        val_idx = rest_idx[val_rel]
        test_idx = rest_idx[test_rel]
    return train_idx, val_idx, test_idx


def build_features(
    h5_path: Path,
    labels_parquet: Path,
    out_path: Path,
    wave_min: float = DEFAULT_WAVE_MIN,
    wave_max: float = DEFAULT_WAVE_MAX,
    rebin: int = DEFAULT_REBIN,
    min_snr: float = DEFAULT_MIN_SNR,
    max_nan_frac: float = DEFAULT_MAX_NAN_FRAC,
    max_spectra: int | None = 5000,
    group_col: str | None = None,
    frac_train: float = 0.70,
    frac_val: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Write an .npz with X, y, wave_centers, splits, and group metadata.

    Returns the same payload as a dict. ``groups`` is zero-filled when
    ``group_col`` is absent from the labels parquet; in that case the
    split falls back to a class-stratified shuffle with a logged warning.
    """
    h5_path = Path(h5_path)
    labels_parquet = Path(labels_parquet)
    out_path = Path(out_path)
    rng = np.random.default_rng(seed)

    labels = pd.read_parquet(labels_parquet)
    if len(labels) == 0:
        raise RuntimeError(f"labels parquet is empty: {labels_parquet}")

    with h5py.File(h5_path, "r") as h5:
        wave = h5["spectra/wavelength"][:]
        flux = h5["spectra/flux"]
        source_files = np.array(h5["metadata/source_file"][:], dtype=object)
        source_files = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in source_files],
            dtype=object,
        )
        snr = np.array(h5["metadata/snr_median"][:], dtype=float)

        sf_to_row = {sf: i for i, sf in enumerate(source_files)}
        rows = np.array(
            [sf_to_row.get(sf, -1) for sf in labels["source_file"].to_numpy()],
            dtype=int,
        )
        if np.any(rows < 0):
            missing = int((rows < 0).sum())
            logger.warning("dropping %d labelled rows missing from HDF5", missing)
            keep = rows >= 0
            labels = labels.loc[keep].reset_index(drop=True)
            rows = rows[keep]

        snr_ok = snr[rows] >= min_snr
        logger.info("S/N >= %.0f keeps %d/%d spectra", min_snr, int(snr_ok.sum()), len(rows))
        labels = labels.loc[snr_ok].reset_index(drop=True)
        rows = rows[snr_ok]

        flux_rows = np.empty((len(rows), wave.shape[0]), dtype=np.float32)
        for k, r in enumerate(rows):
            flux_rows[k, :] = flux[r, :]

    X, wave_centers = rebin_flux(
        flux_rows, wave, wave_min=wave_min, wave_max=wave_max, rebin_factor=rebin
    )

    nan_frac = np.mean(~np.isfinite(X), axis=1)
    keep = nan_frac <= max_nan_frac
    logger.info(
        "NaN-frac <= %.2f keeps %d/%d spectra", max_nan_frac, int(keep.sum()), len(X)
    )
    X = X[keep]
    labels = labels.loc[keep].reset_index(drop=True)

    if max_spectra is not None and len(X) > max_spectra:
        n_classes = labels["mk_class"].nunique()
        per_class = max_spectra // max(n_classes, 1)
        take: list[int] = []
        for _, grp in labels.groupby("mk_class"):
            ids = grp.index.to_numpy()
            if len(ids) > per_class:
                take.extend(rng.choice(ids, size=per_class, replace=False).tolist())
            else:
                take.extend(ids.tolist())
        take_arr = np.array(sorted(take))
        X = X[take_arr]
        labels = labels.loc[take_arr].reset_index(drop=True)
        logger.info("stratified subsample to %d spectra (target %d)", len(X), max_spectra)

    y = labels["mk_int"].to_numpy().astype(np.int8)
    boundary = labels["boundary_distance_k"].to_numpy().astype(np.float32)
    dwarf = labels["dwarf_flag"].to_numpy().astype(bool)

    groups: np.ndarray | None = None
    if group_col and group_col in labels.columns:
        groups = labels[group_col].to_numpy()
        logger.info("group column %r unique values: %d", group_col, len(np.unique(groups)))
    elif group_col:
        logger.warning(
            "group_col=%r not found in labels; falling back to stratified split - "
            "cluster-level leakage possible",
            group_col,
        )

    train_idx, val_idx, test_idx = _split_indices(
        y, groups, frac_train=frac_train, frac_val=frac_val, seed=seed
    )

    median_imp = fit_median_imputer(X[train_idx])
    X = apply_median_imputer(X, median_imp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": X,
        "y": y,
        "wave_centers": wave_centers,
        "boundary_distance_k": boundary,
        "dwarf_flag": dwarf,
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "groups": (groups if groups is not None else np.zeros(len(y), dtype=np.int64)),
        "median_imputer": median_imp,
    }
    np.savez_compressed(out_path, **payload)
    logger.info("wrote %s (n=%d, n_bins=%d)", out_path, len(X), X.shape[1])
    return payload
