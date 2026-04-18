"""Pickles (1998) template-matching benchmark for the MK classifier.

Per test spectrum we resample each Pickles UVKLIB template onto the feature
grid and pick the template with the minimum chi-squared to the (continuum-
normalized) spectrum. The template's MK class (parsed from its filename via
the embedded lookup) is the "Pickles-MK" label; we compare that against the
LightGBM prediction on the same spectrum.

This is a reality check against an external library, not a training signal.
Pickles spans O–M; we collapse everything outside A/F/G/K to "OTHER".

Catalog: Pickles 1998, PASP 110, 863 (VizieR J/PASP/110/863).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

# Filename -> MK type for the UVKLIB mapping (Pickles 1998 Table 3).
# The UVKLIB naming convention is "uk{N}.fits"; this dict covers the 131
# UVKLIB entries by number. Values are the canonical MK class string from
# the Pickles README.
PICKLES_UVKLIB_MAP: Final[dict[int, str]] = {
    1:  "O5V",  2:  "O9V",  3:  "B0V",  4:  "B1V",  5:  "B3V",
    6:  "B57V", 7:  "B8V",  8:  "B9V",  9:  "A0V",  10: "A2V",
    11: "A3V",  12: "A5V",  13: "A7V",  14: "F0V",  15: "F2V",
    16: "F5V",  17: "F6V",  18: "F8V",  19: "G0V",  20: "G2V",
    21: "G5V",  22: "G8V",  23: "K0V",  24: "K2V",  25: "K3V",
    26: "K4V",  27: "K5V",  28: "K7V",  29: "M0V",  30: "M1V",
    31: "M2V",  32: "M2.5V", 33: "M3V", 34: "M4V",  35: "M5V",
    36: "M6V",  37: "B2IV", 38: "B6IV", 39: "A0IV", 40: "A47IV",
    41: "F02IV", 42: "F5IV", 43: "F8IV", 44: "G0IV", 45: "G2IV",
    46: "G5IV",  47: "G8IV", 48: "K0IV", 49: "K1IV", 50: "K3IV",
    51: "O8III", 52: "B12III", 53: "B3III", 54: "B5III", 55: "B9III",
    56: "A0III", 57: "A3III", 58: "A5III", 59: "A7III", 60: "F0III",
    61: "F2III", 62: "F5III", 63: "G0III", 64: "G5III", 65: "G8III",
    66: "K0III", 67: "K1III", 68: "K2III", 69: "K3III", 70: "K4III",
    71: "K5III", 72: "M0III", 73: "M1III", 74: "M2III", 75: "M3III",
    76: "M4III", 77: "M5III", 78: "M6III", 79: "M7III", 80: "M8III",
    81: "M9III", 82: "M10III", 83: "B2II",  84: "B5II",  85: "F0II",
    86: "F2II",  87: "G5II",  88: "K01II", 89: "K34II", 90: "M3II",
    91: "B0I",   92: "B1I",   93: "B3I",   94: "B5I",   95: "B8I",
    96: "A0I",   97: "A2I",   98: "F0I",   99: "F5I",   100: "F8I",
    101: "G0I",  102: "G2I",  103: "G5I",  104: "G8I",  105: "K2I",
    106: "K3I",  107: "K4I",  108: "M2I",
}

_FN_RE = re.compile(r"(?:pickles_)?uk_?(?P<n>\d+)\.(fits|dat|txt)", re.IGNORECASE)


def parse_pickles_filename(filename: str) -> str:
    """Return the MK type string for a Pickles UVKLIB filename."""
    name = Path(filename).name
    m = _FN_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"{name!r} does not match Pickles UVKLIB naming")
    n = int(m.group("n"))
    if n not in PICKLES_UVKLIB_MAP:
        raise KeyError(f"Pickles index {n} not in UVKLIB map (1..108)")
    return PICKLES_UVKLIB_MAP[n]


def collapse_to_mk(type_str: str) -> str:
    """Map a full MK type (e.g. 'G2V', 'K3III') to the A/F/G/K/OTHER coarse label."""
    if not type_str:
        return "OTHER"
    letter = type_str[0].upper()
    if letter in {"A", "F", "G", "K"}:
        return letter
    return "OTHER"


@dataclass
class Template:
    filename: str
    mk_type: str
    mk_class: str
    flux_on_grid: np.ndarray  # continuum-normalized, shape (n_bins,)


def load_template_fits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Pickles FITS template and return (wave_aa, flux)."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
        if data.dtype.names is not None:
            names = [n.upper() for n in data.dtype.names]
            wave = np.asarray(data[data.dtype.names[names.index("WAVELENGTH")]], dtype=float)
            flux = np.asarray(data[data.dtype.names[names.index("FLUX")]], dtype=float)
        else:
            # linear WCS fallback
            h = hdul[0].header
            crval1 = float(h["CRVAL1"])
            cdelt1 = float(h.get("CDELT1", h.get("CD1_1", 1.0)))
            naxis1 = int(h["NAXIS1"])
            wave = crval1 + cdelt1 * np.arange(naxis1)
            flux = np.asarray(data, dtype=float)
    return wave, flux


def continuum_normalize(wave: np.ndarray, flux: np.ndarray, window_aa: float = 200.0) -> np.ndarray:
    """Cheap polynomial-free continuum normalization by median-filtering on a wide window."""
    from scipy.ndimage import median_filter

    dx = float(np.median(np.diff(wave)))
    size = max(3, int(round(window_aa / max(dx, 1e-6))))
    if size % 2 == 0:
        size += 1
    cont = median_filter(flux, size=size)
    cont = np.where(cont > 0, cont, np.nan)
    return (flux / cont).astype(np.float32)


def load_pickles_library(
    pickles_dir: Path, wave_centers: np.ndarray,
) -> list[Template]:
    """Load all UVKLIB FITS templates from ``pickles_dir`` and resample to ``wave_centers``."""
    pickles_dir = Path(pickles_dir)
    files = sorted(pickles_dir.glob("uk*.fits")) + sorted(pickles_dir.glob("pickles_uk*.fits"))
    if not files:
        raise RuntimeError(f"no Pickles FITS templates found in {pickles_dir}")
    templates: list[Template] = []
    for path in files:
        try:
            mk_type = parse_pickles_filename(path.name)
        except (ValueError, KeyError) as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            continue
        wave, flux = load_template_fits(path)
        flux_norm = continuum_normalize(wave, flux)
        flux_on_grid = np.interp(
            wave_centers, wave, flux_norm, left=np.nan, right=np.nan,
        ).astype(np.float32)
        templates.append(Template(
            filename=path.name,
            mk_type=mk_type,
            mk_class=collapse_to_mk(mk_type),
            flux_on_grid=flux_on_grid,
        ))
    logger.info("loaded %d Pickles templates on %d-bin grid", len(templates), len(wave_centers))
    return templates


def best_template_per_spectrum(
    X_test: np.ndarray, templates: list[Template],
) -> np.ndarray:
    """Return an array of length ``len(X_test)`` with the best template index.

    Uses chi-squared assuming unit variance (features are continuum-normalized
    and rebinned, so per-bin variance ~ 1% in the noise regime we care about).
    Templates are restricted to bins where the template is finite for each
    individual spectrum (no global mask).
    """
    T = np.stack([t.flux_on_grid for t in templates], axis=0)
    finite_mask = np.isfinite(T) & np.isfinite(X_test)[:, None, :]
    chi2 = np.full((X_test.shape[0], T.shape[0]), np.inf, dtype=np.float32)
    for i in range(X_test.shape[0]):
        m = finite_mask[i]
        diffs = T - X_test[i][None, :]
        diffs[~m] = 0.0
        counts = m.sum(axis=1)
        sq = (diffs ** 2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2[i] = np.where(counts > 0, sq / counts, np.inf)
    return np.argmin(chi2, axis=1).astype(np.int32)


def benchmark_report(
    y_pred_model: np.ndarray,
    y_pickles_mk: np.ndarray,
    class_labels: list[str],
) -> dict:
    """Confusion + agreement between classifier and Pickles-MK labels.

    Both inputs must be A/F/G/K/OTHER strings of the same length.
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    all_labels = list(class_labels) + ["OTHER"]
    cm = confusion_matrix(y_pickles_mk, y_pred_model, labels=all_labels).tolist()
    p, r, f, _ = precision_recall_fscore_support(
        y_pickles_mk, y_pred_model, labels=all_labels, average=None, zero_division=0,
    )
    macro_f1 = float(f1_score(
        y_pickles_mk, y_pred_model, labels=class_labels, average="macro", zero_division=0,
    ))
    agreement = float(np.mean(np.asarray(y_pickles_mk) == np.asarray(y_pred_model)))
    return {
        "labels": all_labels,
        "confusion_matrix": cm,
        "per_class_precision": {all_labels[i]: float(p[i]) for i in range(len(all_labels))},
        "per_class_recall": {all_labels[i]: float(r[i]) for i in range(len(all_labels))},
        "per_class_f1": {all_labels[i]: float(f[i]) for i in range(len(all_labels))},
        "macro_f1_fgk": macro_f1,
        "agreement_rate": agreement,
        "n_compared": int(len(y_pickles_mk)),
    }


def save_report(out_dir: Path, report: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    confusion = report["confusion_matrix"]
    labels = report["labels"]
    csv_lines = ["," + ",".join(labels)]
    for label, row in zip(labels, confusion):
        csv_lines.append(f"{label}," + ",".join(str(v) for v in row))
    (out_dir / "benchmark_confusion.csv").write_text("\n".join(csv_lines) + "\n")
    with (out_dir / "benchmark_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote benchmark confusion + report -> %s", out_dir)
