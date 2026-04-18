"""Build MK-class labels for Gaia-ESO UVES spectra.

Labels come from the public GES DR5.1 recommended-parameters catalog
(Hourihane et al. 2023, A&A 666, A121, VizieR J/A+A/666/A121,
bibcode 2023A&A...666A.121H). The catalog is fetched via astroquery
on first use and cached locally.

MK Teff bin edges follow Pecaut & Mamajek (2013, Table 5), rounded to the
nearest 100 K:

  A : [7300, 10000)
  F : [6000,  7300)
  G : [5300,  6000)
  K : [3900,  5300)

O/B/M are excluded at label-construction time per the interpretability plan:
GES UVES is FGK-targeted, and the classifier is trained only on A/F/G/K.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import pandas as pd

from src.interpret.lines import ALLOWED_MK_CLASSES

logger = logging.getLogger(__name__)

VIZIER_CATALOG: Final[str] = "J/A+A/666/A121"
VIZIER_BIBCODE: Final[str] = "2023A&A...666A.121H"

MK_BIN_EDGES: Final[dict[str, tuple[float, float]]] = {
    "A": (7300.0, 10000.0),
    "F": (6000.0,  7300.0),
    "G": (5300.0,  6000.0),
    "K": (3900.0,  5300.0),
}

MK_INT: Final[dict[str, int]] = {c: i for i, c in enumerate(ALLOWED_MK_CLASSES)}

_FILENAME_RE = re.compile(
    r"ges_uves_(?P<ra>-?\d+\.\d+)_(?P<dec>-?\d+\.\d+)\.fits",
    re.IGNORECASE,
)


def parse_ra_dec_from_filename(source_file: str | Path) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) parsed from the GES UVES filename convention.

    The spectrum fetcher (src/fetch/fetch_ges.py) writes files as
    ``ges_uves_{RA:.6f}_{Dec:.6f}.fits``, so the pointing RA/Dec is
    recoverable from the filename. Raises ValueError if the filename
    does not match.
    """
    name = Path(source_file).name
    m = _FILENAME_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"filename {name!r} does not match GES UVES convention")
    return float(m.group("ra")), float(m.group("dec"))


def bin_teff_to_mk(teff: np.ndarray) -> np.ndarray:
    """Bin Teff (K) to MK class strings. Values outside FGK range -> 'OTHER'."""
    teff = np.asarray(teff, dtype=float)
    out = np.full(teff.shape, "OTHER", dtype=object)
    for cls, (lo, hi) in MK_BIN_EDGES.items():
        mask = (teff >= lo) & (teff < hi)
        out[mask] = cls
    return out


def compute_boundary_distance_k(
    teff: np.ndarray, mk_class: np.ndarray
) -> np.ndarray:
    """Distance in K from each Teff to the nearest edge of its own MK bin.

    Returns NaN for rows labelled 'OTHER' (no native bin).
    """
    teff = np.asarray(teff, dtype=float)
    mk_class = np.asarray(mk_class, dtype=object)
    dist = np.full(teff.shape, np.nan, dtype=float)
    for cls, (lo, hi) in MK_BIN_EDGES.items():
        mask = mk_class == cls
        if not np.any(mask):
            continue
        dist[mask] = np.minimum(np.abs(teff[mask] - lo), np.abs(teff[mask] - hi))
    return dist


_VIZIER_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "ra":   ("RA_ICRS", "RAJ2000", "_RAJ2000", "RAdeg", "RA_d"),
    "dec":  ("DE_ICRS", "DEJ2000", "_DEJ2000", "DEdeg", "DE_d"),
    "teff": ("Teff", "TEFF", "Teff_rec", "Teff_ges"),
    "logg": ("logg", "LOGG", "logg_rec"),
    "feh":  ("[Fe/H]", "FeH", "Fe_H_", "feh_rec"),
}


def _resolve_column_aliases(columns) -> dict[str, str]:
    """Map canonical short names to actual VizieR column names (case-insensitive)."""
    cols_lower = {c.lower(): c for c in columns}
    resolved: dict[str, str] = {}
    for key, candidates in _VIZIER_ALIASES.items():
        match = next(
            (cols_lower[a.lower()] for a in candidates if a.lower() in cols_lower),
            None,
        )
        if match is None:
            raise KeyError(
                f"no VizieR column found for {key!r} (tried {candidates}; "
                f"columns present: {sorted(columns)})"
            )
        resolved[key] = match
    return resolved


def fetch_ges_params_catalog(cache_dir: Path) -> pd.DataFrame:
    """Fetch (and cache) the GES DR5.1 recommended-parameters catalog.

    Returns a DataFrame with columns ``ra_deg, dec_deg, teff_k, logg, feh``.
    On first call, queries VizieR ``J/A+A/666/A121`` via astroquery and
    writes ``cache_dir/ges_dr5_params.parquet``. Subsequent calls reuse it.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ges_dr5_params.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info("loaded cached GES params (%d rows) from %s", len(df), cache_path)
        return df

    from astroquery.vizier import Vizier  # lazy import so tests avoid the hard dep

    Vizier.ROW_LIMIT = -1
    logger.info("fetching GES params from VizieR %s ...", VIZIER_CATALOG)
    table_list = Vizier.get_catalogs(VIZIER_CATALOG)
    if len(table_list) == 0:
        raise RuntimeError(f"VizieR returned no tables for {VIZIER_CATALOG}")
    table = table_list[0].to_pandas()

    col_map = _resolve_column_aliases(table.columns)
    df = pd.DataFrame({
        "ra_deg":  table[col_map["ra"]].astype(float),
        "dec_deg": table[col_map["dec"]].astype(float),
        "teff_k":  table[col_map["teff"]].astype(float),
        "logg":    table[col_map["logg"]].astype(float),
        "feh":     table[col_map["feh"]].astype(float),
    }).dropna()

    df.to_parquet(cache_path, index=False)
    logger.info(
        "cached GES params (%d rows) -> %s  [bibcode %s]",
        len(df), cache_path, VIZIER_BIBCODE,
    )
    return df


@dataclass
class LabelStats:
    n_input: int
    n_unmatched: int
    n_other: int
    n_final: int
    per_class: dict[str, int]


def build_labels(
    h5_path: Path,
    cache_dir: Path,
    match_radius_arcsec: float = 0.5,
    ambiguity_radius_arcsec: float = 2.0,
    min_per_class: int = 50,
    warn_per_class: int = 200,
    allow_drop_underfilled: bool = False,
) -> tuple[pd.DataFrame, LabelStats]:
    """Build MK-class labels for the GES UVES spectra in ``h5_path``.

    Parameters
    ----------
    h5_path : regridded HDF5 from build_hdf5.py.
    cache_dir : directory for the VizieR catalog cache.
    match_radius_arcsec : position cross-match tolerance (default 0.5 arcsec
        per Gaia DR3 astrometric accuracy; see physicist review).
    ambiguity_radius_arcsec : second-nearest-neighbour distance below which a
        match is treated as ambiguous and dropped.
    min_per_class : hard RuntimeError floor per surviving MK class.
    warn_per_class : emit logger.warning below this count.
    allow_drop_underfilled : if True and a class falls below min_per_class,
        drop the class rather than raising (use for A in FGK-heavy samples).

    Returns
    -------
    (labels_df, stats)
        labels_df columns: source_file, ra_deg, dec_deg, teff_k, logg, feh,
        mk_class, mk_int, boundary_distance_k, dwarf_flag.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    h5_path = Path(h5_path)
    cache_dir = Path(cache_dir)

    with h5py.File(h5_path, "r") as h5:
        source_files = np.array(h5["metadata/source_file"][:], dtype=object)
        surveys = np.array(h5["metadata/survey"][:], dtype=object)
    source_files = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in source_files],
        dtype=object,
    )
    surveys = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in surveys],
        dtype=object,
    )

    ges_mask = surveys == "ges"
    ges_files = source_files[ges_mask]
    if len(ges_files) == 0:
        raise RuntimeError(f"no GES spectra found in {h5_path}")

    ras = np.empty(len(ges_files), dtype=float)
    decs = np.empty(len(ges_files), dtype=float)
    for i, fname in enumerate(ges_files):
        ras[i], decs[i] = parse_ra_dec_from_filename(fname)

    cat = fetch_ges_params_catalog(cache_dir)
    spec_coord = SkyCoord(ras * u.deg, decs * u.deg)
    cat_coord = SkyCoord(
        cat["ra_deg"].to_numpy() * u.deg,
        cat["dec_deg"].to_numpy() * u.deg,
    )

    idx, d2d, _ = spec_coord.match_to_catalog_sky(cat_coord)
    sep_arcsec = d2d.to(u.arcsec).value
    matched = sep_arcsec <= match_radius_arcsec

    _, d2d2, _ = spec_coord.match_to_catalog_sky(cat_coord, nthneighbor=2)
    amb = d2d2.to(u.arcsec).value < ambiguity_radius_arcsec
    matched = matched & ~amb

    n_unmatched = int((~matched).sum())
    if n_unmatched:
        logger.info(
            "dropped %d/%d GES spectra with no unambiguous match within %.2f arcsec",
            n_unmatched, len(ges_files), match_radius_arcsec,
        )

    teff_raw = cat["teff_k"].to_numpy()[idx]
    logg_raw = cat["logg"].to_numpy()[idx]
    feh_raw = cat["feh"].to_numpy()[idx]

    df = pd.DataFrame({
        "source_file": ges_files,
        "ra_deg": ras,
        "dec_deg": decs,
        "teff_k": teff_raw,
        "logg": logg_raw,
        "feh": feh_raw,
    }).loc[matched].reset_index(drop=True)

    mk_class = bin_teff_to_mk(df["teff_k"].to_numpy())
    n_other = int((mk_class == "OTHER").sum())
    keep = mk_class != "OTHER"
    df = df.loc[keep].reset_index(drop=True)
    df["mk_class"] = mk_class[keep]
    df["boundary_distance_k"] = compute_boundary_distance_k(
        df["teff_k"].to_numpy(), df["mk_class"].to_numpy()
    )
    df["dwarf_flag"] = df["logg"] > 3.5
    df["mk_int"] = df["mk_class"].map(MK_INT).astype("int8")

    counts = df["mk_class"].value_counts().to_dict()
    to_drop: list[str] = []
    for cls in ALLOWED_MK_CLASSES:
        n = int(counts.get(cls, 0))
        if n < min_per_class:
            if allow_drop_underfilled:
                logger.warning(
                    "class %s under-filled (%d < %d) - dropping",
                    cls, n, min_per_class,
                )
                to_drop.append(cls)
            else:
                raise RuntimeError(
                    f"class {cls!r} has only {n} spectra (< {min_per_class}); "
                    "pass allow_drop_underfilled=True to proceed with fewer classes."
                )
        elif n < warn_per_class:
            logger.warning(
                "class %s has only %d spectra (< warn threshold %d)",
                cls, n, warn_per_class,
            )

    if to_drop:
        df = df[~df["mk_class"].isin(to_drop)].reset_index(drop=True)

    per_class = df["mk_class"].value_counts().to_dict()
    stats = LabelStats(
        n_input=int(len(ges_files)),
        n_unmatched=n_unmatched,
        n_other=n_other,
        n_final=int(len(df)),
        per_class={k: int(v) for k, v in per_class.items()},
    )
    return (
        df[[
            "source_file", "ra_deg", "dec_deg", "teff_k", "logg", "feh",
            "mk_class", "mk_int", "boundary_distance_k", "dwarf_flag",
        ]],
        stats,
    )
