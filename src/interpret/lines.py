"""Rest wavelengths of MK diagnostic lines and ablation windows.

All wavelengths are in AIR, matching the UVES pipeline convention used by
the Gaia-ESO Survey DR5 spectra (Sacco et al. 2014). Values verified
against NIST ASD (Kramida et al. 2023) and Moore's multiplet table.

``diag_for`` attributions follow Gray & Corbally (2009) Stellar Spectral
Classification and Gray (2005) Observation and Analysis of Stellar
Photospheres, restricted to the A/F/G/K label set used by the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Line:
    """A single MK diagnostic absorption line."""

    name: str
    wavelength_aa: float
    species: str
    diag_for: tuple[str, ...]


MK_LINES: Final[list[Line]] = [
    Line("H_beta",    4861.33, "HI",  ("A", "F")),
    Line("Mg_b1",     5167.32, "MgI", ("F", "G", "K")),
    Line("Mg_b2",     5172.68, "MgI", ("F", "G", "K")),
    Line("Mg_b3",     5183.60, "MgI", ("F", "G", "K")),
    Line("Na_D2",     5889.95, "NaI", ("G", "K")),
    Line("Na_D1",     5895.92, "NaI", ("G", "K")),
    Line("Ca_I_6162", 6162.17, "CaI", ("G", "K")),
    Line("Ca_I_6439", 6439.08, "CaI", ("G", "K")),
    Line("H_alpha",   6562.80, "HI",  ("A", "F")),
]


LINE_SETS: Final[dict[str, list[tuple[float, float]]]] = {
    "H_balmer": [(6542.8, 6582.8), (4841.3, 4881.3)],
    "Mg_b":     [(5165.0, 5186.0)],
    "Na_D":     [(5887.0, 5898.0)],
    "Ca_I":     [(6160.0, 6164.0), (6437.0, 6441.0)],
}


ALLOWED_MK_CLASSES: Final[tuple[str, ...]] = ("A", "F", "G", "K")


def lines_in_window(wave_min: float, wave_max: float) -> list[Line]:
    """Return MK_LINES entries whose rest wavelength is inside ``[wave_min, wave_max]``."""
    return [line for line in MK_LINES if wave_min <= line.wavelength_aa <= wave_max]


def lines_for_class(mk_class: str) -> list[Line]:
    """Return MK_LINES entries whose ``diag_for`` includes ``mk_class``."""
    if mk_class not in ALLOWED_MK_CLASSES:
        raise ValueError(f"{mk_class!r} not in {ALLOWED_MK_CLASSES}")
    return [line for line in MK_LINES if mk_class in line.diag_for]
