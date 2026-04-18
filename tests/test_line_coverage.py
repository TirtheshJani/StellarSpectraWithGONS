"""Guard: every LINE_SETS entry intersects the default feature grid.

This test protects the *contract* between src/interpret/lines.py and
src/interpret/features.py -- if the feature window or rebin factor changes
in a way that would zero-out any ablation window, we find out at test time
rather than at ablation time.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.interpret.features import (
    DEFAULT_REBIN,
    DEFAULT_WAVE_MAX,
    DEFAULT_WAVE_MIN,
    rebin_flux,
)
from src.interpret.lines import LINE_SETS, MK_LINES, lines_in_window


@pytest.fixture(scope="module")
def default_wave_centers() -> np.ndarray:
    """Wave centres produced by the default rebin of the default window."""
    n_pix = 3636  # ~0.55 A / pixel across 2000 A -> representative native grid
    wave = np.linspace(DEFAULT_WAVE_MIN, DEFAULT_WAVE_MAX, n_pix)
    flux = np.ones((1, n_pix), dtype=np.float32)
    _, wc = rebin_flux(
        flux, wave,
        wave_min=DEFAULT_WAVE_MIN, wave_max=DEFAULT_WAVE_MAX,
        rebin_factor=DEFAULT_REBIN,
    )
    return wc


def test_every_line_set_intersects_grid(default_wave_centers):
    wc = default_wave_centers
    for set_name, windows in LINE_SETS.items():
        for lo, hi in windows:
            bins_in = np.sum((wc >= lo) & (wc <= hi))
            assert bins_in > 0, (
                f"LINE_SETS[{set_name!r}] window ({lo}, {hi}) has no "
                f"overlap with default wave_centers -- ablation would be a no-op."
            )


def test_every_in_window_line_has_a_bin(default_wave_centers):
    wc = default_wave_centers
    dx = float(np.median(np.diff(wc)))
    for line in lines_in_window(DEFAULT_WAVE_MIN, DEFAULT_WAVE_MAX):
        nearest = np.min(np.abs(wc - line.wavelength_aa))
        assert nearest <= dx, (
            f"{line.name} at {line.wavelength_aa} A is more than one bin "
            f"({dx:.2f} A) from any wave_centre (nearest: {nearest:.2f} A)"
        )


def test_mk_lines_within_default_window():
    # contract: all registry entries should fall inside the default window
    # so the classifier has a chance to attend to them
    for line in MK_LINES:
        assert DEFAULT_WAVE_MIN <= line.wavelength_aa <= DEFAULT_WAVE_MAX, (
            f"{line.name} at {line.wavelength_aa} A outside default window"
        )
