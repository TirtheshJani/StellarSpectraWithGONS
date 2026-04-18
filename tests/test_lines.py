"""Registry-integrity tests for src/interpret/lines.py.

These gate the *contract* of the registry; they do not verify that any
line falls inside the training feature window (that is test_line_coverage).
"""

from __future__ import annotations

from src.interpret.lines import (
    ALLOWED_MK_CLASSES,
    LINE_SETS,
    MK_LINES,
    lines_for_class,
    lines_in_window,
)


def test_line_names_unique():
    names = [line.name for line in MK_LINES]
    assert len(names) == len(set(names))


def test_diag_for_subset_of_allowed():
    allowed = set(ALLOWED_MK_CLASSES)
    for line in MK_LINES:
        assert set(line.diag_for).issubset(allowed), (
            f"{line.name} has diag_for {line.diag_for}, not subset of {allowed}"
        )


def test_no_obm_classes():
    forbidden = {"O", "B", "M"}
    for line in MK_LINES:
        assert not forbidden.intersection(line.diag_for), (
            f"{line.name} references an excluded class: {line.diag_for}"
        )


def test_wavelengths_sorted_ascending():
    wls = [line.wavelength_aa for line in MK_LINES]
    assert wls == sorted(wls), "MK_LINES must be sorted by wavelength"


def test_wavelengths_plausible():
    for line in MK_LINES:
        assert 3000.0 < line.wavelength_aa < 10000.0


def test_line_sets_windows_well_formed():
    for set_name, windows in LINE_SETS.items():
        for lo, hi in windows:
            assert lo < hi, f"{set_name} window ({lo}, {hi}) not increasing"
            assert 3000.0 < lo < 10000.0
            assert 3000.0 < hi < 10000.0


def test_line_sets_required_keys():
    # contract with ablation.py
    assert {"H_balmer", "Mg_b", "Na_D"} <= set(LINE_SETS.keys())


def test_line_sets_no_overlap_within_set():
    for set_name, windows in LINE_SETS.items():
        sorted_w = sorted(windows)
        for (_lo1, hi1), (lo2, _hi2) in zip(sorted_w, sorted_w[1:]):
            assert hi1 < lo2, f"{set_name} has overlapping windows"


def test_h_balmer_windows_cover_lines():
    windows = LINE_SETS["H_balmer"]
    assert any(lo <= 6562.80 <= hi for lo, hi in windows), "H_balmer misses Halpha"
    assert any(lo <= 4861.33 <= hi for lo, hi in windows), "H_balmer misses Hbeta"


def test_mg_b_window_covers_triplet():
    (lo, hi), = LINE_SETS["Mg_b"]
    for wl in (5167.32, 5172.68, 5183.60):
        assert lo <= wl <= hi


def test_na_d_window_covers_doublet():
    (lo, hi), = LINE_SETS["Na_D"]
    for wl in (5889.95, 5895.92):
        assert lo <= wl <= hi


def test_lines_in_window_matches_direct_filter():
    window = (5000.0, 6000.0)
    result = lines_in_window(*window)
    expected = [line for line in MK_LINES if 5000.0 <= line.wavelength_aa <= 6000.0]
    assert result == expected


def test_lines_for_class_rejects_unknown_class():
    import pytest

    with pytest.raises(ValueError):
        lines_for_class("M")


def test_lines_for_class_a_contains_balmer_only():
    names = {line.name for line in lines_for_class("A")}
    assert names == {"H_alpha", "H_beta"}


def test_lines_for_class_k_excludes_balmer():
    names = {line.name for line in lines_for_class("K")}
    assert "H_alpha" not in names
    assert "H_beta" not in names
    assert {"Mg_b1", "Mg_b2", "Mg_b3"} <= names
    assert {"Na_D1", "Na_D2"} <= names
