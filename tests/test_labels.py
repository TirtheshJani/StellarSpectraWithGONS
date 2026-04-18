"""Unit tests for src/interpret/labels.py (no network calls)."""
from __future__ import annotations

import numpy as np
import pytest

from src.interpret.labels import (
    MK_BIN_EDGES,
    MK_INT,
    _resolve_column_aliases,
    bin_teff_to_mk,
    compute_boundary_distance_k,
    parse_ra_dec_from_filename,
)
from src.interpret.lines import ALLOWED_MK_CLASSES


class TestBinTeffToMk:
    def test_canonical_cases(self):
        teff = np.array([8000.0, 6500.0, 5800.0, 4500.0, 3000.0, 11000.0])
        expected = np.array(["A", "F", "G", "K", "OTHER", "OTHER"], dtype=object)
        np.testing.assert_array_equal(bin_teff_to_mk(teff), expected)

    def test_edges_inclusive_lower_exclusive_upper(self):
        assert bin_teff_to_mk(np.array([7300.0]))[0] == "A"
        assert bin_teff_to_mk(np.array([9999.9]))[0] == "A"
        assert bin_teff_to_mk(np.array([10000.0]))[0] == "OTHER"
        assert bin_teff_to_mk(np.array([6000.0]))[0] == "F"
        assert bin_teff_to_mk(np.array([7299.9]))[0] == "F"
        assert bin_teff_to_mk(np.array([5300.0]))[0] == "G"
        assert bin_teff_to_mk(np.array([5999.9]))[0] == "G"
        assert bin_teff_to_mk(np.array([3900.0]))[0] == "K"
        assert bin_teff_to_mk(np.array([5299.9]))[0] == "K"

    def test_no_obm_labels_ever(self):
        rng = np.random.default_rng(42)
        teff = rng.uniform(2000.0, 40000.0, size=1000)
        labels = bin_teff_to_mk(teff)
        assert not np.isin(labels.astype(str), ["O", "B", "M"]).any()


class TestComputeBoundaryDistance:
    def test_midpoint_of_bin(self):
        teff = np.array([8650.0])  # midway in A
        mk = np.array(["A"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        assert d[0] == pytest.approx(1350.0)

    def test_near_lower_edge(self):
        teff = np.array([7400.0])  # 100 K above 7300
        mk = np.array(["A"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        assert d[0] == pytest.approx(100.0)

    def test_other_gives_nan(self):
        teff = np.array([2500.0])
        mk = np.array(["OTHER"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        assert np.isnan(d[0])

    def test_vectorises_correctly(self):
        teff = np.array([7400.0, 6100.0, 5900.0, 4000.0])
        mk = np.array(["A", "F", "G", "K"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        np.testing.assert_allclose(d, [100.0, 100.0, 100.0, 100.0])


class TestParseRaDecFromFilename:
    def test_canonical_name(self):
        ra, dec = parse_ra_dec_from_filename("ges_uves_123.456789_-45.678901.fits")
        assert ra == pytest.approx(123.456789)
        assert dec == pytest.approx(-45.678901)

    def test_with_path_prefix(self):
        ra, dec = parse_ra_dec_from_filename(
            "data/ges/uves/ges_uves_10.123456_-20.654321.fits"
        )
        assert ra == pytest.approx(10.123456)
        assert dec == pytest.approx(-20.654321)

    def test_rejects_non_ges_filename(self):
        with pytest.raises(ValueError):
            parse_ra_dec_from_filename("apogee_123.fits")


class TestMkIntContract:
    def test_integer_codes_match_allowed_classes(self):
        assert list(MK_INT.keys()) == list(ALLOWED_MK_CLASSES)
        assert list(MK_INT.values()) == list(range(len(ALLOWED_MK_CLASSES)))


class TestMkBinEdges:
    def test_no_gaps_between_bins(self):
        ordered = sorted(MK_BIN_EDGES.items(), key=lambda kv: kv[1][1])
        prev_hi = None
        for _, (lo, hi) in ordered:
            if prev_hi is not None:
                assert lo == prev_hi, "MK bins must be contiguous"
            prev_hi = hi

    def test_all_allowed_classes_have_edges(self):
        assert set(MK_BIN_EDGES.keys()) == set(ALLOWED_MK_CLASSES)


class TestResolveColumnAliases:
    def test_canonical_vizier_names(self):
        cols = ["RA_ICRS", "DE_ICRS", "Teff", "logg", "[Fe/H]", "Other"]
        m = _resolve_column_aliases(cols)
        assert m == {
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "teff": "Teff",
            "logg": "logg",
            "feh": "[Fe/H]",
        }

    def test_case_insensitive_match(self):
        cols = ["raj2000", "dej2000", "teff", "LOGG", "FeH"]
        m = _resolve_column_aliases(cols)
        assert m["ra"] == "raj2000"
        assert m["feh"] == "FeH"

    def test_missing_column_raises(self):
        cols = ["RA_ICRS", "DE_ICRS", "Teff", "logg"]  # no [Fe/H]
        with pytest.raises(KeyError):
            _resolve_column_aliases(cols)
