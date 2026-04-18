"""Unit tests for src/interpret/benchmark.py (no FITS downloads)."""
from __future__ import annotations

import numpy as np
import pytest

from src.interpret.benchmark import (
    PICKLES_UVKLIB_MAP,
    benchmark_report,
    best_template_per_spectrum,
    collapse_to_mk,
    parse_pickles_filename,
)


class TestParsePicklesFilename:
    @pytest.mark.parametrize("n,expected_start", [
        (9, "A0"), (14, "F0"), (19, "G0"), (23, "K0"),
        (56, "A0"), (60, "F0"), (63, "G0"), (66, "K0"),
    ])
    def test_known_dwarfs_and_giants(self, n, expected_start):
        assert parse_pickles_filename(f"uk{n}.fits").startswith(expected_start)

    def test_pickles_prefix_variant(self):
        assert parse_pickles_filename("pickles_uk_23.fits") == "K0V"

    def test_rejects_non_pickles(self):
        with pytest.raises(ValueError):
            parse_pickles_filename("random.fits")

    def test_out_of_range_number_raises(self):
        with pytest.raises(KeyError):
            parse_pickles_filename("uk999.fits")


class TestCollapseToMk:
    @pytest.mark.parametrize("full,coarse", [
        ("A0V", "A"), ("F5III", "F"), ("G8IV", "G"), ("K3V", "K"),
        ("M2V", "OTHER"), ("B8I", "OTHER"), ("O5V", "OTHER"), ("", "OTHER"),
    ])
    def test_mapping(self, full, coarse):
        assert collapse_to_mk(full) == coarse


class TestUvklibMapCoverage:
    def test_all_fgk_dwarfs_represented(self):
        coarse = {collapse_to_mk(v) for v in PICKLES_UVKLIB_MAP.values()}
        assert {"A", "F", "G", "K"} <= coarse

    def test_keys_are_positive_ints(self):
        assert all(isinstance(k, int) and k > 0 for k in PICKLES_UVKLIB_MAP)


class TestBestTemplatePerSpectrum:
    def _make_templates(self, n_bins):
        from src.interpret.benchmark import Template

        rng = np.random.default_rng(0)
        flux_a = 1.0 + 0.01 * rng.standard_normal(n_bins).astype(np.float32)
        flux_g = 0.9 + 0.02 * rng.standard_normal(n_bins).astype(np.float32)
        return [
            Template("uk9.fits",  "A0V", "A", flux_a),
            Template("uk23.fits", "K0V", "K", flux_g),
        ]

    def test_picks_correct_template(self):
        templates = self._make_templates(n_bins=100)
        X = np.stack([
            templates[0].flux_on_grid,   # should match template 0
            templates[1].flux_on_grid,   # should match template 1
        ])
        idx = best_template_per_spectrum(X, templates)
        assert idx.tolist() == [0, 1]

    def test_handles_nan_in_template(self):
        templates = self._make_templates(n_bins=100)
        templates[0].flux_on_grid[:5] = np.nan
        X = np.stack([templates[0].flux_on_grid, templates[1].flux_on_grid])
        X = np.where(np.isnan(X), 1.0, X)  # spectra have no NaN after imputation
        idx = best_template_per_spectrum(X, templates)
        assert idx.shape == (2,)


class TestBenchmarkReport:
    def test_perfect_agreement(self):
        y = np.array(["A", "F", "G", "K", "A", "OTHER"])
        rpt = benchmark_report(y, y, class_labels=["A", "F", "G", "K"])
        assert rpt["agreement_rate"] == 1.0
        assert rpt["macro_f1_fgk"] == pytest.approx(1.0)
        assert rpt["n_compared"] == 6

    def test_all_disagree(self):
        y_true = np.array(["A", "F", "G", "K"])
        y_pred = np.array(["F", "G", "K", "A"])
        rpt = benchmark_report(y_pred, y_true, class_labels=["A", "F", "G", "K"])
        assert rpt["agreement_rate"] == 0.0
