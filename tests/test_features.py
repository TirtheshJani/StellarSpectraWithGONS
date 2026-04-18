"""Unit tests for src/interpret/features.py (synthetic inputs, no HDF5 needed)."""
from __future__ import annotations

import numpy as np
import pytest

from src.interpret.features import (
    apply_median_imputer,
    fit_median_imputer,
    rebin_flux,
    _split_indices,
)


class TestRebinFlux:
    def _synthetic(self, n_spec: int = 10, n_pix: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        wave = np.linspace(4800.0, 6800.0, n_pix)
        flux = 1.0 + 0.01 * rng.standard_normal((n_spec, n_pix)).astype(np.float32)
        return flux, wave

    def test_shape_and_dtype(self):
        flux, wave = self._synthetic()
        X, wc = rebin_flux(flux, wave, 4800.0, 6800.0, rebin_factor=5)
        assert X.shape == (10, 2000 // 5)
        assert X.dtype == np.float32
        assert wc.dtype == np.float32
        assert wc.shape == (400,)

    def test_wave_centers_monotonic_ascending(self):
        flux, wave = self._synthetic()
        _, wc = rebin_flux(flux, wave, 4800.0, 6800.0, rebin_factor=5)
        assert np.all(np.diff(wc) > 0)

    def test_wave_centers_inside_window(self):
        flux, wave = self._synthetic()
        _, wc = rebin_flux(flux, wave, 4800.0, 6800.0, rebin_factor=5)
        assert wc.min() >= 4800.0
        assert wc.max() <= 6800.0

    def test_nan_pixels_propagate_only_when_majority_bad(self):
        # r=5; set 2 of 5 pixels in block 0 to NaN -> bin should still be finite
        flux, wave = self._synthetic(n_spec=1)
        flux[0, :2] = np.nan
        X, _ = rebin_flux(flux, wave, 4800.0, 6800.0, rebin_factor=5)
        assert np.isfinite(X[0, 0])

    def test_nan_pixels_propagate_when_majority_bad(self):
        flux, wave = self._synthetic(n_spec=1)
        flux[0, :3] = np.nan  # 3 of 5 non-finite
        X, _ = rebin_flux(flux, wave, 4800.0, 6800.0, rebin_factor=5)
        assert np.isnan(X[0, 0])

    def test_values_match_simple_average(self):
        # constant flux -> rebin must return the same constant
        flux = np.full((3, 1000), 0.42, dtype=np.float32)
        wave = np.linspace(5000.0, 6000.0, 1000)
        X, _ = rebin_flux(flux, wave, 5000.0, 6000.0, rebin_factor=5)
        np.testing.assert_allclose(X, 0.42, atol=1e-6)

    def test_rejects_bad_rebin_factor(self):
        flux, wave = self._synthetic()
        with pytest.raises(ValueError):
            rebin_flux(flux, wave, rebin_factor=0)

    def test_determinism(self):
        flux, wave = self._synthetic(seed=7)
        X1, wc1 = rebin_flux(flux, wave, rebin_factor=5)
        X2, wc2 = rebin_flux(flux, wave, rebin_factor=5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(wc1, wc2)


class TestMedianImputer:
    def test_fit_ignores_nan(self):
        X = np.array([[1.0, np.nan], [2.0, 10.0], [3.0, np.nan]], dtype=np.float32)
        med = fit_median_imputer(X)
        assert med[0] == pytest.approx(2.0)
        assert med[1] == pytest.approx(10.0)

    def test_all_nan_column_falls_back_to_unity(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=np.float32)
        med = fit_median_imputer(X)
        assert med[0] == 1.0

    def test_apply_fills_only_nans(self):
        X = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        med = np.array([9.0, 9.0], dtype=np.float32)
        out = apply_median_imputer(X, med)
        assert out[0, 0] == 1.0 and out[0, 1] == 9.0
        assert out[1, 0] == 9.0 and out[1, 1] == 4.0

    def test_apply_no_nan_returns_same_array(self):
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        med = np.array([99.0, 99.0], dtype=np.float32)
        out = apply_median_imputer(X, med)
        np.testing.assert_array_equal(out, X)


class TestSplitIndices:
    def test_stratified_split_preserves_class_balance(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 200 + [1] * 200 + [2] * 200 + [3] * 200, dtype=np.int8)
        rng.shuffle(y)
        tr, va, te = _split_indices(y, None, 0.7, 0.15, seed=0)
        assert set(tr) | set(va) | set(te) == set(range(len(y)))
        assert set(tr).isdisjoint(va)
        assert set(va).isdisjoint(te)
        assert 0.65 < len(tr) / len(y) < 0.75

    def test_group_split_no_leakage(self):
        y = np.repeat([0, 1, 2, 3], 50)
        # 40 groups of size 5 each -- same group stays on one side
        groups = np.repeat(np.arange(40), 5)
        tr, va, te = _split_indices(y, groups, 0.7, 0.15, seed=0)
        gtr, gva, gte = set(groups[tr]), set(groups[va]), set(groups[te])
        assert gtr.isdisjoint(gva)
        assert gva.isdisjoint(gte)
        assert gtr.isdisjoint(gte)

    def test_deterministic(self):
        y = np.repeat([0, 1, 2, 3], 50)
        s1 = _split_indices(y, None, 0.7, 0.15, seed=42)
        s2 = _split_indices(y, None, 0.7, 0.15, seed=42)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)
