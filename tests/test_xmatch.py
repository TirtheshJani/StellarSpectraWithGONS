"""Unit tests for the cross-match helper."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("astropy")

from astropy import units as u  # noqa: E402

from src.utils.xmatch import xmatch  # noqa: E402


def test_xmatch_self_returns_zero_separation():
    ra = np.array([10.0, 20.0, 30.0])
    dec = np.array([-5.0, 0.0, 15.0])
    m1, m2, sep = xmatch(ra, dec, ra, dec, maxdist=1.0)
    assert m1.tolist() == [0, 1, 2]
    assert m2.tolist() == [0, 1, 2]
    assert np.all(sep.to_value(u.arcsec) < 1e-3)


def test_xmatch_rejects_far_pairs():
    ra1 = np.array([10.0])
    dec1 = np.array([0.0])
    ra2 = np.array([20.0])
    dec2 = np.array([0.0])
    m1, m2, sep = xmatch(ra1, dec1, ra2, dec2, maxdist=1.0)
    assert m1.size == 0
    assert m2.size == 0
