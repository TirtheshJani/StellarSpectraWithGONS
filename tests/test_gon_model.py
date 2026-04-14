"""Smoke tests for the GON generator.

These tests verify that imports work, tensor shapes are correct, and a
single forward/backward pass with GON latent inference runs end-to-end on
CPU without errors.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models import (  # noqa: E402
    FourierPositionalEncoding,
    GONModel,
    apply_rv_shift_to_wavelengths,
    build_line_window_weights,
)


def test_fourier_encoding_shape():
    enc = FourierPositionalEncoding(num_frequencies=4)
    wl = torch.linspace(4000.0, 5000.0, 32).unsqueeze(0)  # (1, 32)
    out = enc(wl)
    assert out.shape == (1, 32, enc.out_dim)


def test_rv_shift_passthrough_when_none():
    wl = torch.linspace(4000.0, 5000.0, 16).unsqueeze(0)
    shifted = apply_rv_shift_to_wavelengths(wl, None)
    assert torch.equal(wl, shifted)


def test_rv_shift_formula():
    wl = torch.full((1, 4), 5000.0)
    rv = torch.tensor([299.792458])  # 0.001 c
    shifted = apply_rv_shift_to_wavelengths(wl, rv)
    # rest = observed / (1 + v/c) -> observed/1.001
    expected = wl / 1.001
    assert torch.allclose(shifted, expected, atol=1e-3)


def test_line_window_weights():
    wl = torch.linspace(6550.0, 6575.0, 26).unsqueeze(0)
    weights = build_line_window_weights(wl, [(6562.8, 5.0, 3.0)], base_weight=1.0)
    assert weights is not None
    assert weights.shape == wl.shape
    # Points inside the window should have weight 3.0, outside should stay 1.0.
    assert weights.max().item() == pytest.approx(3.0)
    assert weights.min().item() == pytest.approx(1.0)


def test_gon_forward_and_latent_inference():
    torch.manual_seed(0)
    model = GONModel(
        latent_dim=4,
        coord_encoding="pe",
        pe_num_frequencies=4,
        mlp_hidden_dim=32,
        mlp_layers=3,
        mlp_activation="relu",
    )
    B, N = 2, 64
    wl = torch.linspace(4000.0, 5000.0, N).unsqueeze(0).expand(B, N).contiguous() / 5000.0
    target = torch.zeros(B, N)

    out = model(
        wavelengths_observed=wl,
        target_flux=target,
        infer_latent=True,
        infer_rv=False,
    )

    assert out["pred"].shape == (B, N)
    assert out["z"].shape == (B, model.latent_dim)
    assert torch.isfinite(out["loss"]).item()
