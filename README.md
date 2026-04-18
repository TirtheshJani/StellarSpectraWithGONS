# Stellar Spectra with Gradient Origin Networks

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A unified, cross-survey deep generative model for stellar spectra. This project
trains a **Gradient Origin Network (GON)** on homogenised spectra drawn from
three major surveys — **APOGEE DR17**, **GALAH DR3**, and **Gaia-ESO DR4
(UVES)** — to learn a single latent representation of stellar atmospheres that
transfers across instruments, resolutions, and wavelength coverage.

> GONs replace a learned encoder with a single gradient step from a zero
> latent. For spectra this means the same generator can be used to *fit*
> observations, *impute* missing wavelength coverage, and *estimate* radial
> velocity — all with one forward / backward pass.

---

## Why this project

Stellar spectra from different surveys live on different wavelength grids,
resolutions, and continuum normalisations. Building a single model that works
across surveys requires:

1. A disciplined **cross-match** of stars that appear in more than one catalog.
2. A **preprocessing pipeline** that brings heterogeneous FITS files onto a
   shared log-λ grid with consistent continuum treatment and quality masks.
3. A **generative model** flexible enough to absorb survey-specific
   calibration while still learning transferable stellar physics.

This repository implements all three pieces end-to-end.

## Highlights

- **Survey ingestion** for APOGEE `apStar` / `apVisit`, GALAH 4-camera FITS,
  and Gaia-ESO UVES Phase-3 products, with resumable parallel downloads and
  FITS-level verification.
- **Cross-match pipeline** that consolidates a clean ~30k common-star list
  from a seed APOGEE catalog with RA/Dec matching to GALAH DR3 and GES DR4.
- **Preprocessing** with log-λ resampling (R ≈ 10,000 over 3500–17,000 Å),
  survey-specific continuum normalisation (polynomial, Gaussian, percentile),
  telluric / detector-gap masking, and SNR-aware quality flags.
- **GON model** (`src/models/gon.py`) with Fourier / SIREN coordinate encoders,
  empirical-Bayes latent inference (`z = −η ∇_z NLL|_{z=0}`), optional
  gradient-based RV inference, and line-window-weighted losses.
- **Storage** in compressed HDF5 (regridded) and PyArrow Parquet (native
  resolution, ragged arrays) with stratified train / val / test splits.
- **Interpretability audit** for a LightGBM MK classifier trained on the GES
  UVES subset — permutation importance, SHAP, sliding-window occlusion, and
  masked-line ablation against a random-window null (see below).

## Repository layout

```
.
├── src/
│   ├── fetch/                 # Survey-specific downloaders (APOGEE / GALAH / GES)
│   ├── preprocess/            # Readers, resampling, continuum, HDF5 / Parquet builders
│   ├── models/                # GON generator, coordinate encoders, losses
│   ├── interpret/             # MK classifier + interpretability pipeline
│   └── utils/                 # Cross-match and legacy HDF5 loaders
├── scripts/                   # End-to-end CLI drivers
├── notebooks/                 # Tutorial notebooks (plotting, crossmatching, interpret demo)
├── docs/                      # Design PDF and implementation status
├── data/                      # Manifests (tracked); large FITS/H5 payloads ignored
├── tests/                     # Smoke tests
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks]"

# 2. Minimal offline demo (APOGEE only)
python create_offline_starlist.py

# 3. Full 30k cross-matched starlist + downloads
python scripts/download_spectra_30k.py \
    --apogee-csv scripts/Apogee_ID.csv \
    --target-size 30000 \
    --concurrency 8 \
    --build-hdf5
```

The downloader writes manifests to `data/<survey>/manifests/`, raw FITS to
`data/<survey>/{apStar,apVisit,…}/`, and the final regridded dataset to
`data/common/processed/baseline_spectra.h5`.

## Training the GON

```python
from src.models import GONModel

model = GONModel(
    latent_dim=16,
    coord_encoding="pe",           # "pe" | "siren" | "none"
    pe_num_frequencies=10,
    mlp_hidden_dim=256,
    mlp_layers=5,
    mlp_activation="gelu",
)

out = model(
    wavelengths_observed=wavelengths,    # (B, N)
    target_flux=flux,                    # (B, N)
    infer_latent=True,                   # one-step GON latent
    infer_rv=True,                       # one-step RV estimate
    loss_type="mse",
    line_windows=[(6562.8, 5.0, 5.0)],   # up-weight Hα, for example
)
loss, z_hat, rv_hat = out["loss"], out["z"], out.get("rv_kms")
```

See `src/models/gon.py` for the full API, including `compute_gon_latent`
(pure latent inference), `reconstruction_loss`, and SIREN / Fourier
coordinate encoders.

## Interpretability pipeline

A LightGBM classifier predicts MK class (A / F / G / K) from the
continuum-normalised GES UVES flux vector over 4800–6800 Å, rebinned to
~1000 features. The audit then checks whether the classifier attends to
canonical diagnostic lines (H Balmer, Mg b, Na D, Ca I 6162/6439) or to
continuum / normalisation artefacts, and validates the finding causally
with masked-line ablation against a random-window null.

```bash
pip install -e '.[interpret]'

python scripts/build_labels.py \
    --h5 data/common/processed/regridded_spectra.h5 \
    --cache-dir data/ges/catalogs/ \
    --out data/ges/labels/ges_mk_labels.parquet

python scripts/build_features.py \
    --h5 data/common/processed/regridded_spectra.h5 \
    --labels data/ges/labels/ges_mk_labels.parquet \
    --out data/interpret/features.npz \
    --max-spectra 5000 --min-snr 20

python scripts/train_classifier.py \
    --features data/interpret/features.npz \
    --model-out models/lightgbm_mk.pkl \
    --metrics-out metrics/lightgbm_metrics.json

python scripts/run_interpret.py \
    --features data/interpret/features.npz \
    --model models/lightgbm_mk.pkl \
    --out-dir artifacts/

python scripts/ablation.py \
    --features data/interpret/features.npz \
    --model models/lightgbm_mk.pkl \
    --out-dir artifacts/

python scripts/run_benchmark.py \
    --features data/interpret/features.npz \
    --model models/lightgbm_mk.pkl \
    --pickles-dir data/external/pickles/ \
    --out-dir artifacts/

python scripts/make_figure.py \
    --features data/interpret/features.npz \
    --importance artifacts/perm_importance.npz \
    --shap artifacts/shap_values.npz \
    --out-dir figures/
```

Design choices (Pecaut & Mamajek 2013 Teff bins, UVES-air wavelengths,
train-set median imputation, group-aware splits, bootstrap + random-null
ablation) are documented in
[`plan-stellar-wild-manatee.md`](plan-stellar-wild-manatee.md) and in the
module docstrings under `src/interpret/`. Minimal walkthrough:
[`notebooks/interpretability_demo.ipynb`](notebooks/interpretability_demo.ipynb).

The citation grep gate ensures the shorthand "Liu et al 2019" is never used —
the correct attribution is **Li, Lin & Qiu (2019)**:

```bash
bash scripts/check_citations.sh
```

## Status

Phase 1 (data collection infrastructure) and Phase 2 (preprocessing) are
complete. See [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)
for a detailed breakdown. Phase 3 (model training at scale) is the next step.

## Design notes

The underlying research motivation and architecture are described in
[`docs/DeepGenerativeSpectra.pdf`](docs/DeepGenerativeSpectra.pdf). In brief,
we train a generator `g([γ(λ), z]; θ) → f̂(λ)` and infer the per-spectrum
latent `z` by a single gradient step from zero:

```
z_i = −η · ∇_{z_i} ½ ‖f_i − g([γ(λ_i), z_i]; θ)‖² / σ²  |_{z_i = 0}
```

The same trick is applied (optionally) to the radial velocity, giving a
fully differentiable, encoder-free inference procedure that is trivially
compatible with line-window weighting and Huber losses.

## Testing

```bash
pytest
bash scripts/check_citations.sh
```

Tests are intentionally lightweight — they exercise imports, tensor shapes
on the GON model, the MK-labels binning, the feature rebin/imputation,
the LINE_SETS coverage contract, and the Pickles filename parser. Heavy
I/O (real FITS, real HDF5) is covered by the tutorial notebooks in
`notebooks/`.

## License

MIT — see [`LICENSE`](LICENSE). The APOGEE, GALAH, and Gaia-ESO data products
retain their respective survey licenses.

## Citation / acknowledgements

- Bond-Taylor & Willcocks, *Gradient Origin Networks* (ICLR 2021).
- SDSS-IV / APOGEE-2 (DR17), GALAH (DR3), and Gaia-ESO (DR4 / DR5.1).
- Pecaut & Mamajek (2013) for MK Teff calibration; Gray & Corbally (2009)
  for MK line-strength diagnostics; Hourihane et al. (2023) for the GES
  DR5.1 recommended-parameters catalog.
- Pickles (1998, PASP 110, 863) stellar-spectrum library for the external
  MK benchmark.
- Li, Lin & Qiu (2019) for the LightGBM attribution-style audit that the
  interpretability pipeline is modelled on. (This replaces the shorthand
  "Liu et al 2019" referenced in early design drafts.)
- The cross-match helper in `src/utils/xmatch.py` is adapted from
  [`astroNN`](https://github.com/henrysky/astroNN) by Henry Leung.
