#!/usr/bin/env python
"""Build the LightGBM feature matrix from the regridded HDF5."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.interpret.features import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_MAX_NAN_FRAC,
    DEFAULT_MIN_COVERED_SPECTRA,
    DEFAULT_MIN_SNR,
    DEFAULT_REBIN,
    DEFAULT_WAVE_MAX,
    DEFAULT_WAVE_MIN,
    build_features,
    coverage_probe,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--wave-min", type=float, default=DEFAULT_WAVE_MIN)
    p.add_argument("--wave-max", type=float, default=DEFAULT_WAVE_MAX)
    p.add_argument("--rebin", type=int, default=DEFAULT_REBIN)
    p.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR)
    p.add_argument("--max-nan-frac", type=float, default=DEFAULT_MAX_NAN_FRAC)
    p.add_argument("--max-spectra", type=int, default=5000)
    p.add_argument("--group-col", default=None,
                   help="labels column to use for group-stratified split")
    p.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    p.add_argument("--min-covered-spectra", type=int, default=DEFAULT_MIN_COVERED_SPECTRA)
    p.add_argument("--skip-coverage-probe", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not args.skip_coverage_probe:
        coverage_probe(
            h5_path=args.h5,
            wave_min=args.wave_min,
            wave_max=args.wave_max,
            coverage_threshold=args.coverage_threshold,
            min_spectra=args.min_covered_spectra,
        )

    payload = build_features(
        h5_path=args.h5,
        labels_parquet=args.labels,
        out_path=args.out,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        rebin=args.rebin,
        min_snr=args.min_snr,
        max_nan_frac=args.max_nan_frac,
        max_spectra=args.max_spectra,
        group_col=args.group_col,
        seed=args.seed,
    )
    X = payload["X"]
    print(f"wrote {args.out}  shape={X.shape}  "
          f"train/val/test={len(payload['train_idx'])}/"
          f"{len(payload['val_idx'])}/{len(payload['test_idx'])}")


if __name__ == "__main__":
    main()
