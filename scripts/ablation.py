#!/usr/bin/env python
"""Masked-line ablation driver: CSV of deltas + bootstrap CI + random-null p-values."""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.interpret.classifier import load_model
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS
from src.interpret.occlusion import masked_line_ablation


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    header = list(asdict(rows[0]).keys())
    lines_out = [",".join(header)]
    for r in rows:
        d = asdict(r)
        lines_out.append(",".join(
            f"{d[k]:.6g}" if isinstance(d[k], float) else str(d[k]) for k in header
        ))
    path.write_text("\n".join(lines_out) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--n-random-controls", type=int, default=100)
    p.add_argument("--boundary-k", type=float, default=150.0,
                   help="robustness subset: keep only spectra with boundary_distance_k >= this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    wc = payload["wave_centers"]
    boundary = payload["boundary_distance_k"]
    test_idx = payload["test_idx"]

    model = load_model(args.model)
    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]

    logger = logging.getLogger(__name__)
    logger.info("running full-test ablation (n=%d)", len(test_idx))
    rows_full = masked_line_ablation(
        model, X[test_idx], y[test_idx], wc, LINE_SETS, class_labels,
        per_class=True,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        seed=args.seed,
    )
    _write_csv(args.out_dir / "masked_line_ablation_full.csv", rows_full)

    boundary_mask = boundary[test_idx] >= args.boundary_k
    kept = test_idx[boundary_mask]
    logger.info(
        "boundary-distance subset (>= %.0f K): %d/%d test rows",
        args.boundary_k, int(boundary_mask.sum()), len(test_idx),
    )
    if len(kept) >= 50:
        rows_sub = masked_line_ablation(
            model, X[kept], y[kept], wc, LINE_SETS, class_labels,
            per_class=True,
            n_bootstrap=args.n_bootstrap,
            n_random_controls=args.n_random_controls,
            seed=args.seed,
        )
        _write_csv(
            args.out_dir / f"masked_line_ablation_boundary{int(args.boundary_k)}.csv",
            rows_sub,
        )
    else:
        logger.warning("fewer than 50 rows after boundary cut; skipping subset run")

    print(f"wrote ablation CSVs -> {args.out_dir}")


if __name__ == "__main__":
    main()
