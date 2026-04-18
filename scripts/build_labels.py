#!/usr/bin/env python
"""Build MK-class labels for Gaia-ESO UVES spectra and write to parquet."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.interpret.labels import build_labels


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, type=Path,
                   help="regridded HDF5 from build_hdf5.py")
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="directory for the VizieR catalog cache")
    p.add_argument("--out", required=True, type=Path,
                   help="output parquet path")
    p.add_argument("--match-radius-arcsec", type=float, default=0.5)
    p.add_argument("--min-per-class", type=int, default=50)
    p.add_argument("--warn-per-class", type=int, default=200)
    p.add_argument("--allow-drop-underfilled", action="store_true",
                   help="drop a class below min-per-class rather than raising")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    labels_df, stats = build_labels(
        h5_path=args.h5,
        cache_dir=args.cache_dir,
        match_radius_arcsec=args.match_radius_arcsec,
        min_per_class=args.min_per_class,
        warn_per_class=args.warn_per_class,
        allow_drop_underfilled=args.allow_drop_underfilled,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(args.out, index=False)
    stats_path = args.out.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"wrote {len(labels_df)} labels -> {args.out}")
    print(f"stats -> {stats_path}")
    print(f"per-class: {stats.per_class}")


if __name__ == "__main__":
    main()
