#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.starlist_pipeline import run_pipeline  # noqa: E402
from src.fetch.fetch_apogee import main as apogee_main  # noqa: E402
from src.fetch.fetch_galah import main as galah_main  # noqa: E402
from src.fetch.fetch_ges import main as ges_main  # noqa: E402
from src.preprocess.build_hdf5 import main as build_hdf5_main  # noqa: E402


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Build 30k starlist and download APOGEE, GALAH, GES spectra')
    p.add_argument('--apogee-csv', default=os.path.join('scripts', 'Apogee_ID.csv'))
    p.add_argument('--base-dir', default='data', help='Base directory for downloads and manifests')
    p.add_argument('--target-size', type=int, default=30000)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--downloader', choices=['python', 'wget'], default='python')
    p.add_argument('--apogee-sas-only', action='store_true', help='Skip SkyServer and build a minimal APOGEE-only starlist; download apStar via SAS only')
    # DR17 tuning
    p.add_argument('--dr17-chunk-size', type=int, default=1000)
    p.add_argument('--dr17-timeout', type=int, default=90)
    p.add_argument('--dr17-retries', type=int, default=5)
    p.add_argument('--dr17-sleep-ms', type=int, default=0)
    p.add_argument('--build-hdf5', action='store_true', help='Build combined HDF5 after downloads')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Paths
    starlist_parquet = os.path.join(args.base_dir, 'common', 'manifests', 'starlist_30k.parquet')
    apogee_pos_parquet = os.path.join(args.base_dir, 'apogee', 'manifests', 'apogee_positions.parquet')
    galah_pos_parquet = os.path.join(args.base_dir, 'galah', 'manifests', 'galah_positions.parquet')
    ges_pos_parquet = os.path.join(args.base_dir, 'ges', 'manifests', 'ges_positions.parquet')
    apogee_manifest_csv = os.path.join(args.base_dir, 'apogee', 'manifests', 'apogee_manifest.csv')
    galah_manifest_csv = os.path.join(args.base_dir, 'galah', 'manifests', 'galah_manifest.csv')
    ges_manifest_csv = os.path.join(args.base_dir, 'ges', 'manifests', 'ges_manifest.csv')

    # Ensure dirs
    for p in [starlist_parquet, apogee_pos_parquet, galah_pos_parquet, ges_pos_parquet,
              apogee_manifest_csv, galah_manifest_csv, ges_manifest_csv]:
        ensure_dir(os.path.dirname(p))

    # 1) Build star list
    if args.apogee_sas_only:
        # Minimal starlist with only apogee_id (first N unique)
        import pandas as pd
        df = pd.read_csv(args.apogee_csv, low_memory=False)
        apogee_cols = [c for c in df.columns if 'apogee' in c.lower()]
        if not apogee_cols:
            raise SystemExit('Could not find an APOGEE ID column in the input CSV')
        ids = df[apogee_cols[0]].astype(str).dropna().drop_duplicates()
        if len(ids) > args.target_size:
            ids = ids.iloc[:args.target_size]
        pd.DataFrame({'apogee_id': ids}).to_parquet(starlist_parquet, index=False)
        print(f"Wrote minimal APOGEE starlist to {starlist_parquet}")
    else:
        run_pipeline(
            apogee_csv=args.apogee_csv,
            galah_positions=None,
            ges_positions=None,
            apogee_positions_out=apogee_pos_parquet,
            galah_positions_out=galah_pos_parquet,
            ges_positions_out=ges_pos_parquet,
            starlist_out=starlist_parquet,
            target_size=args.target_size,
            dr17_chunk_size=args.dr17_chunk_size,
            dr17_timeout=args.dr17_timeout,
            dr17_retries=args.dr17_retries,
            dr17_sleep_ms=args.dr17_sleep_ms,
        )

    # 2) Build + download manifests per survey
    apogee_main([
        '--starlist', starlist_parquet,
        '--manifest', apogee_manifest_csv,
        '--base-dir', args.base_dir,
        '--mode', 'both',
        '--concurrency', str(args.concurrency),
        '--downloader', args.downloader,
    ])

    if not args.apogee_sas_only:
        galah_main([
            '--starlist', starlist_parquet,
            '--manifest', galah_manifest_csv,
            '--base-dir', args.base_dir,
            '--mode', 'both',
            '--concurrency', str(args.concurrency),
            '--downloader', args.downloader,
        ])

        ges_main([
            '--starlist', starlist_parquet,
            '--manifest', ges_manifest_csv,
            '--base-dir', args.base_dir,
            '--mode', 'both',
            '--concurrency', str(max(1, args.concurrency // 2)),
            '--downloader', args.downloader,
        ])

    if args.build_hdf5:
        out_h5 = os.path.join(args.base_dir, 'common', 'processed', 'baseline_spectra.h5')
        ensure_dir(os.path.dirname(out_h5))
        build_hdf5_main([
            '--apogee-manifest', apogee_manifest_csv,
            '--galah-manifest', galah_manifest_csv if not args.apogee_sas_only else '',
            '--ges-manifest', ges_manifest_csv if not args.apogee_sas_only else '',
            '--out', out_h5,
        ])


if __name__ == '__main__':
    main()


