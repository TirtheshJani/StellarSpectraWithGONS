import os
import argparse
from typing import List, Dict, Tuple
import pandas as pd

from .common import ensure_dir, write_manifest, parallel_download, verify_fits_basic

# Public mirror documented for GALAH DR3 spectra
# Note: Actual access might require GALAH-provided links. We include common pattern placeholders.
GALAH_BASE = "https://datacentral.org.au/services/download/gx/galah/dr3/spectra"

CAMERAS = ['blue', 'green', 'red', 'ir']


def build_galah_urls(galah_id: str) -> List[Tuple[str, str]]:
	urls: List[Tuple[str, str]] = []
	for cam in CAMERAS:
		name = f"{galah_id}_{cam}.fits"
		urls.append((f"{GALAH_BASE}/{cam}/{name}", f"galah/{galah_id}/{name}"))
	return urls


def build_manifest(starlist_parquet: str, out_csv: str) -> None:
	if not os.path.exists(starlist_parquet):
		raise FileNotFoundError(starlist_parquet)
	df = pd.read_parquet(starlist_parquet)
	if 'galah_id' not in df.columns:
		raise ValueError("starlist parquet missing 'galah_id'")
	from .common import http_head
	rows: List[Dict[str, object]] = []
	for galah_id in df['galah_id'].dropna().astype(str).unique():
		for url, lp in build_galah_urls(galah_id):
			local_path = os.path.join('/workspace/data', lp)
			code, size = http_head(url, timeout=20)
			rows.append({
				'remote_url': url,
				'local_path': local_path,
				'status': 'pending',
				'http_status': code,
				'bytes': size,
			})
	write_manifest(rows, out_csv)
	print(f"Wrote GALAH manifest with {len(rows)} entries -> {out_csv}")


def download_from_manifest(manifest_csv: str, concurrency: int = 8) -> None:
	import csv
	pairs: List[Tuple[str, str]] = []
	with open(manifest_csv, 'r') as f:
		r = csv.DictReader(f)
		for row in r:
			pairs.append((row['remote_url'], row['local_path']))
	def check_galah(path: str) -> bool:
		# Ensure it opens and has some basic structure; GALAH spectra often have wavelength/flux in extensions
		from astropy.io import fits
		with fits.open(path, memmap=False) as hdul:
			# Look for common data HDU with flux
			for hdu in hdul:
				if getattr(hdu, 'data', None) is not None:
					return True
		return False
	results = parallel_download(pairs, concurrency=concurrency, timeout=120,
								 verify_cb=check_galah)
	rows = []
	for res in results:
		rows.append({
			'remote_url': res.remote_url,
			'local_path': res.local_path,
			'status': res.status,
			'http_status': res.http_status,
			'bytes': res.bytes,
		})
	write_manifest(rows, manifest_csv)
	from .common import log_failures
	log_failures(results, manifest_csv.replace('.csv', '.failures.log'))
	ok = sum(1 for r in results if r.status == 'ok')
	print(f"Downloaded {ok}/{len(results)} OK from GALAH manifest")


def main(argv: List[str] = None) -> None:
	p = argparse.ArgumentParser(description='GALAH DR3 manifest builder and downloader')
	p.add_argument('--starlist', default='/workspace/data/common/manifests/starlist_30k.parquet')
	p.add_argument('--manifest', default='/workspace/data/galah/manifests/galah_manifest.csv')
	p.add_argument('--mode', choices=['build', 'download', 'both'], default='both')
	p.add_argument('--concurrency', type=int, default=8)
	args = p.parse_args(argv)

	ensure_dir(os.path.dirname(args.manifest))
	if args.mode in {'build', 'both'}:
		build_manifest(args.starlist, args.manifest)
	if args.mode in {'download', 'both'}:
		download_from_manifest(args.manifest, concurrency=args.concurrency)


if __name__ == '__main__':
	main()