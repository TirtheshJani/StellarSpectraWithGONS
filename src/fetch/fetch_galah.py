import os
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import requests

from .common import ensure_dir, write_manifest, parallel_download, verify_fits_basic

# Public mirror documented for GALAH DR3 spectra
# Note: Actual access might require GALAH-provided links. We include common pattern placeholders.
GALAH_BASE = "https://datacentral.org.au/services/download/gx/galah/dr3/spectra"

CAMERAS = ['blue', 'green', 'red', 'ir']

# Data Central TAP endpoint for positional fallback
DC_TAP_SYNC = "https://datacentral.org.au/services/tap/sync"

ADQL_TEMPLATE = """
SELECT TOP 1
    o.obs_publisher_did as did,
    o.dataproduct_type,
    o.calib_level,
    o.instrument_name,
    o.obs_collection,
    o.access_url,
    o.access_format,
    o.ra,
    o.dec
FROM ivoa.obscore AS o
WHERE 1=1
  AND (o.obs_collection LIKE 'GALAH%')
  AND o.dataproduct_type='spectrum'
  AND CONTAINS(POINT('ICRS', o.ra, o.dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))=1
ORDER BY o.calib_level DESC
"""


def query_dc_tap(ra: float, dec: float, radius_arcsec: float = 1.0) -> List[Dict[str, str]]:
	radius_deg = radius_arcsec / 3600.0
	adql = ADQL_TEMPLATE.format(ra=ra, dec=dec, radius_deg=radius_deg)
	resp = requests.post(DC_TAP_SYNC, data={
		'REQUEST': 'doQuery',
		'LANG': 'ADQL',
		'FORMAT': 'json',
		'QUERY': adql,
	})
	resp.raise_for_status()
	data = resp.json()
	rows = data.get('data') or []
	cols = [c['name'] for c in data.get('metadata', [])]
	results: List[Dict[str, str]] = []
	for row in rows:
		results.append({cols[i]: row[i] for i in range(len(cols))})
	return results


def build_galah_urls(galah_id: str) -> List[Tuple[str, str]]:
	urls: List[Tuple[str, str]] = []
	for cam in CAMERAS:
		name = f"{galah_id}_{cam}.fits"
		urls.append((f"{GALAH_BASE}/{cam}/{name}", f"galah/{galah_id}/{name}"))
	return urls


def build_manifest(starlist_parquet: str, out_csv: str, base_dir: str = 'data') -> None:
	if not os.path.exists(starlist_parquet):
		raise FileNotFoundError(starlist_parquet)
	df = pd.read_parquet(starlist_parquet)
	from .common import http_head
	rows: List[Dict[str, object]] = []
	# If GALAH IDs are available, build camera URLs directly
	if 'galah_id' in df.columns and df['galah_id'].notna().any():
		for galah_id in df['galah_id'].dropna().astype(str).unique():
			for url, lp in build_galah_urls(galah_id):
				local_path = os.path.join(base_dir, lp)
				code, size = http_head(url, timeout=20)
				rows.append({
					'remote_url': url,
					'local_path': local_path,
					'status': 'pending',
					'http_status': code,
					'bytes': size,
				})
	else:
		# Positional fallback: query around each star for a GALAH spectrum
		if not {'ra', 'dec'}.issubset(df.columns):
			raise ValueError("starlist parquet missing 'galah_id' and 'ra/dec' for GALAH query")
		coords = df[['ra', 'dec']].dropna().drop_duplicates()
		for _, r in coords.iterrows():
			ra = float(r['ra']); dec = float(r['dec'])
			try:
				cands = query_dc_tap(ra, dec, radius_arcsec=1.0)
			except Exception:
				cands = []
			if not cands:
				continue
			best = cands[0]
			remote = best.get('access_url')
			if not remote:
				continue
			code, size = http_head(remote, timeout=20)
			local_name = f"galah_{ra:.6f}_{dec:.6f}.fits"
			local_path = os.path.join(base_dir, 'galah', 'positional', local_name)
			rows.append({
				'remote_url': remote,
				'local_path': local_path,
				'status': 'pending',
				'http_status': code,
				'bytes': size,
			})
	write_manifest(rows, out_csv)
	print(f"Wrote GALAH manifest with {len(rows)} entries -> {out_csv}")


def download_from_manifest(manifest_csv: str, concurrency: int = 8, downloader: str = 'python') -> None:
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
								 verify_cb=check_galah, downloader=downloader)
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
	p.add_argument('--starlist', default=os.path.join('data', 'common', 'manifests', 'starlist_30k.parquet'))
	p.add_argument('--manifest', default=os.path.join('data', 'galah', 'manifests', 'galah_manifest.csv'))
	p.add_argument('--base-dir', default='data')
	p.add_argument('--mode', choices=['build', 'download', 'both'], default='both')
	p.add_argument('--concurrency', type=int, default=8)
	p.add_argument('--downloader', choices=['python', 'wget'], default='python')
	args = p.parse_args(argv)

	ensure_dir(os.path.dirname(args.manifest))
	if args.mode in {'build', 'both'}:
		build_manifest(args.starlist, args.manifest, base_dir=args.base_dir)
	if args.mode in {'download', 'both'}:
		download_from_manifest(args.manifest, concurrency=args.concurrency, downloader=args.downloader)


if __name__ == '__main__':
	main()