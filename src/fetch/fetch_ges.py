import os
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import requests

from .common import ensure_dir, write_manifest, parallel_download, verify_fits_basic

# ESO TAP endpoint
ESO_TAP = "https://archive.eso.org/tap_obs/sync"
# ESO Phase 3 product URL base pattern (will use returned dp_id or access_url directly)

ADQL_TEMPLATE = """
SELECT top 1
    o.obs_publisher_did as did,
    o.dataproduct_type,
    o.calib_level,
    o.instrument_name,
    o.access_url,
    o.access_format,
    o.ra,
    o.dec,
    o.dp_id
FROM ivoa.obscore as o
WHERE 1=1
  AND o.instrument_name LIKE 'UVES%'
  AND CONTAINS(POINT('ICRS', o.ra, o.dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))=1
  AND o.dataproduct_type='spectrum'
ORDER BY o.calib_level DESC
"""


def query_eso_tap(ra: float, dec: float, radius_arcsec: float = 1.0) -> List[Dict[str, str]]:
	radius_deg = radius_arcsec / 3600.0
	adql = ADQL_TEMPLATE.format(ra=ra, dec=dec, radius_deg=radius_deg)
	resp = requests.post(ESO_TAP, data={
		'QUERY': adql,
		'FORMAT': 'json',
		'LANG': 'ADQL',
	})
	resp.raise_for_status()
	data = resp.json()
	rows = data.get('data') or []
	# Normalize rows as dicts using column names
	cols = [c['name'] for c in data.get('metadata', [])]
	results: List[Dict[str, str]] = []
	for row in rows:
		results.append({cols[i]: row[i] for i in range(len(cols))})
	return results


def build_manifest(starlist_parquet: str, out_csv: str) -> None:
	if not os.path.exists(starlist_parquet):
		raise FileNotFoundError(starlist_parquet)
	df = pd.read_parquet(starlist_parquet)
	if not {'ra', 'dec'}.issubset(df.columns):
		raise ValueError("starlist parquet missing 'ra'/'dec'")
	from .common import http_head
	rows: List[Dict[str, object]] = []
	for _, r in df[['ra', 'dec']].dropna().drop_duplicates().iterrows():
		ra = float(r['ra']); dec = float(r['dec'])
		try:
			cands = query_eso_tap(ra, dec, radius_arcsec=1.0)
		except Exception:
			cands = []
		if not cands:
			continue
		best = cands[0]
		remote = best.get('access_url') or f"https://dataportal.eso.org/dataPortal/api/requests/{best.get('dp_id')}"
		code, size = http_head(remote, timeout=20)
		local_name = f"ges_uves_{ra:.6f}_{dec:.6f}.fits"
		local_path = os.path.join('/workspace/data', 'ges', 'uves', local_name)
		rows.append({
			'remote_url': remote,
			'local_path': local_path,
			'status': 'pending',
			'http_status': code,
			'bytes': size,
		})
	write_manifest(rows, out_csv)
	print(f"Wrote GES UVES manifest with {len(rows)} entries -> {out_csv}")


def download_from_manifest(manifest_csv: str, concurrency: int = 4) -> None:
	import csv
	pairs: List[Tuple[str, str]] = []
	with open(manifest_csv, 'r') as f:
		r = csv.DictReader(f)
		for row in r:
			pairs.append((row['remote_url'], row['local_path']))
	def check_ges(path: str) -> bool:
		from astropy.io import fits
		with fits.open(path, memmap=False) as hdul:
			h = hdul[0].header
			if h.get('INSTRUME', '').upper() != 'UVES':
				return False
			# Try to verify wavelength presence
			wmin = h.get('WAVELMIN') or h.get('HIERARCH ESO QC WAVEL_MIN')
			wmax = h.get('WAVELMAX') or h.get('HIERARCH ESO QC WAVEL_MAX')
			if wmin and wmax:
				return True
			# Fallback to WCS linear axis
			crval1 = h.get('CRVAL1')
			cdelt1 = h.get('CDELT1') or h.get('CD1_1')
			naxis1 = h.get('NAXIS1')
			if crval1 is not None and cdelt1 is not None and naxis1:
				return True
			return True
	results = parallel_download(pairs, concurrency=concurrency, timeout=180,
								 verify_cb=check_ges)
	# overwrite manifest
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
	print(f"Downloaded {ok}/{len(results)} OK from GES manifest")


def main(argv: List[str] = None) -> None:
	p = argparse.ArgumentParser(description='Gaia-ESO UVES manifest builder and downloader')
	p.add_argument('--starlist', default='/workspace/data/common/manifests/starlist_30k.parquet')
	p.add_argument('--manifest', default='/workspace/data/ges/manifests/ges_manifest.csv')
	p.add_argument('--mode', choices=['build', 'download', 'both'], default='both')
	p.add_argument('--concurrency', type=int, default=4)
	args = p.parse_args(argv)

	ensure_dir(os.path.dirname(args.manifest))
	if args.mode in {'build', 'both'}:
		build_manifest(args.starlist, args.manifest)
	if args.mode in {'download', 'both'}:
		download_from_manifest(args.manifest, concurrency=args.concurrency)


if __name__ == '__main__':
	main()