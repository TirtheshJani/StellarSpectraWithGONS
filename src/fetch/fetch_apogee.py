import os
import argparse
import io
from typing import List, Dict, Tuple, Optional

import pandas as pd
import requests

from .common import ensure_dir, write_manifest, parallel_download, verify_fits_basic, http_head, log_failures

# SAS base for DR17
SAS_REDUX_BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17"
SAS_DATA_BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/data/dr17"
SDSS_DR17_SQL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"


def query_sdss_sql(sql: str, fmt: str = 'json', timeout: int = 90) -> pd.DataFrame:
	params = {'cmd': sql, 'format': fmt}
	r = requests.get(SDSS_DR17_SQL, params=params, timeout=timeout)
	r.raise_for_status()
	if fmt == 'json':
		payload = r.json()
		rows = payload.get('Rows') if isinstance(payload, dict) else payload
		return pd.DataFrame(rows)
	return pd.read_csv(io.StringIO(r.text))


def chunked(items: List[str], size: int) -> List[List[str]]:
	return [items[i:i + size] for i in range(0, len(items), size)]


def fetch_apogee_meta(apogee_ids: List[str], chunk_size: int = 4000) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Return (stars_df, visits_df) with metadata needed to form SAS URLs.

	stars_df columns: apogee_id, telescope, field, location_id
	visits_df columns: apogee_id, telescope, plate, mjd, fiberid
	"""
	uids = sorted(set([x for x in apogee_ids if isinstance(x, str) and x.strip()]))
	stars_parts: List[pd.DataFrame] = []
	vis_parts: List[pd.DataFrame] = []
	for batch in chunked(uids, chunk_size):
		id_list = ",".join([f"'{i}'" for i in batch])
		star_qs = [
			f"SELECT apogee_id, telescope, field, location_id FROM apogeeStar WHERE apogee_id IN ({id_list})",
			f"SELECT apogee_id, telescope, field, location_id FROM aspcapStar WHERE apogee_id IN ({id_list})",
		]
		star_df: Optional[pd.DataFrame] = None
		for q in star_qs:
			try:
				df = query_sdss_sql(q, fmt='json')
				if not df.empty:
					star_df = df
					break
			except Exception:
				continue
		if star_df is not None and not star_df.empty:
			star_df = star_df[['apogee_id', 'telescope', 'field', 'location_id']].copy()
			stars_parts.append(star_df)

		vis_qs = [
			f"SELECT apogee_id, telescope, plate, mjd, fiberid FROM apogeeStarVisit WHERE apogee_id IN ({id_list})",
			f"SELECT apogee_id, telescope, plate, mjd, fiberid FROM apogeeVisit WHERE apogee_id IN ({id_list})",
		]
		vis_df: Optional[pd.DataFrame] = None
		for q in vis_qs:
			try:
				df = query_sdss_sql(q, fmt='json')
				if not df.empty:
					vis_df = df
					break
			except Exception:
				continue
		if vis_df is not None and not vis_df.empty:
			vis_df = vis_df[['apogee_id', 'telescope', 'plate', 'mjd', 'fiberid']].copy()
			vis_parts.append(vis_df)

	stars = pd.concat(stars_parts, ignore_index=True) if stars_parts else pd.DataFrame(columns=['apogee_id','telescope','field','location_id'])
	visits = pd.concat(vis_parts, ignore_index=True) if vis_parts else pd.DataFrame(columns=['apogee_id','telescope','plate','mjd','fiberid'])
	return stars.drop_duplicates(), visits.drop_duplicates()


def candidate_apstar_urls(apogee_id: str, telescope: Optional[str], field: Optional[str], location_id: Optional[int]) -> List[str]:
	urls: List[str] = []
	name = f"apStar-dr17-{apogee_id}.fits"
	if telescope and field:
		urls.append(f"{SAS_REDUX_BASE}/stars/{telescope}/{field}/{name}")
	# Sometimes field-less layout exists
	if telescope:
		urls.append(f"{SAS_REDUX_BASE}/stars/{telescope}/{name}")
	# Historical hashed/star directory layout fallback
	urls.append(f"{SAS_REDUX_BASE}/stars/{apogee_id[0]}/{apogee_id}/{name}")
	# Generic fallback
	urls.append(f"{SAS_REDUX_BASE}/stars/{name}")
	return list(dict.fromkeys(urls))


def candidate_apvisit_urls(telescope: str, plate: int, mjd: int, fiberid: int) -> List[str]:
	name = f"apVisit-dr17-{telescope}-{int(plate)}-{int(mjd)}-{int(fiberid):03d}.fits"
	urls = [
		f"{SAS_DATA_BASE}/{telescope}/{int(plate)}/{int(mjd)}/{name}",
	]
	return urls


def build_manifest(starlist_parquet: str, out_csv: str, base_dir: str = 'data') -> None:
	if not os.path.exists(starlist_parquet):
		raise FileNotFoundError(starlist_parquet)
	df = pd.read_parquet(starlist_parquet)
	if 'apogee_id' not in df.columns:
		raise ValueError("starlist parquet missing 'apogee_id'")
	ids = df['apogee_id'].dropna().astype(str).unique().tolist()
	try:
		stars, visits = fetch_apogee_meta(ids)
	except Exception:
		stars = pd.DataFrame(columns=['apogee_id','telescope','field','location_id'])
		visits = pd.DataFrame(columns=['apogee_id','telescope','plate','mjd','fiberid'])
	rows: List[Dict[str, object]] = []
	# apStar entries
	processed_ids = set()
	for _, r in stars.iterrows():
		apogee_id = str(r['apogee_id'])
		tel = str(r['telescope']) if pd.notna(r['telescope']) else None
		field = str(r['field']) if pd.notna(r['field']) else None
		locid = int(r['location_id']) if pd.notna(r['location_id']) else None
		for url in candidate_apstar_urls(apogee_id, tel, field, locid):
			status, size = http_head(url, timeout=20)
			if status == 200:
				local_path = os.path.join(base_dir, 'apogee', 'apStar', f"apStar-dr17-{apogee_id}.fits")
				rows.append({
					'remote_url': url,
					'local_path': local_path,
					'status': 'pending',
					'http_status': status,
					'bytes': size,
				})
				processed_ids.add(apogee_id)
				break
	# Fallback for apStar using hashed/generic paths if metadata failed
	for apogee_id in ids:
		if apogee_id in processed_ids:
			continue
		for url in candidate_apstar_urls(apogee_id, None, None, None):
			status, size = http_head(url, timeout=20)
			if status == 200:
				local_path = os.path.join(base_dir, 'apogee', 'apStar', f"apStar-dr17-{apogee_id}.fits")
				rows.append({
					'remote_url': url,
					'local_path': local_path,
					'status': 'pending',
					'http_status': status,
					'bytes': size,
				})
				break
	# apVisit entries
	for _, r in visits.iterrows():
		tel = str(r['telescope']) if pd.notna(r['telescope']) else None
		if not tel:
			continue
		plate = int(r['plate']) if pd.notna(r['plate']) else None
		mjd = int(r['mjd']) if pd.notna(r['mjd']) else None
		fiber = int(r['fiberid']) if pd.notna(r['fiberid']) else None
		if None in (plate, mjd, fiber):
			continue
		for url in candidate_apvisit_urls(tel, plate, mjd, fiber):
			status, size = http_head(url, timeout=20)
			if status == 200:
				name = f"apVisit-dr17-{tel}-{plate}-{mjd}-{fiber:03d}.fits"
				local_path = os.path.join(base_dir, 'apogee', 'apVisit', name)
				rows.append({
					'remote_url': url,
					'local_path': local_path,
					'status': 'pending',
					'http_status': status,
					'bytes': size,
				})
				break
	write_manifest(rows, out_csv)
	print(f"Wrote APOGEE manifest with {len(rows)} entries -> {out_csv}")


def download_from_manifest(manifest_csv: str, concurrency: int = 8, downloader: str = 'python') -> None:
	import csv
	pairs: List[Tuple[str, str]] = []
	with open(manifest_csv, 'r') as f:
		r = csv.DictReader(f)
		for row in r:
			pairs.append((row['remote_url'], row['local_path']))
	results = parallel_download(pairs, concurrency=concurrency, timeout=180,
								 verify_cb=lambda p: verify_fits_basic(p, required_headers=['TELESCOP']),
								 downloader=downloader)
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
	log_failures(results, manifest_csv.replace('.csv', '.failures.log'))
	ok = sum(1 for r in results if r.status == 'ok')
	print(f"Downloaded {ok}/{len(results)} OK from manifest")


def main(argv: List[str] = None) -> None:
	p = argparse.ArgumentParser(description='APOGEE DR17 manifest builder and downloader')
	p.add_argument('--starlist', default=os.path.join('data', 'common', 'manifests', 'starlist_30k.parquet'))
	p.add_argument('--manifest', default=os.path.join('data', 'apogee', 'manifests', 'apogee_manifest.csv'))
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