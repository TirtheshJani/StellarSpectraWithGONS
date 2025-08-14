#!/usr/bin/env python3
import argparse
import io
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord

# Local crossmatch utility
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from xmatch import xmatch  # noqa: E402


# ------------------------------
# Utilities
# ------------------------------

BSTR_RE = re.compile(r"^b'(.*)'$")


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def normalize_bstring(value: object) -> Optional[str]:
	"""Convert values like b'2M....' or bytes to plain Python str.

	Returns None for NaN/None/empty strings.
	"""
	if value is None:
		return None
	if isinstance(value, bytes):
		try:
			return value.decode('utf-8')
		except Exception:
			return value.decode('latin-1', errors='ignore')
	text = str(value).strip()
	if text == '' or text.lower() == 'nan' or text.lower() == 'none':
		return None
	m = BSTR_RE.match(text)
	if m:
		return m.group(1)
	return text


# ------------------------------
# T1.1 – Consolidate candidate IDs
# ------------------------------

def ingest_apogee_ids(apogee_csv: str) -> pd.DataFrame:
	"""Load the APOGEE ID list and normalize columns.

	Expected to find at least a column like APOGEE_ID/APOGEEID/apogee_id.
	We will also attempt to pick RA/Dec columns if present, but positions
	will be re-fetched from SDSS DR17 in T1.2.
	"""
	if not os.path.exists(apogee_csv):
		raise FileNotFoundError(f"APOGEE CSV not found at {apogee_csv}")

	# Be permissive with separators and encodings
	df = pd.read_csv(apogee_csv, low_memory=False)

	# Identify APOGEE ID column
	candidate_cols = [
		c for c in df.columns
		if c.strip().lower() in {'apogee_id', 'apogeeid', 'apogee id', 'apogee', 'aspcap_id'}
	]
	if not candidate_cols:
		# Try common case where column is named like b'APOGEE_ID' after bytes in CSV
		for c in df.columns:
			if 'apogee' in c.lower():
				candidate_cols.append(c)
				break
	if not candidate_cols:
		raise ValueError(
			"Could not find an APOGEE ID column in the input CSV. Expected one of:"
			" apogee_id, APOGEE_ID, apogee, aspcap_id"
		)
	apogee_col = candidate_cols[0]

	# Normalize APOGEE IDs
	df['apogee_id'] = df[apogee_col].apply(normalize_bstring)
	# Drop rows without IDs
	before = len(df)
	df = df[df['apogee_id'].notna()].copy()
	if len(df) < before:
		print(f"Dropped {before - len(df)} rows without APOGEE IDs")

	# Try to capture RA/Dec if present
	ra_col = None
	dec_col = None
	for c in df.columns:
		lc = c.lower().strip()
		if lc in {'ra', 'alpha', 'ra_icrs', 'raj2000'}:
			ra_col = c
		if lc in {'dec', 'delta', 'dec_icrs', 'dej2000'}:
			dec_col = c

	result = pd.DataFrame({
		'apogee_id': df['apogee_id']
	})
	result['ra'] = df[ra_col].astype(float) if ra_col else np.nan
	result['dec'] = df[dec_col].astype(float) if dec_col else np.nan

	# Standardize output schema for consolidation stage
	result['source_id'] = pd.Series(dtype='float64')
	result['galah_id'] = pd.Series(dtype='object')
	result['ges_id'] = pd.Series(dtype='object')

	# Reorder columns
	result = result[['source_id', 'ra', 'dec', 'apogee_id', 'galah_id', 'ges_id']]
	return result.drop_duplicates(subset=['apogee_id']).reset_index(drop=True)


# ------------------------------
# T1.2 – Get positions for APOGEE IDs via SDSS SkyServer DR17
# ------------------------------

SDSS_DR17_SQL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"


def query_sdss_sql(sql: str, fmt: str = 'json', timeout: int = 60) -> pd.DataFrame:
	"""Query SDSS SkyServer SQL endpoint and return a DataFrame.

	fmt can be 'json', 'csv', or 'tsv'.
	"""
	params = {'cmd': sql, 'format': fmt}
	resp = requests.get(SDSS_DR17_SQL, params=params, timeout=timeout)
	resp.raise_for_status()
	if fmt == 'json':
		payload = resp.json()
		# Expect a list of dicts under 'Rows' or direct list
		rows = payload.get('Rows') if isinstance(payload, dict) else payload
		return pd.DataFrame(rows)
	else:
		return pd.read_csv(io.StringIO(resp.text))


def chunked(iterable: Sequence[str], size: int) -> Iterable[List[str]]:
	for i in range(0, len(iterable), size):
		yield list(iterable[i:i + size])


def fetch_apogee_positions(apogee_ids: Sequence[str], chunk_size: int = 4000) -> pd.DataFrame:
	"""Fetch RA/Dec for given APOGEE IDs from DR17.

	Tries apogeeStar first; falls back to apogeeObject/aspcapStar if needed.
	"""
	clean_ids = [normalize_bstring(x) for x in apogee_ids if normalize_bstring(x)]
	unique_ids = sorted(set(clean_ids))
	print(f"Querying DR17 for {len(unique_ids)} unique APOGEE IDs in chunks of {chunk_size}")

	results: List[pd.DataFrame] = []
	for batch in chunked(unique_ids, chunk_size):
		id_list = ",".join([f"'{i}'" for i in batch])
		queries = [
			# Preferred: apogeeStar view with ra/dec
			f"SELECT apogee_id, ra, dec FROM apogeeStar WHERE apogee_id IN ({id_list})",
			# Fallback: aspcapStar for some DRs
			f"SELECT apogee_id, ra, dec FROM aspcapStar WHERE apogee_id IN ({id_list})",
			# Fallback: apogeeObject stores location_id based positions in some DRs
			f"SELECT apogee_id, ra, dec FROM apogeeObject WHERE apogee_id IN ({id_list})",
		]
		batch_df = None
		last_err: Optional[Exception] = None
		for q in queries:
			try:
				df = query_sdss_sql(q, fmt='json')
				if not df.empty and {'apogee_id', 'ra', 'dec'}.issubset(set(df.columns)):
					batch_df = df[['apogee_id', 'ra', 'dec']].copy()
					break
			except Exception as e:  # keep trying fallbacks
				last_err = e
		if batch_df is None:
			if last_err:
				print(f"Warning: DR17 SQL batch failed; last error: {last_err}")
			else:
				print("Warning: DR17 SQL batch returned no data")
			continue
		# Normalize types
		batch_df['apogee_id'] = batch_df['apogee_id'].apply(normalize_bstring)
		batch_df['ra'] = batch_df['ra'].astype(float)
		batch_df['dec'] = batch_df['dec'].astype(float)
		results.append(batch_df)

	if not results:
		raise RuntimeError("No APOGEE positions were retrieved from DR17. Check connectivity or IDs.")

	full = pd.concat(results, ignore_index=True)
	full = full.dropna(subset=['apogee_id']).drop_duplicates(subset=['apogee_id'])
	return full


def store_apogee_positions(df: pd.DataFrame, out_path: str) -> None:
	ensure_dir(os.path.dirname(out_path))
	# Normalize schema to requested columns
	out = pd.DataFrame({
		'source_id': pd.Series(dtype='float64'),
		'ra': df['ra'].astype(float),
		'dec': df['dec'].astype(float),
		'apogee_id': df['apogee_id'].apply(normalize_bstring),
		'galah_id': pd.Series(dtype='object'),
		'ges_id': pd.Series(dtype='object'),
	})
	out.to_parquet(out_path, index=False)
	print(f"Wrote {len(out):,} APOGEE positions to {out_path}")


# ------------------------------
# T1.3 – Crossmatch to GALAH DR3 & Gaia-ESO DR4
# ------------------------------

@dataclass
class CatalogueSpec:
	name: str
	id_col: str
	ra_col: str
	dec_col: str
	local_glob: Optional[str] = None


def load_galah_positions(local_path: Optional[str] = None) -> pd.DataFrame:
	"""Load or retrieve GALAH DR3 positions.

	Priority:
	1) local_path if provided
	2) /mnt/data/GALAH_DR3_positions.csv or parquet if present
	3) raise with instructions for obtaining the DR3 subset
	"""
	candidates = []
	if local_path:
		candidates.append(local_path)
	candidates.extend([
		'/mnt/data/GALAH_DR3_positions.parquet',
		'/mnt/data/GALAH_DR3_positions.csv',
	])
	for p in candidates:
		if os.path.exists(p):
			if p.endswith('.parquet'):
				df = pd.read_parquet(p)
			else:
				df = pd.read_csv(p)
			break
	else:
		raise FileNotFoundError(
			"GALAH DR3 positions not found. Provide a subset CSV/Parquet with columns "
			"['sobject_id' or 'galah_id', 'ra', 'dec'] at /mnt/data/GALAH_DR3_positions.* or via --galah"
		)

	# Normalize columns
	colmap = {}
	for c in df.columns:
		lc = c.lower()
		if lc in {'sobject_id', 'galah_id'}:
			colmap['galah_id'] = c
		elif lc in {'ra', 'ra_icrs', 'raj2000'}:
			colmap['ra'] = c
		elif lc in {'dec', 'dec_icrs', 'dej2000'}:
			colmap['dec'] = c
	missing = {'galah_id', 'ra', 'dec'} - set(colmap)
	if missing:
		raise ValueError(f"GALAH positions missing columns: {missing}")

	out = pd.DataFrame({
		'galah_id': df[colmap['galah_id']].apply(normalize_bstring),
		'ra': df[colmap['ra']].astype(float),
		'dec': df[colmap['dec']].astype(float),
	})
	return out.dropna(subset=['galah_id']).drop_duplicates(subset=['galah_id']).reset_index(drop=True)


def load_ges_positions(local_path: Optional[str] = None) -> pd.DataFrame:
	"""Load or retrieve Gaia-ESO DR4 UVES positions.

	Priority:
	1) local_path if provided
	2) /mnt/data/GES_DR4_UVES_positions.csv or parquet if present
	3) raise with instructions for obtaining the DR4 subset
	"""
	candidates = []
	if local_path:
		candidates.append(local_path)
	candidates.extend([
		'/mnt/data/GES_DR4_UVES_positions.parquet',
		'/mnt/data/GES_DR4_UVES_positions.csv',
	])
	for p in candidates:
		if os.path.exists(p):
			if p.endswith('.parquet'):
				df = pd.read_parquet(p)
			else:
				df = pd.read_csv(p)
			break
	else:
		raise FileNotFoundError(
			"GES DR4 UVES positions not found. Provide a subset CSV/Parquet with columns "
			"['ges_id' or 'objid', 'ra', 'dec'] at /mnt/data/GES_DR4_UVES_positions.* or via --ges"
		)

	# Normalize columns
	colmap = {}
	for c in df.columns:
		lc = c.lower()
		if lc in {'ges_id', 'objid', 'target', 'object', 'object_id'}:
			colmap['ges_id'] = c
		elif lc in {'ra', 'ra_icrs', 'raj2000'}:
			colmap['ra'] = c
		elif lc in {'dec', 'dec_icrs', 'dej2000'}:
			colmap['dec'] = c
	missing = {'ges_id', 'ra', 'dec'} - set(colmap)
	if missing:
		raise ValueError(f"GES positions missing columns: {missing}")

	out = pd.DataFrame({
		'ges_id': df[colmap['ges_id']].apply(normalize_bstring),
		'ra': df[colmap['ra']].astype(float),
		'dec': df[colmap['dec']].astype(float),
	})
	return out.dropna(subset=['ges_id']).drop_duplicates(subset=['ges_id']).reset_index(drop=True)


# ------------------------------
# Crossmatch helpers
# ------------------------------


def spherical_match(left: pd.DataFrame, right: pd.DataFrame,
				    left_ra: str, left_dec: str,
				    right_ra: str, right_dec: str,
				    max_arcsec: float = 1.0) -> pd.DataFrame:
	"""Return a DataFrame mapping left index to right index with separations.

	Uses astropy's match_to_catalog_sky via our xmatch wrapper.
	"""
	m1_idx, m2_idx, sep = xmatch(
		left[left_ra].to_numpy(float),
		left[left_dec].to_numpy(float),
		right[right_ra].to_numpy(float),
		right[right_dec].to_numpy(float),
		maxdist=max_arcsec,
	)
	return pd.DataFrame({'left_ix': m1_idx, 'right_ix': m2_idx, 'sep_arcsec': sep.to(u.arcsec).value})


def resolve_duplicates(mapping: pd.DataFrame, prefer_cols: Optional[List[str]] = None,
					   prefer_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
	"""Resolve duplicate matches by minimum separation; optionally use prefer_cols from prefer_df.

	- mapping columns: left_ix, right_ix, sep_arcsec
	- If multiple right_ix for a left_ix, keep the one with smallest sep_arcsec; if equal and prefer_cols
	  provided and exist in prefer_df (e.g., S/N), prefer higher values.
	"""
	if mapping.empty:
		return mapping

	mapping = mapping.sort_values('sep_arcsec').copy()
	# If S/N is available
	if prefer_cols and prefer_df is not None:
		for col in prefer_cols:
			if col in prefer_df.columns:
				# Attach and sort by -col to prefer higher values
				mapping[col] = prefer_df.iloc[mapping['right_ix'].to_numpy()][col].to_numpy()
				mapping = mapping.sort_values(['left_ix', 'sep_arcsec', col], ascending=[True, True, False])
				break
	# Keep the first match per left_ix
	return mapping.drop_duplicates(subset=['left_ix'], keep='first')


def build_three_way(apogee_pos: pd.DataFrame,
				   galah_pos: pd.DataFrame,
				   ges_pos: pd.DataFrame,
				   max_arcsec_primary: float = 1.0,
				   max_arcsec_fallback: float = 2.0) -> pd.DataFrame:
	"""Three-way crossmatch APOGEE–GALAH–GES returning unified rows.

	Resolves to unique matches by min separation (and S/N if provided in the position tables).
	"""
	# 1) APOGEE x GALAH
	m_ag = spherical_match(apogee_pos, galah_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=max_arcsec_primary)
	m_ag = resolve_duplicates(m_ag)

	# 2) APOGEE x GES
	m_ae = spherical_match(apogee_pos, ges_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=max_arcsec_primary)
	m_ae = resolve_duplicates(m_ae)

	# Merge on left_ix (APOGEE index)
	triple = pd.merge(m_ag, m_ae, on='left_ix', how='inner', suffixes=('_ag', '_ae'))
	if triple.empty and max_arcsec_fallback and max_arcsec_fallback > max_arcsec_primary:
		# Try fallback radius
		m_ag2 = spherical_match(apogee_pos, galah_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=max_arcsec_fallback)
		m_ag2 = resolve_duplicates(m_ag2)
		m_ae2 = spherical_match(apogee_pos, ges_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=max_arcsec_fallback)
		m_ae2 = resolve_duplicates(m_ae2)
		triple = pd.merge(m_ag2, m_ae2, on='left_ix', how='inner', suffixes=('_ag', '_ae'))

	# Construct unified DataFrame
	rows = []
	for _, r in triple.iterrows():
		ap_ix = int(r['left_ix'])
		ga_ix = int(r['right_ix_ag'])
		ge_ix = int(r['right_ix_ae'])
		rows.append({
			'ra': float(apogee_pos.iloc[ap_ix]['ra']),
			'dec': float(apogee_pos.iloc[ap_ix]['dec']),
			'apogee_id': apogee_pos.iloc[ap_ix]['apogee_id'],
			'galah_id': galah_pos.iloc[ga_ix].get('galah_id', None),
			'ges_id': ges_pos.iloc[ge_ix].get('ges_id', None),
			'sep_ag': float(r['sep_arcsec_ag']),
			'sep_ae': float(r['sep_arcsec_ae'])
		})
	result = pd.DataFrame(rows)

	# Attach Gaia-style source_id if available in GALAH or GES positions
	result['source_id'] = pd.Series(dtype='float64')
	cols_order = ['source_id', 'ra', 'dec', 'apogee_id', 'galah_id', 'ges_id', 'sep_ag', 'sep_ae']
	return result[cols_order]


# ------------------------------
# T1.4 – Downselect to 30,000
# ------------------------------


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
	"""Stratify by Teff/logg/[Fe/H]/SNR if available; else uniform random sample.

	The function expects columns among ['teff', 'logg', 'fe_h', 'snr'] (case-insensitive).
	If none present, performs uniform sampling.
	"""
	rng = np.random.default_rng(seed)
	cols = {c.lower(): c for c in df.columns}
	feat_names = []
	for k in ['teff', 'logg', 'fe_h', 'feh', 'snr']:
		if k in cols:
			feat_names.append(cols[k])
	if not feat_names:
		return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

	# Build bins per feature
	df_work = df.copy()
	for col in feat_names:
		series = df_work[col]
		try:
			bins = pd.qcut(series.rank(method='first'), q=min(5, series.notna().sum()), labels=False, duplicates='drop')
		except Exception:
			bins = pd.cut(series, bins=5, labels=False)
		df_work[f'bin_{col}'] = bins

	grp_cols = [f'bin_{c}' for c in feat_names]
	groups = df_work.groupby(grp_cols, dropna=True)
	# Determine per-group quota
	quotalist = []
	remaining = n
	for gkey, gdf in groups:
		if remaining <= 0:
			break
		quota = max(1, math.floor(n * len(gdf) / len(df_work)))
		quota = min(quota, len(gdf))
		quotalist.append((gkey, quota))
		resting = quota
		remaining -= quota
	# Sample each group
	sampled = []
	for gkey, quota in quotalist:
		gdf = groups.get_group(gkey)
		if len(gdf) <= quota:
			sampled.append(gdf)
		else:
			sampled.append(gdf.sample(n=quota, random_state=seed))
	out = pd.concat(sampled, ignore_index=True)
	if len(out) > n:
		out = out.sample(n=n, random_state=seed)
	return out.reset_index(drop=True)


# ------------------------------
# Orchestration
# ------------------------------


def run_pipeline(apogee_csv: str,
				  galah_positions: Optional[str],
				  ges_positions: Optional[str],
				  apogee_positions_out: str,
				  galah_positions_out: str,
				  ges_positions_out: str,
				  starlist_out: str,
				  target_size: int = 30000) -> None:
	# T1.1
	apogee_ids_df = ingest_apogee_ids(apogee_csv)
	print(f"Loaded {len(apogee_ids_df):,} APOGEE candidate IDs from {apogee_csv}")

	# T1.2
	apogee_positions = fetch_apogee_positions(apogee_ids_df['apogee_id'].tolist())
	store_apogee_positions(apogee_positions, apogee_positions_out)

	# T1.3
	galah_pos = load_galah_positions(galah_positions)
	ensure_dir(os.path.dirname(galah_positions_out))
	galah_pos.to_parquet(galah_positions_out, index=False)
	print(f"Wrote GALAH positions to {galah_positions_out}")

	ges_pos = load_ges_positions(ges_positions)
	ensure_dir(os.path.dirname(ges_positions_out))
	ges_pos.to_parquet(ges_positions_out, index=False)
	print(f"Wrote GES positions to {ges_positions_out}")

	triple = build_three_way(
		apogee_pos=apogee_positions[['apogee_id', 'ra', 'dec']].copy(),
		galah_pos=galah_pos[['galah_id', 'ra', 'dec']].copy(),
		ges_pos=ges_pos[['ges_id', 'ra', 'dec']].copy(),
	)

	# T1.4 – Downselect to exactly 30,000
	if len(triple) >= target_size:
		final = stratified_sample(triple, target_size)
	else:
		# Augment with two-way matches flags
		final = triple.copy()
		final['match_flag'] = 'three-way'
		missing = target_size - len(final)
		print(f"Three-way matches: {len(triple):,}; need to augment {missing:,} from two-way matches")
		# Two-way APOGEE–GALAH
		m_ag = spherical_match(apogee_positions, galah_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=1.0)
		m_ag = resolve_duplicates(m_ag)
		df_ag = pd.DataFrame({
			'ra': apogee_positions.iloc[m_ag['left_ix']]['ra'].to_numpy(),
			'dec': apogee_positions.iloc[m_ag['left_ix']]['dec'].to_numpy(),
			'apogee_id': apogee_positions.iloc[m_ag['left_ix']]['apogee_id'].to_numpy(),
			'galah_id': galah_pos.iloc[m_ag['right_ix']]['galah_id'].to_numpy(),
			'ges_id': pd.Series([None] * len(m_ag)),
			'sep_ag': m_ag['sep_arcsec'].to_numpy(),
			'sep_ae': pd.Series([np.nan] * len(m_ag)),
			'source_id': pd.Series([np.nan] * len(m_ag)),
			'match_flag': 'two-way-ag',
		})
		# Two-way APOGEE–GES
		m_ae = spherical_match(apogee_positions, ges_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=1.0)
		m_ae = resolve_duplicates(m_ae)
		df_ae = pd.DataFrame({
			'ra': apogee_positions.iloc[m_ae['left_ix']]['ra'].to_numpy(),
			'dec': apogee_positions.iloc[m_ae['left_ix']]['dec'].to_numpy(),
			'apogee_id': apogee_positions.iloc[m_ae['left_ix']]['apogee_id'].to_numpy(),
			'galah_id': pd.Series([None] * len(m_ae)),
			'ges_id': ges_pos.iloc[m_ae['right_ix']]['ges_id'].to_numpy(),
			'sep_ag': pd.Series([np.nan] * len(m_ae)),
			'sep_ae': m_ae['sep_arcsec'].to_numpy(),
			'source_id': pd.Series([np.nan] * len(m_ae)),
			'match_flag': 'two-way-ae',
		})
		aug = pd.concat([df_ag, df_ae], ignore_index=True)
		# Remove any rows that already present in three-way
		if not final.empty:
			already = set(zip(final['apogee_id'], final['galah_id'], final['ges_id']))
			aug = aug[[tuple(x) not in already for x in zip(aug['apogee_id'], aug['galah_id'], aug['ges_id'])]]
		need = max(0, target_size - len(final))
		if need > 0 and not aug.empty:
			final = pd.concat([final, aug.sample(n=min(need, len(aug)), random_state=42)], ignore_index=True)
		if len(final) < target_size:
			print(f"Warning: only {len(final):,} rows after augmentation; fewer than target {target_size:,}")

	# Ensure final schema and save
	final = final[['source_id', 'ra', 'dec', 'apogee_id', 'galah_id', 'ges_id'] +
				  [c for c in final.columns if c not in {'source_id', 'ra', 'dec', 'apogee_id', 'galah_id', 'ges_id'}]]
	ensure_dir(os.path.dirname(starlist_out))
	final.to_parquet(starlist_out, index=False)
	print(f"Wrote final star list ({len(final):,} rows) to {starlist_out}")


# ------------------------------
# CLI
# ------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Build consolidated 30k star list from APOGEE/GALAH/GES')
	p.add_argument('--apogee-csv', default='/mnt/data/Apogee_ID.csv', help='Path to Apogee_ID.csv')
	p.add_argument('--galah', default=None, help='Path to GALAH DR3 positions CSV/Parquet (optional)')
	p.add_argument('--ges', default=None, help='Path to GES DR4 UVES positions CSV/Parquet (optional)')
	p.add_argument('--apogee-out', default='/workspace/data/apogee/manifests/apogee_positions.parquet')
	p.add_argument('--galah-out', default='/workspace/data/galah/manifests/galah_positions.parquet')
	p.add_argument('--ges-out', default='/workspace/data/ges/manifests/ges_positions.parquet')
	p.add_argument('--starlist-out', default='/workspace/data/common/manifests/starlist_30k.parquet')
	p.add_argument('--target-size', type=int, default=30000)
	p.add_argument('--mode', choices=['full', 'apogee-only'], default='full', help='full pipeline or only fetch APOGEE positions')
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
	args = parse_args(argv)
	if args.mode == 'apogee-only':
		apogee_ids_df = ingest_apogee_ids(args.apogee_csv)
		apogee_positions = fetch_apogee_positions(apogee_ids_df['apogee_id'].tolist())
		store_apogee_positions(apogee_positions, args.apogee_out)
		return
	# default full
	run_pipeline(
		apogee_csv=args.apogee_csv,
		galah_positions=args.galah,
		ges_positions=args.ges,
		apogee_positions_out=args.apogee_out,
		galah_positions_out=args.galah_out,
		ges_positions_out=args.ges_out,
		starlist_out=args.starlist_out,
		target_size=args.target_size,
	)


if __name__ == '__main__':
	main()