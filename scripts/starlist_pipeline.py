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


def query_sdss_sql(sql: str, fmt: str = 'json', timeout: int = 60, retries: int = 5) -> pd.DataFrame:
	"""Query SDSS SkyServer SQL endpoint and return a DataFrame.

	fmt can be 'json', 'csv', or 'tsv'.
	"""
	params = {'cmd': sql, 'format': fmt}
	headers = {
		'Connection': 'close',
		'Content-Type': 'application/x-www-form-urlencoded',
		'User-Agent': 'StellarSpectraWithGONS/1.0 (+https://github.com/)'
	}
	last_exc: Optional[Exception] = None
	for attempt in range(max(1, retries)):
		try:
			# Use POST to avoid overly long URLs when querying many IDs
			resp = requests.post(SDSS_DR17_SQL, data=params, timeout=timeout, headers=headers)
			resp.raise_for_status()
			break
		except Exception as e:
			last_exc = e
			# Exponential backoff: 0.5, 1, 2, 4, ... seconds
			sleep_s = min(8.0, 0.5 * (2 ** attempt))
			try:
				import time as _time
				_time.sleep(sleep_s)
			except Exception:
				pass
	else:
		# Exhausted retries
		raise last_exc if last_exc else RuntimeError('Unknown SDSS query failure')
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


def fetch_apogee_positions(apogee_ids: Sequence[str], chunk_size: int = 1000, timeout: int = 90, retries: int = 5, sleep_ms: int = 0) -> pd.DataFrame:
	"""Fetch RA/Dec for given APOGEE IDs from DR17.

	Tries apogeeStar first; falls back to apogeeObject/aspcapStar if needed.
	"""
	clean_ids = [normalize_bstring(x) for x in apogee_ids if normalize_bstring(x)]
	unique_ids = sorted(set(clean_ids))
	print(f"Querying DR17 for {len(unique_ids)} unique APOGEE IDs in chunks of {chunk_size}")

	results: List[pd.DataFrame] = []

	def _sleep_between_requests() -> None:
		if sleep_ms and sleep_ms > 0:
			try:
				import time as _time
				_time.sleep(max(0.0, sleep_ms) / 1000.0)
			except Exception:
				pass

	def _try_fetch_for_ids(ids: Sequence[str]) -> Optional[pd.DataFrame]:
		id_list_local = ",".join([f"'{i}'" for i in ids])
		queries_local = [
			f"SELECT apogee_id, ra, dec FROM apogeeStar WHERE apogee_id IN ({id_list_local})",
			f"SELECT apogee_id, ra, dec FROM aspcapStar WHERE apogee_id IN ({id_list_local})",
			f"SELECT apogee_id, ra, dec FROM apogeeObject WHERE apogee_id IN ({id_list_local})",
		]
		last_err_local: Optional[Exception] = None
		for q in queries_local:
			try:
				df_local = query_sdss_sql(q, fmt='csv', timeout=timeout, retries=retries)
				if not df_local.empty and {'apogee_id', 'ra', 'dec'}.issubset(set(df_local.columns)):
					return df_local[['apogee_id', 'ra', 'dec']].copy()
			except Exception as e:
				last_err_local = e
		# If all queries failed and we have multiple IDs, recursively split
		if len(ids) > 1:
			midpoint = len(ids) // 2
			left = _try_fetch_for_ids(ids[:midpoint])
			_sleep_between_requests()
			right = _try_fetch_for_ids(ids[midpoint:])
			if left is None and right is None:
				return None
			frames = []
			if left is not None:
				frames.append(left)
			if right is not None:
				frames.append(right)
			return pd.concat(frames, ignore_index=True) if frames else None
		# Single ID failed completely
		if last_err_local:
			print(f"Warning: DR17 query failed for {ids[0]} ; last error: {last_err_local}")
		return None

	for batch in chunked(unique_ids, chunk_size):
		batch_df = _try_fetch_for_ids(batch)
		_sleep_between_requests()
		if batch_df is None or batch_df.empty:
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
				  target_size: int = 30000,
				  dr17_chunk_size: int = 1000,
				  dr17_timeout: int = 90,
				  dr17_retries: int = 5,
				  dr17_sleep_ms: int = 0) -> None:
	# T1.1
	apogee_ids_df = ingest_apogee_ids(apogee_csv)
	print(f"Loaded {len(apogee_ids_df):,} APOGEE candidate IDs from {apogee_csv}")

	# T1.2
	apogee_positions = fetch_apogee_positions(
		apogee_ids=apogee_ids_df['apogee_id'].tolist(),
		chunk_size=dr17_chunk_size,
		timeout=dr17_timeout,
		retries=dr17_retries,
		sleep_ms=dr17_sleep_ms,
	)
	store_apogee_positions(apogee_positions, apogee_positions_out)

	# T1.3 – Crossmatch GALAH & GES to the APOGEE list without downselection
	# Allow skipping GALAH/GES if not provided; downstream downloaders can use positional queries.
	if galah_positions is not None:
		galah_pos = load_galah_positions(galah_positions)
		ensure_dir(os.path.dirname(galah_positions_out))
		galah_pos.to_parquet(galah_positions_out, index=False)
		print(f"Wrote GALAH positions to {galah_positions_out}")
	else:
		galah_pos = pd.DataFrame(columns=['galah_id', 'ra', 'dec'])

	if ges_positions is not None:
		ges_pos = load_ges_positions(ges_positions)
		ensure_dir(os.path.dirname(ges_positions_out))
		ges_pos.to_parquet(ges_positions_out, index=False)
		print(f"Wrote GES positions to {ges_positions_out}")
	else:
		ges_pos = pd.DataFrame(columns=['ges_id', 'ra', 'dec'])

	# Two separate left-joins on sky to keep exactly the APOGEE set
	m_ag = spherical_match(apogee_positions, galah_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=1.0)
	m_ag = resolve_duplicates(m_ag)
	m_ae = spherical_match(apogee_positions, ges_pos, 'ra', 'dec', 'ra', 'dec', max_arcsec=1.0)
	m_ae = resolve_duplicates(m_ae)

	# Prepare arrays aligned to apogee_positions index
	galah_id_aligned = pd.Series([None] * len(apogee_positions))
	sep_ag_aligned = pd.Series([np.nan] * len(apogee_positions), dtype=float)
	if not m_ag.empty:
		galah_id_aligned.iloc[m_ag['left_ix'].to_numpy()] = galah_pos.iloc[m_ag['right_ix'].to_numpy()]['galah_id'].to_numpy()
		sep_ag_aligned.iloc[m_ag['left_ix'].to_numpy()] = m_ag['sep_arcsec'].to_numpy()

	ges_id_aligned = pd.Series([None] * len(apogee_positions))
	sep_ae_aligned = pd.Series([np.nan] * len(apogee_positions), dtype=float)
	if not m_ae.empty:
		ges_id_aligned.iloc[m_ae['left_ix'].to_numpy()] = ges_pos.iloc[m_ae['right_ix'].to_numpy()]['ges_id'].to_numpy()
		sep_ae_aligned.iloc[m_ae['left_ix'].to_numpy()] = m_ae['sep_arcsec'].to_numpy()

	final = pd.DataFrame({
		'source_id': pd.Series(dtype='float64'),
		'ra': apogee_positions['ra'].astype(float).to_numpy(),
		'dec': apogee_positions['dec'].astype(float).to_numpy(),
		'apogee_id': apogee_positions['apogee_id'].to_numpy(),
		'galah_id': galah_id_aligned.to_numpy(object),
		'ges_id': ges_id_aligned.to_numpy(object),
		'sep_ag': sep_ag_aligned.to_numpy(float),
		'sep_ae': sep_ae_aligned.to_numpy(float),
	})

	# Ensure acceptance schema:
	# star_id, ra, dec, apogee_id, galah_id, ges_id,
	# in_all_three (bool), in_apogee, in_galah, in_ges, match_sep_arcsec
	work = final.copy()
	# Compute booleans
	work['in_apogee'] = work['apogee_id'].notna()
	work['in_galah'] = work['galah_id'].notna()
	work['in_ges'] = work['ges_id'].notna()
	work['in_all_three'] = work['in_apogee'] & work['in_galah'] & work['in_ges']
	# One separation field: prefer the larger of available separations for conservative reporting,
	# otherwise whichever exists
	sep_ag = work['sep_ag'] if 'sep_ag' in work.columns else pd.Series([np.nan] * len(work))
	sep_ae = work['sep_ae'] if 'sep_ae' in work.columns else pd.Series([np.nan] * len(work))
	work['match_sep_arcsec'] = np.fmax(sep_ag.fillna(-np.inf), sep_ae.fillna(-np.inf))
	work.loc[~np.isfinite(work['match_sep_arcsec']), 'match_sep_arcsec'] = sep_ag.combine_first(sep_ae)
	# Create deterministic star_id 1..N
	work = work.reset_index(drop=True)
	work.insert(0, 'star_id', (work.index + 1).astype(np.int64))
	# Enforce exactly target_size rows if possible
	if len(work) > target_size:
		work = stratified_sample(work, target_size)
		work = work.reset_index(drop=True)
		work['star_id'] = (work.index + 1).astype(np.int64)
	# Select and order columns
	cols = ['star_id', 'ra', 'dec', 'apogee_id', 'galah_id', 'ges_id',
			'in_all_three', 'in_apogee', 'in_galah', 'in_ges', 'match_sep_arcsec']
	missing_cols = [c for c in cols if c not in work.columns]
	for c in missing_cols:
		work[c] = pd.Series([np.nan] * len(work))
	out_df = work[cols]
	# Save
	ensure_dir(os.path.dirname(starlist_out))
	out_df.to_parquet(starlist_out, index=False)
	print(f"Wrote final star list ({len(out_df):,} rows) to {starlist_out}")


# ------------------------------
# CLI
# ------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Build consolidated 30k star list from APOGEE/GALAH/GES')
	p.add_argument('--apogee-csv', default=os.path.join('scripts', 'Apogee_ID.csv'), help='Path to Apogee_ID.csv')
	p.add_argument('--galah', default=None, help='Path to GALAH DR3 positions CSV/Parquet (optional)')
	p.add_argument('--ges', default=None, help='Path to GES DR4 UVES positions CSV/Parquet (optional)')
	p.add_argument('--apogee-out', default=os.path.join('data', 'apogee', 'manifests', 'apogee_positions.parquet'))
	p.add_argument('--galah-out', default=os.path.join('data', 'galah', 'manifests', 'galah_positions.parquet'))
	p.add_argument('--ges-out', default=os.path.join('data', 'ges', 'manifests', 'ges_positions.parquet'))
	p.add_argument('--starlist-out', default=os.path.join('data', 'common', 'manifests', 'starlist_30k.parquet'))
	p.add_argument('--target-size', type=int, default=30000)
	p.add_argument('--mode', choices=['full', 'apogee-only'], default='full', help='full pipeline or only fetch APOGEE positions')
	# Network tuning
	p.add_argument('--dr17-chunk-size', type=int, default=1000, help='APOGEE DR17 query chunk size')
	p.add_argument('--dr17-timeout', type=int, default=90, help='HTTP timeout seconds for DR17 queries')
	p.add_argument('--dr17-retries', type=int, default=5, help='Number of retries per DR17 request')
	p.add_argument('--dr17-sleep-ms', type=int, default=0, help='Sleep milliseconds between DR17 requests')
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
	args = parse_args(argv)
	if args.mode == 'apogee-only':
		apogee_ids_df = ingest_apogee_ids(args.apogee_csv)
		apogee_positions = fetch_apogee_positions(
			apogee_ids=apogee_ids_df['apogee_id'].tolist(),
			chunk_size=args.dr17_chunk_size,
			timeout=args.dr17_timeout,
			retries=args.dr17_retries,
			sleep_ms=args.dr17_sleep_ms,
		)
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
		dr17_chunk_size=args.dr17_chunk_size,
		dr17_timeout=args.dr17_timeout,
		dr17_retries=args.dr17_retries,
		dr17_sleep_ms=args.dr17_sleep_ms,
	)


if __name__ == '__main__':
	main()