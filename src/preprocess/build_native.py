import os
import argparse
import csv
from typing import List, Dict

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from ..fetch.common import ensure_dir
from .readers import read_apogee_apstar, read_galah_camera, read_ges_uves


def build_records_from_manifest(manifest_csv: str, survey: str) -> List[Dict[str, object]]:
	records: List[Dict[str, object]] = []
	with open(manifest_csv, 'r') as f:
		r = csv.DictReader(f)
		for row in r:
			path = row['local_path']
			if not os.path.exists(path):
				continue
			try:
				if survey == 'apogee':
					obj = read_apogee_apstar(path)
				elif survey == 'galah':
					obj = read_galah_camera(path)
				elif survey == 'ges':
					obj = read_ges_uves(path)
				else:
					continue
				wave = obj['wave']; flux = obj['flux']; err = obj['err']
				if wave is None:
					# Skip if wavelength cannot be derived
					continue
				records.append({
					'survey': survey,
					'file': path,
					'n_pix': int(len(wave)),
					'wave': np.asarray(wave, dtype=np.float64),
					'flux': np.asarray(flux, dtype=np.float32),
					'err': np.asarray(err, dtype=np.float32) if err is not None else None,
				})
			except Exception:
				continue
	return records


def write_ragged_parquet(records: List[Dict[str, object]], out_path: str) -> None:
	ensure_dir(os.path.dirname(out_path))
	# Convert to pyarrow arrays with list types
	survey_arr = pa.array([r['survey'] for r in records], type=pa.string())
	file_arr = pa.array([r['file'] for r in records], type=pa.string())
	npix_arr = pa.array([r['n_pix'] for r in records], type=pa.int32())
	wave_arr = pa.array([pa.array(r['wave'], type=pa.float64) for r in records])
	flux_arr = pa.array([pa.array(r['flux'], type=pa.float32) for r in records])
	err_arr = pa.array([
		pa.array(r['err'], type=pa.float32) if r['err'] is not None else None
		for r in records
	])
	table = pa.table({
		'survey': survey_arr,
		'file': file_arr,
		'n_pix': npix_arr,
		'wave': wave_arr,
		'flux': flux_arr,
		'err': err_arr,
	})
	pq.write_table(table, out_path)


def main(argv: List[str] = None) -> None:
	p = argparse.ArgumentParser(description='Build ragged native spectra parquet')
	p.add_argument('--apogee-manifest', default='/workspace/data/apogee/manifests/apogee_manifest.csv')
	p.add_argument('--galah-manifest', default='/workspace/data/galah/manifests/galah_manifest.csv')
	p.add_argument('--ges-manifest', default='/workspace/data/ges/manifests/ges_manifest.csv')
	p.add_argument('--out', default='/workspace/data/common/processed/native_spectra.parquet')
	args = p.parse_args(argv)

	records: List[Dict[str, object]] = []
	if os.path.exists(args.apogee_manifest):
		records.extend(build_records_from_manifest(args.apogee_manifest, 'apogee'))
	if os.path.exists(args.galah_manifest):
		records.extend(build_records_from_manifest(args.galah_manifest, 'galah'))
	if os.path.exists(args.ges_manifest):
		records.extend(build_records_from_manifest(args.ges_manifest, 'ges'))
	write_ragged_parquet(records, args.out)
	print(f"Wrote {len(records)} spectra to {args.out}")


if __name__ == '__main__':
	main()