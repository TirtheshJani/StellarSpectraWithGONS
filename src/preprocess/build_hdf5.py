import os
import argparse
import csv
from typing import List

import h5py
import numpy as np

from ..fetch.common import ensure_dir
from .readers import read_apogee_apstar, read_galah_camera, read_ges_uves


def make_log_lambda_grid(wmin: float = 3500.0, wmax: float = 17000.0, dv_kms: float = 10.0) -> np.ndarray:
	c_kms = 299792.458
	dln = dv_kms / c_kms
	n = int(np.floor(np.log(wmax / wmin) / dln)) + 1
	return wmin * np.exp(np.arange(n) * dln)


def resample_to_grid(wave: np.ndarray, flux: np.ndarray, grid: np.ndarray) -> np.ndarray:
	# Simple nearest-neighbor for baseline; can replace with linear
	idx = np.searchsorted(wave, grid)
	idx = np.clip(idx, 1, len(wave) - 1)
	left = idx - 1
	right = idx
	wl = wave[left]; wr = wave[right]
	fl = flux[left]; fr = flux[right]
	w = (grid - wl) / np.clip(wr - wl, 1e-6, None)
	return (fl * (1 - w) + fr * w).astype(np.float32)


def write_hdf5_from_manifests(apogee_manifest: str, galah_manifest: str, ges_manifest: str, out_path: str) -> None:
	ensure_dir(os.path.dirname(out_path))
	grid = make_log_lambda_grid()
	with h5py.File(out_path, 'w') as h5:
		grp = h5.create_group('spectra')
		grp.create_dataset('lambda', data=grid, dtype='f8')
		flux_ds = grp.create_dataset('flux', shape=(0, grid.size), maxshape=(None, grid.size), dtype='f4', chunks=True)
		survey_ds = grp.create_dataset('survey', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
		file_ds = grp.create_dataset('file', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

		def append_row(survey: str, file_path: str, wave: np.ndarray, flux: np.ndarray) -> None:
			res = resample_to_grid(wave, flux, grid)
			n = flux_ds.shape[0]
			flux_ds.resize((n + 1, grid.size))
			flux_ds[n, :] = res
			survey_ds.resize((n + 1,))
			survey_ds[n] = survey
			file_ds.resize((n + 1,))
			file_ds[n] = file_path

		def process_manifest(manifest_csv: str, survey: str) -> None:
			if not manifest_csv or not os.path.exists(manifest_csv):
				return
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
						else:
							obj = read_ges_uves(path)
						if obj['wave'] is None:
							continue
						append_row(survey, path, obj['wave'], obj['flux'])
					except Exception:
						continue

		process_manifest(apogee_manifest, 'apogee')
		process_manifest(galah_manifest, 'galah')
		process_manifest(ges_manifest, 'ges')

	print(f"Wrote HDF5 baseline to {out_path}")


def main(argv: List[str] = None) -> None:
	p = argparse.ArgumentParser(description='Build baseline regridded HDF5 spectra')
	p.add_argument('--apogee-manifest', default='/workspace/data/apogee/manifests/apogee_manifest.csv')
	p.add_argument('--galah-manifest', default='/workspace/data/galah/manifests/galah_manifest.csv')
	p.add_argument('--ges-manifest', default='/workspace/data/ges/manifests/ges_manifest.csv')
	p.add_argument('--out', default='/workspace/data/common/processed/baseline_spectra.h5')
	args = p.parse_args(argv)
	write_hdf5_from_manifests(args.apogee_manifest, args.galah_manifest, args.ges_manifest, args.out)


if __name__ == '__main__':
	main()