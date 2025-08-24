import os
import argparse
import csv
import time
from typing import List, Dict, Optional

import h5py
import numpy as np

from ..fetch.common import ensure_dir
from .readers import read_apogee_apstar, read_galah_camera, read_ges_uves
from .wavelength_grid import make_log_lambda_grid, resample_spectrum, apply_quality_masks
from .continuum import apply_continuum_normalization


def write_hdf5_from_manifests(apogee_manifest: str, 
                             galah_manifest: str, 
                             ges_manifest: str, 
                             out_path: str,
                             apply_continuum: bool = True,
                             apply_quality_masks_flag: bool = True,
                             resolution: float = 10000.0,
                             chunk_size: int = 1000) -> None:
	"""Build regridded HDF5 dataset with optional continuum normalization.
	
	Args:
		apogee_manifest: Path to APOGEE manifest CSV
		galah_manifest: Path to GALAH manifest CSV  
		ges_manifest: Path to GES manifest CSV
		out_path: Output HDF5 file path
		apply_continuum: Whether to apply continuum normalization
		apply_quality_masks_flag: Whether to apply quality masks
		resolution: Spectral resolution for wavelength grid
		chunk_size: HDF5 chunk size for performance
	"""
	ensure_dir(os.path.dirname(out_path))
	
	# Create common wavelength grid
	grid = make_log_lambda_grid(wave_min=3500.0, wave_max=17000.0, resolution=resolution)
	print(f"Created wavelength grid: {len(grid)} points from {grid[0]:.1f} to {grid[-1]:.1f} Å")
	
	start_time = time.time()
	processed_count = 0
	failed_count = 0
	
	with h5py.File(out_path, 'w') as h5:
		# Create main groups
		spectra_grp = h5.create_group('spectra')
		metadata_grp = h5.create_group('metadata')
		
		# Store wavelength grid
		spectra_grp.create_dataset('wavelength', data=grid, dtype='f8')
		
		# Create extensible datasets
		flux_ds = spectra_grp.create_dataset('flux', 
											shape=(0, grid.size), 
											maxshape=(None, grid.size), 
											dtype='f4', 
											chunks=(chunk_size, grid.size),
											compression='gzip', 
											compression_opts=6)
		
		err_ds = spectra_grp.create_dataset('error',
										   shape=(0, grid.size),
										   maxshape=(None, grid.size), 
										   dtype='f4',
										   chunks=(chunk_size, grid.size),
										   compression='gzip',
										   compression_opts=6)
		
		# Metadata datasets
		survey_ds = metadata_grp.create_dataset('survey', 
											   shape=(0,), 
											   maxshape=(None,), 
											   dtype=h5py.string_dtype())
		
		file_ds = metadata_grp.create_dataset('source_file',
											 shape=(0,),
											 maxshape=(None,),
											 dtype=h5py.string_dtype())
		
		snr_ds = metadata_grp.create_dataset('snr_median',
											shape=(0,),
											maxshape=(None,),
											dtype='f4')
		
		quality_ds = metadata_grp.create_dataset('quality_score',
											   shape=(0,),
											   maxshape=(None,),
											   dtype='f4')
		
		continuum_method_ds = metadata_grp.create_dataset('continuum_method',
														shape=(0,),
														maxshape=(None,),
														dtype=h5py.string_dtype())

		def append_spectrum(survey: str, 
						  file_path: str, 
						  wave: np.ndarray, 
						  flux: np.ndarray,
						  err: Optional[np.ndarray] = None) -> bool:
			"""Process and append a single spectrum."""
			try:
				# Apply quality masks if requested
				if apply_quality_masks_flag:
					wave_clean, flux_clean, err_clean = apply_quality_masks(
						wave, flux, err, survey=survey)
				else:
					wave_clean, flux_clean, err_clean = wave, flux, err
				
				# Apply continuum normalization if requested
				if apply_continuum:
					flux_norm, err_norm, cont_metadata = apply_continuum_normalization(
						wave_clean, flux_clean, err_clean, survey=survey)
					continuum_method = cont_metadata.get('method', 'unknown')
					quality_score = cont_metadata.get('quality', {}).get('good_continuum_fraction', 0.0)
				else:
					flux_norm, err_norm = flux_clean, err_clean
					continuum_method = 'none'
					quality_score = 1.0
				
				# Resample to common grid
				flux_regrid, err_regrid = resample_spectrum(wave_clean, flux_norm, grid, 
														  err_norm, method='linear')
				
				# Calculate SNR
				if err_regrid is not None:
					finite_mask = np.isfinite(flux_regrid) & np.isfinite(err_regrid) & (err_regrid > 0)
					if np.any(finite_mask):
						snr_values = np.abs(flux_regrid[finite_mask]) / err_regrid[finite_mask]
						snr_median = np.median(snr_values)
					else:
						snr_median = 0.0
				else:
					snr_median = np.nan
				
				# Fill error array if not provided
				if err_regrid is None:
					err_regrid = np.full_like(flux_regrid, 0.1)  # Assume 10% error
				
				# Append to datasets
				n = flux_ds.shape[0]
				
				# Resize datasets
				flux_ds.resize((n + 1, grid.size))
				err_ds.resize((n + 1, grid.size))
				survey_ds.resize((n + 1,))
				file_ds.resize((n + 1,))
				snr_ds.resize((n + 1,))
				quality_ds.resize((n + 1,))
				continuum_method_ds.resize((n + 1,))
				
				# Store data
				flux_ds[n, :] = flux_regrid.astype(np.float32)
				err_ds[n, :] = err_regrid.astype(np.float32)
				survey_ds[n] = survey
				file_ds[n] = file_path
				snr_ds[n] = snr_median
				quality_ds[n] = quality_score
				continuum_method_ds[n] = continuum_method
				
				return True
				
			except Exception as e:
				print(f"Failed to process {file_path}: {e}")
				return False

		def process_manifest(manifest_csv: str, survey: str) -> None:
			"""Process all spectra in a manifest file."""
			nonlocal processed_count, failed_count
			
			if not manifest_csv or not os.path.exists(manifest_csv):
				print(f"Manifest not found: {manifest_csv}")
				return
			
			print(f"Processing {survey.upper()} manifest: {manifest_csv}")
			
			with open(manifest_csv, 'r') as f:
				reader = csv.DictReader(f)
				rows = list(reader)
			
			for i, row in enumerate(rows):
				path = row['local_path']
				if not os.path.exists(path):
					continue
				
				try:
					# Read spectrum based on survey
					if survey == 'apogee':
						obj = read_apogee_apstar(path)
					elif survey == 'galah':
						obj = read_galah_camera(path)
					elif survey == 'ges':
						obj = read_ges_uves(path)
					else:
						continue
					
					# Check for valid wavelength array
					if obj['wave'] is None or len(obj['wave']) < 10:
						failed_count += 1
						continue
					
					# Process spectrum
					success = append_spectrum(survey, path, obj['wave'], obj['flux'], obj['err'])
					
					if success:
						processed_count += 1
					else:
						failed_count += 1
					
					# Progress reporting
					if (i + 1) % 100 == 0:
						elapsed = time.time() - start_time
						print(f"  Processed {i + 1}/{len(rows)} {survey} spectra "
							  f"({processed_count} success, {failed_count} failed) "
							  f"in {elapsed:.1f}s")
				
				except Exception as e:
					print(f"Error reading {path}: {e}")
					failed_count += 1
					continue

		# Process each survey
		process_manifest(apogee_manifest, 'apogee')
		process_manifest(galah_manifest, 'galah')
		process_manifest(ges_manifest, 'ges')
		
		# Store processing metadata
		metadata_grp.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
		metadata_grp.attrs['wavelength_min'] = grid[0]
		metadata_grp.attrs['wavelength_max'] = grid[-1]
		metadata_grp.attrs['wavelength_points'] = len(grid)
		metadata_grp.attrs['resolution'] = resolution
		metadata_grp.attrs['continuum_normalized'] = apply_continuum
		metadata_grp.attrs['quality_masked'] = apply_quality_masks_flag
		metadata_grp.attrs['total_spectra'] = processed_count
		metadata_grp.attrs['failed_spectra'] = failed_count

	elapsed = time.time() - start_time
	print(f"\nCompleted HDF5 dataset creation:")
	print(f"  Output file: {out_path}")
	print(f"  Total spectra: {processed_count}")
	print(f"  Failed spectra: {failed_count}")
	print(f"  Processing time: {elapsed:.1f}s")
	print(f"  Average time per spectrum: {elapsed/max(1, processed_count):.3f}s")


def main(argv: List[str] = None) -> None:
	p = argparse.ArgumentParser(description='Build regridded HDF5 spectra with continuum normalization')
	p.add_argument('--apogee-manifest', default=os.path.join('data', 'apogee', 'manifests', 'apogee_manifest.csv'))
	p.add_argument('--galah-manifest', default=os.path.join('data', 'galah', 'manifests', 'galah_manifest.csv'))
	p.add_argument('--ges-manifest', default=os.path.join('data', 'ges', 'manifests', 'ges_manifest.csv'))
	p.add_argument('--out', default=os.path.join('data', 'common', 'processed', 'regridded_spectra.h5'))
	p.add_argument('--no-continuum', action='store_true', help='Skip continuum normalization')
	p.add_argument('--no-quality-masks', action='store_true', help='Skip quality masking')
	p.add_argument('--resolution', type=float, default=10000.0, help='Spectral resolution for wavelength grid')
	p.add_argument('--chunk-size', type=int, default=1000, help='HDF5 chunk size')
	args = p.parse_args(argv)
	
	write_hdf5_from_manifests(
		args.apogee_manifest, 
		args.galah_manifest, 
		args.ges_manifest, 
		args.out,
		apply_continuum=not args.no_continuum,
		apply_quality_masks_flag=not args.no_quality_masks,
		resolution=args.resolution,
		chunk_size=args.chunk_size
	)


if __name__ == '__main__':
	main()