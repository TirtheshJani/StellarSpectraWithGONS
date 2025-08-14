import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
from astropy.io import fits


def read_apogee_apstar(path: str) -> Dict[str, Any]:
	with fits.open(path, memmap=False) as hdul:
		h = hdul[0].header
		# apStar combined spectrum often in HDU 1 as FLUX, with LAMBDA in header via WCS
		data = hdul[1].data if len(hdul) > 1 else hdul[0].data
		n = data.shape[-1]
		crval1 = h.get('CRVAL1'); cdelt1 = h.get('CDELT1') or h.get('CD1_1'); crpix1 = h.get('CRPIX1', 1.0)
		if crval1 is not None and cdelt1 is not None:
			pix = np.arange(n)
			wave = (crval1 + (pix + 1 - crpix1) * cdelt1).astype(np.float64)
		else:
			# Some apStar files carry wavelength in ext 4 named WAVE
			wave = None
		flux = np.array(data, dtype=np.float32)
		err = None
		for ext in range(1, len(hdul)):
			if 'ERR' in (hdul[ext].name or '').upper():
				err = np.array(hdul[ext].data, dtype=np.float32)
				break
		return {'wave': wave, 'flux': flux, 'err': err, 'meta': dict(h)}


def read_galah_camera(path: str) -> Dict[str, Any]:
	with fits.open(path, memmap=False) as hdul:
		h = hdul[0].header
		# GALAH typically stores wavelength array explicitly in extension 1 column or as a vector HDU
		# Try vector in ext 1
		wave = None
		flux = None
		for i in range(1, len(hdul)):
			hdu = hdul[i]
			if getattr(hdu, 'data', None) is None:
				continue
			data = hdu.data
			if hasattr(data, 'dtype') and data.dtype.names:
				names = [n.lower() for n in data.dtype.names]
				if 'wavelength' in names and 'flux' in names:
					wave = np.array(data['wavelength'], dtype=np.float64)
					flux = np.array(data['flux'], dtype=np.float32)
					break
			elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
				wave = np.array(data[:, 0], dtype=np.float64)
				flux = np.array(data[:, 1], dtype=np.float32)
				break
		err = None
		return {'wave': wave, 'flux': flux, 'err': err, 'meta': dict(h)}


def read_ges_uves(path: str) -> Dict[str, Any]:
	with fits.open(path, memmap=False) as hdul:
		h = hdul[0].header
		# Many ESO Phase3 UVES spectra provide WCS in primary and flux in ext 1
		data_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
		flux = np.array(data_hdu.data, dtype=np.float32)
		hd = data_hdu.header
		crval1 = hd.get('CRVAL1') or hd.get('HIERARCH ESO QC WAVEL_MIN')
		cdelt1 = hd.get('CDELT1') or hd.get('CD1_1')
		crpix1 = hd.get('CRPIX1', 1.0)
		n = flux.shape[-1]
		wave = None
		if crval1 is not None and cdelt1 is not None:
			pix = np.arange(n)
			wave = (float(crval1) + (pix + 1 - float(crpix1)) * float(cdelt1)).astype(np.float64)
		return {'wave': wave, 'flux': flux, 'err': None, 'meta': dict(h)}