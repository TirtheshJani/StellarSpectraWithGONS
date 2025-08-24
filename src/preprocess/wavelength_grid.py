"""
Wavelength grid standardization and resampling utilities.
"""
import numpy as np
from typing import Optional, Tuple, Dict, List
from scipy import interpolate
import warnings


def make_log_lambda_grid(wave_min: float = 3500.0, 
                        wave_max: float = 17000.0, 
                        resolution: float = 10000.0) -> np.ndarray:
    """Create a logarithmically-spaced wavelength grid.
    
    Args:
        wave_min: Minimum wavelength in Angstroms
        wave_max: Maximum wavelength in Angstroms  
        resolution: Spectral resolution R = λ/Δλ
        
    Returns:
        Wavelength array in Angstroms
    """
    # Convert resolution to velocity step
    c_kms = 299792.458  # Speed of light in km/s
    dv_kms = c_kms / resolution  # Velocity step in km/s
    
    # Convert to natural log step
    dln = dv_kms / c_kms
    
    # Calculate number of points
    n_points = int(np.floor(np.log(wave_max / wave_min) / dln)) + 1
    
    # Create log-lambda grid
    wave_grid = wave_min * np.exp(np.arange(n_points) * dln)
    
    # Ensure we don't exceed wave_max
    wave_grid = wave_grid[wave_grid <= wave_max]
    
    return wave_grid


def resample_spectrum(wave_in: np.ndarray, 
                     flux_in: np.ndarray, 
                     wave_out: np.ndarray,
                     err_in: Optional[np.ndarray] = None,
                     method: str = 'linear',
                     flux_conserve: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resample spectrum to new wavelength grid with flux conservation.
    
    Args:
        wave_in: Input wavelength array
        flux_in: Input flux array
        wave_out: Output wavelength grid
        err_in: Input error array (optional)
        method: Interpolation method ('linear', 'cubic')
        flux_conserve: Whether to conserve total flux
        
    Returns:
        Tuple of (resampled_flux, resampled_error)
    """
    # Remove NaN and infinite values
    mask = np.isfinite(wave_in) & np.isfinite(flux_in)
    if err_in is not None:
        mask &= np.isfinite(err_in)
    
    wave_clean = wave_in[mask]
    flux_clean = flux_in[mask]
    err_clean = err_in[mask] if err_in is not None else None
    
    if len(wave_clean) < 2:
        # Not enough data for interpolation
        flux_out = np.full_like(wave_out, np.nan, dtype=np.float32)
        err_out = np.full_like(wave_out, np.nan, dtype=np.float32) if err_in is not None else None
        return flux_out, err_out
    
    # Sort by wavelength if needed
    if not np.all(np.diff(wave_clean) >= 0):
        sort_idx = np.argsort(wave_clean)
        wave_clean = wave_clean[sort_idx]
        flux_clean = flux_clean[sort_idx]
        if err_clean is not None:
            err_clean = err_clean[sort_idx]
    
    # Only interpolate within the input wavelength range
    valid_range = (wave_out >= wave_clean[0]) & (wave_out <= wave_clean[-1])
    
    # Initialize output arrays
    flux_out = np.full_like(wave_out, np.nan, dtype=np.float32)
    err_out = np.full_like(wave_out, np.nan, dtype=np.float32) if err_in is not None else None
    
    if not np.any(valid_range):
        return flux_out, err_out
    
    # Interpolate flux
    if method == 'linear':
        interp_func = interpolate.interp1d(wave_clean, flux_clean, 
                                         kind='linear', 
                                         bounds_error=False, 
                                         fill_value=np.nan)
        flux_out[valid_range] = interp_func(wave_out[valid_range])
        
        # Interpolate errors if provided
        if err_clean is not None:
            err_interp_func = interpolate.interp1d(wave_clean, err_clean, 
                                                 kind='linear',
                                                 bounds_error=False, 
                                                 fill_value=np.nan)
            err_out[valid_range] = err_interp_func(wave_out[valid_range])
    
    elif method == 'cubic':
        if len(wave_clean) >= 4:  # Need at least 4 points for cubic
            try:
                interp_func = interpolate.interp1d(wave_clean, flux_clean, 
                                                 kind='cubic',
                                                 bounds_error=False, 
                                                 fill_value=np.nan)
                flux_out[valid_range] = interp_func(wave_out[valid_range])
                
                if err_clean is not None:
                    err_interp_func = interpolate.interp1d(wave_clean, err_clean, 
                                                         kind='cubic',
                                                         bounds_error=False, 
                                                         fill_value=np.nan)
                    err_out[valid_range] = err_interp_func(wave_out[valid_range])
            except Exception:
                # Fall back to linear if cubic fails
                return resample_spectrum(wave_in, flux_in, wave_out, err_in, 
                                       method='linear', flux_conserve=flux_conserve)
        else:
            # Fall back to linear for insufficient points
            return resample_spectrum(wave_in, flux_in, wave_out, err_in, 
                                   method='linear', flux_conserve=flux_conserve)
    
    # Flux conservation correction if requested
    if flux_conserve and np.any(np.isfinite(flux_out)):
        # Calculate flux ratios in overlapping regions
        overlap_in = (wave_clean >= wave_out[0]) & (wave_clean <= wave_out[-1])
        overlap_out = np.isfinite(flux_out)
        
        if np.any(overlap_in) and np.any(overlap_out):
            # Integrate flux in overlapping regions
            flux_in_int = np.trapz(flux_clean[overlap_in], wave_clean[overlap_in])
            flux_out_int = np.trapz(flux_out[overlap_out], wave_out[overlap_out])
            
            if flux_out_int > 0 and np.isfinite(flux_in_int):
                correction = flux_in_int / flux_out_int
                flux_out[overlap_out] *= correction
    
    return flux_out.astype(np.float32), err_out.astype(np.float32) if err_out is not None else None


def create_detector_masks(wave_grid: np.ndarray, survey: str) -> Dict[str, np.ndarray]:
    """Create masks for detector gaps and known problematic regions.
    
    Args:
        wave_grid: Wavelength grid
        survey: Survey name ('apogee', 'galah', 'ges')
        
    Returns:
        Dictionary of boolean masks
    """
    masks = {}
    
    if survey.lower() == 'apogee':
        # APOGEE detector gaps (approximate)
        gap1 = (wave_grid >= 15890) & (wave_grid <= 15960)  # Blue-green gap
        gap2 = (wave_grid >= 16430) & (wave_grid <= 16500)  # Green-red gap
        masks['detector_gaps'] = gap1 | gap2
        
        # Known problematic regions
        masks['bad_regions'] = np.zeros_like(wave_grid, dtype=bool)
        
    elif survey.lower() == 'galah':
        # GALAH camera boundaries (approximate)
        blue_red = wave_grid <= 4900  # Blue camera
        green_start = (wave_grid >= 5650) & (wave_grid <= 5700)  # Green camera start
        red_start = (wave_grid >= 6480) & (wave_grid <= 6530)    # Red camera start  
        ir_start = wave_grid >= 7680    # IR camera start
        
        # No significant gaps in GALAH
        masks['detector_gaps'] = np.zeros_like(wave_grid, dtype=bool)
        
        # Camera boundaries as potentially problematic
        masks['camera_edges'] = green_start | red_start
        
    elif survey.lower() == 'ges':
        # GES UVES has gaps between blue and red arms
        blue_end = wave_grid <= 3800   # Blue arm end
        red_start = wave_grid >= 4800  # Red arm start
        gap = (wave_grid > 3800) & (wave_grid < 4800)
        
        masks['detector_gaps'] = gap
        masks['bad_regions'] = blue_end  # Very blue region often noisy
    
    else:
        # Generic masks
        masks['detector_gaps'] = np.zeros_like(wave_grid, dtype=bool)
        masks['bad_regions'] = np.zeros_like(wave_grid, dtype=bool)
    
    return masks


def create_telluric_mask(wave_grid: np.ndarray, 
                        strength_threshold: float = 0.1) -> np.ndarray:
    """Create mask for telluric absorption regions.
    
    Args:
        wave_grid: Wavelength grid in Angstroms
        strength_threshold: Minimum strength to mask (0=weak, 1=strong)
        
    Returns:
        Boolean mask (True = telluric region)
    """
    # Major telluric absorption bands
    telluric_bands = [
        # Water vapor bands
        (6860, 6890),   # 686 nm H2O
        (7160, 7340),   # 720 nm H2O (strong)
        (8120, 8250),   # 820 nm H2O  
        (9300, 9650),   # 930 nm H2O (very strong)
        (11300, 11800), # 1.13 μm H2O
        (13500, 14500), # 1.4 μm H2O (very strong)
        
        # Oxygen bands
        (6270, 6330),   # 627 nm O2
        (6860, 6890),   # 686 nm O2  
        (7590, 7680),   # 760 nm O2 (A-band, strong)
        (12700, 13000), # 1.27 μm O2 (very strong)
        
        # Carbon dioxide
        (15950, 16050), # 1.6 μm CO2
        (20100, 20600), # 2.0 μm CO2
    ]
    
    mask = np.zeros_like(wave_grid, dtype=bool)
    
    for wave_min, wave_max in telluric_bands:
        # Apply strength threshold (stronger bands have priority)
        if strength_threshold <= 0.3:  # Include weak bands
            band_mask = (wave_grid >= wave_min) & (wave_grid <= wave_max)
        elif strength_threshold <= 0.6:  # Only medium and strong bands
            if wave_max - wave_min > 100:  # Broader bands are usually stronger
                band_mask = (wave_grid >= wave_min) & (wave_grid <= wave_max)
            else:
                continue
        else:  # Only very strong bands
            if wave_max - wave_min > 200:
                band_mask = (wave_grid >= wave_min) & (wave_grid <= wave_max)
            else:
                continue
        
        mask |= band_mask
    
    return mask


def apply_quality_masks(wave: np.ndarray, 
                       flux: np.ndarray,
                       err: Optional[np.ndarray] = None,
                       survey: str = 'unknown',
                       mask_tellurics: bool = True,
                       mask_detectors: bool = True,
                       snr_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply quality masks to spectrum.
    
    Args:
        wave: Wavelength array
        flux: Flux array
        err: Error array (optional)
        survey: Survey name for detector-specific masks
        mask_tellurics: Whether to mask telluric regions
        mask_detectors: Whether to mask detector gaps
        snr_threshold: SNR threshold for pixel rejection
        
    Returns:
        Tuple of (wavelength, flux, error) with bad pixels set to NaN
    """
    # Start with input arrays
    wave_out = wave.copy()
    flux_out = flux.copy()
    err_out = err.copy() if err is not None else None
    
    # Create composite mask
    bad_mask = np.zeros_like(wave, dtype=bool)
    
    # Mask non-finite values
    bad_mask |= ~np.isfinite(flux)
    if err is not None:
        bad_mask |= ~np.isfinite(err)
    
    # SNR-based masking
    if err is not None and snr_threshold > 0:
        snr = np.abs(flux) / np.maximum(err, 1e-10)
        bad_mask |= snr < snr_threshold
    
    # Survey-specific detector masks
    if mask_detectors:
        detector_masks = create_detector_masks(wave, survey)
        for mask_name, mask_array in detector_masks.items():
            bad_mask |= mask_array
    
    # Telluric masks
    if mask_tellurics:
        telluric_mask = create_telluric_mask(wave, strength_threshold=0.3)
        bad_mask |= telluric_mask
    
    # Apply masks by setting bad pixels to NaN
    flux_out[bad_mask] = np.nan
    if err_out is not None:
        err_out[bad_mask] = np.nan
    
    return wave_out, flux_out, err_out


def validate_wavelength_grid(wave: np.ndarray) -> bool:
    """Validate that wavelength array is monotonic and reasonable.
    
    Args:
        wave: Wavelength array
        
    Returns:
        True if valid, False otherwise
    """
    if len(wave) < 2:
        return False
    
    # Check for monotonic increasing
    if not np.all(np.diff(wave) > 0):
        return False
    
    # Check for reasonable wavelength range (optical/IR)
    if wave[0] < 1000 or wave[-1] > 50000:
        return False
    
    # Check for reasonable spacing
    median_spacing = np.median(np.diff(wave))
    if median_spacing <= 0 or median_spacing > 100:
        return False
    
    return True
