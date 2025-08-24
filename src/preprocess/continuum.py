"""
Continuum normalization utilities for stellar spectra.
"""
import numpy as np
from typing import Optional, Tuple, Union, List
from scipy import ndimage, interpolate
from scipy.optimize import curve_fit
import warnings


def sigma_clip(data: np.ndarray, 
               sigma: float = 3.0, 
               maxiters: int = 5,
               mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Iterative sigma clipping to identify outliers.
    
    Args:
        data: Input data array
        sigma: Sigma threshold for clipping
        maxiters: Maximum number of iterations
        mask: Initial mask (True = bad pixel)
        
    Returns:
        Boolean mask where True indicates outliers/bad pixels
    """
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    else:
        mask = mask.copy()
    
    # Remove non-finite values
    mask |= ~np.isfinite(data)
    
    for iteration in range(maxiters):
        if np.sum(~mask) < 10:  # Need minimum points
            break
            
        good_data = data[~mask]
        if len(good_data) == 0:
            break
            
        mean_val = np.mean(good_data)
        std_val = np.std(good_data)
        
        if std_val == 0:
            break
            
        # Identify new outliers
        new_outliers = np.abs(data - mean_val) > sigma * std_val
        
        # Check if any new outliers found
        added_outliers = new_outliers & ~mask
        if not np.any(added_outliers):
            break
            
        mask |= new_outliers
    
    return mask


def polynomial_continuum(wave: np.ndarray, 
                        flux: np.ndarray,
                        degree: int = 3,
                        sigma_clip_iters: int = 3,
                        sigma_threshold: float = 3.0,
                        mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fit polynomial continuum with sigma clipping.
    
    Args:
        wave: Wavelength array
        flux: Flux array
        degree: Polynomial degree
        sigma_clip_iters: Number of sigma clipping iterations
        sigma_threshold: Sigma threshold for clipping
        mask: Initial mask (True = bad pixel)
        
    Returns:
        Tuple of (continuum_flux, outlier_mask)
    """
    if mask is None:
        mask = np.zeros_like(flux, dtype=bool)
    else:
        mask = mask.copy()
    
    # Remove non-finite values
    mask |= ~np.isfinite(flux) | ~np.isfinite(wave)
    
    if np.sum(~mask) <= degree + 1:
        # Not enough points for fit
        return np.ones_like(flux), mask
    
    # Normalize wavelength for numerical stability
    wave_norm = (wave - wave.mean()) / wave.std()
    
    # Iterative fitting with sigma clipping
    for iteration in range(sigma_clip_iters):
        good_idx = ~mask
        
        if np.sum(good_idx) <= degree + 1:
            break
        
        # Fit polynomial
        try:
            coeffs = np.polyfit(wave_norm[good_idx], flux[good_idx], degree)
            continuum = np.polyval(coeffs, wave_norm)
        except np.linalg.LinAlgError:
            # Singular matrix, use lower degree
            if degree > 1:
                return polynomial_continuum(wave, flux, degree-1, 
                                          sigma_clip_iters, sigma_threshold, mask)
            else:
                # Fallback to mean
                continuum = np.full_like(flux, np.mean(flux[good_idx]))
                break
        
        # Sigma clipping
        residuals = flux - continuum
        outlier_mask = sigma_clip(residuals, sigma_threshold, 1, mask)
        
        # Check convergence
        if np.array_equal(outlier_mask, mask):
            break
            
        mask = outlier_mask
    
    return continuum, mask


def gaussian_smooth_continuum(wave: np.ndarray,
                             flux: np.ndarray, 
                             smooth_width: float = 50.0,
                             mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate continuum using Gaussian smoothing.
    
    Args:
        wave: Wavelength array
        flux: Flux array  
        smooth_width: Smoothing width in Angstroms
        mask: Mask for bad pixels (True = bad)
        
    Returns:
        Continuum flux array
    """
    if mask is None:
        mask = np.zeros_like(flux, dtype=bool)
    
    # Handle non-finite values
    mask |= ~np.isfinite(flux) | ~np.isfinite(wave)
    flux_work = flux.copy()
    flux_work[mask] = np.nan
    
    # Convert smoothing width to pixels
    median_spacing = np.median(np.diff(wave))
    smooth_pixels = smooth_width / median_spacing
    
    # Apply Gaussian filter, handling NaN values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Fill NaN values temporarily for filtering
        flux_filled = flux_work.copy()
        if np.any(mask):
            # Interpolate over masked regions for smoothing
            good_idx = ~mask
            if np.sum(good_idx) > 1:
                interp_func = interpolate.interp1d(wave[good_idx], flux[good_idx],
                                                 kind='linear', 
                                                 bounds_error=False,
                                                 fill_value='extrapolate')
                flux_filled[mask] = interp_func(wave[mask])
        
        # Apply Gaussian smoothing
        continuum = ndimage.gaussian_filter1d(flux_filled, smooth_pixels, 
                                            mode='nearest')
    
    return continuum


def running_percentile_continuum(wave: np.ndarray,
                                flux: np.ndarray,
                                window_width: float = 100.0,
                                percentile: float = 95.0,
                                mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate continuum using running percentile filter.
    
    Args:
        wave: Wavelength array
        flux: Flux array
        window_width: Window width in Angstroms
        percentile: Percentile to use (typically 90-99)
        mask: Mask for bad pixels (True = bad)
        
    Returns:
        Continuum flux array
    """
    if mask is None:
        mask = np.zeros_like(flux, dtype=bool)
    
    mask |= ~np.isfinite(flux) | ~np.isfinite(wave)
    
    continuum = np.full_like(flux, np.nan)
    
    # Convert window width to number of pixels
    median_spacing = np.median(np.diff(wave))
    half_window_pix = int(window_width / (2 * median_spacing))
    
    for i in range(len(flux)):
        # Define window around current pixel
        start_idx = max(0, i - half_window_pix)
        end_idx = min(len(flux), i + half_window_pix + 1)
        
        # Extract window data
        window_flux = flux[start_idx:end_idx]
        window_mask = mask[start_idx:end_idx]
        
        # Calculate percentile of good pixels in window
        good_flux = window_flux[~window_mask]
        if len(good_flux) > 0:
            continuum[i] = np.percentile(good_flux, percentile)
        else:
            # Use global median if no good pixels in window
            global_good = flux[~mask]
            if len(global_good) > 0:
                continuum[i] = np.median(global_good)
            else:
                continuum[i] = 1.0
    
    return continuum


def normalize_spectrum(wave: np.ndarray,
                      flux: np.ndarray,
                      err: Optional[np.ndarray] = None,
                      method: str = 'polynomial',
                      **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Normalize spectrum by dividing by fitted continuum.
    
    Args:
        wave: Wavelength array
        flux: Flux array
        err: Error array (optional)
        method: Continuum fitting method ('polynomial', 'gaussian', 'percentile')
        **kwargs: Additional arguments for continuum fitting method
        
    Returns:
        Tuple of (normalized_flux, normalized_error, continuum)
    """
    # Default parameters for each method
    if method == 'polynomial':
        degree = kwargs.get('degree', 3)
        sigma_clip_iters = kwargs.get('sigma_clip_iters', 3)
        sigma_threshold = kwargs.get('sigma_threshold', 3.0)
        mask = kwargs.get('mask', None)
        
        continuum, outlier_mask = polynomial_continuum(wave, flux, degree, 
                                                      sigma_clip_iters, 
                                                      sigma_threshold, mask)
    
    elif method == 'gaussian':
        smooth_width = kwargs.get('smooth_width', 50.0)
        mask = kwargs.get('mask', None)
        
        continuum = gaussian_smooth_continuum(wave, flux, smooth_width, mask)
        outlier_mask = np.zeros_like(flux, dtype=bool)
    
    elif method == 'percentile':
        window_width = kwargs.get('window_width', 100.0)
        percentile = kwargs.get('percentile', 95.0)
        mask = kwargs.get('mask', None)
        
        continuum = running_percentile_continuum(wave, flux, window_width, 
                                               percentile, mask)
        outlier_mask = np.zeros_like(flux, dtype=bool)
    
    else:
        raise ValueError(f"Unknown continuum method: {method}")
    
    # Avoid division by zero or very small values
    continuum_safe = np.where(np.abs(continuum) < 1e-10, 1.0, continuum)
    
    # Normalize flux
    normalized_flux = flux / continuum_safe
    
    # Normalize error if provided
    normalized_err = None
    if err is not None:
        normalized_err = err / np.abs(continuum_safe)
    
    return normalized_flux, normalized_err, continuum


def assess_normalization_quality(wave: np.ndarray,
                                flux: np.ndarray, 
                                continuum: np.ndarray,
                                line_regions: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
    """Assess the quality of continuum normalization.
    
    Args:
        wave: Wavelength array
        flux: Normalized flux array
        continuum: Fitted continuum array
        line_regions: List of (wave_min, wave_max) tuples for spectral lines to exclude
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Create mask for continuum regions (avoiding lines)
    continuum_mask = np.ones_like(wave, dtype=bool)
    
    if line_regions:
        for wave_min, wave_max in line_regions:
            line_mask = (wave >= wave_min) & (wave <= wave_max)
            continuum_mask &= ~line_mask
    
    # Extract continuum regions
    cont_flux = flux[continuum_mask]
    
    if len(cont_flux) > 0:
        # RMS deviation from unity in continuum regions
        metrics['continuum_rms'] = np.sqrt(np.mean((cont_flux - 1.0)**2))
        
        # Mean and std of continuum regions
        metrics['continuum_mean'] = np.mean(cont_flux)
        metrics['continuum_std'] = np.std(cont_flux)
        
        # Fraction of continuum pixels close to unity
        metrics['good_continuum_fraction'] = np.mean(np.abs(cont_flux - 1.0) < 0.1)
    
    else:
        metrics['continuum_rms'] = np.nan
        metrics['continuum_mean'] = np.nan  
        metrics['continuum_std'] = np.nan
        metrics['good_continuum_fraction'] = 0.0
    
    # Overall flux statistics
    finite_flux = flux[np.isfinite(flux)]
    if len(finite_flux) > 0:
        metrics['flux_min'] = np.min(finite_flux)
        metrics['flux_max'] = np.max(finite_flux)
        metrics['flux_median'] = np.median(finite_flux)
    else:
        metrics['flux_min'] = np.nan
        metrics['flux_max'] = np.nan
        metrics['flux_median'] = np.nan
    
    return metrics


def apply_continuum_normalization(wave: np.ndarray,
                                 flux: np.ndarray,
                                 err: Optional[np.ndarray] = None,
                                 survey: str = 'unknown',
                                 auto_method: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Apply appropriate continuum normalization for a given survey.
    
    Args:
        wave: Wavelength array
        flux: Flux array  
        err: Error array (optional)
        survey: Survey name for method selection
        auto_method: Whether to automatically select method
        
    Returns:
        Tuple of (normalized_flux, normalized_error, metadata)
    """
    metadata = {'survey': survey, 'method': 'unknown', 'quality': {}}
    
    # Survey-specific default methods
    if auto_method:
        if survey.lower() == 'apogee':
            # APOGEE benefits from polynomial fitting in log-lambda space
            method = 'polynomial'
            params = {'degree': 4, 'sigma_threshold': 2.5}
        elif survey.lower() == 'galah':
            # GALAH has good flux calibration, gentle normalization
            method = 'gaussian'
            params = {'smooth_width': 75.0}
        elif survey.lower() == 'ges':
            # GES can have variable continuum, use robust percentile
            method = 'percentile' 
            params = {'percentile': 90.0, 'window_width': 150.0}
        else:
            # Generic approach
            method = 'polynomial'
            params = {'degree': 3}
    else:
        method = 'polynomial'
        params = {'degree': 3}
    
    # Apply normalization
    try:
        normalized_flux, normalized_err, continuum = normalize_spectrum(
            wave, flux, err, method=method, **params)
        
        metadata['method'] = method
        metadata['parameters'] = params
        metadata['success'] = True
        
        # Assess quality
        quality = assess_normalization_quality(wave, normalized_flux, continuum)
        metadata['quality'] = quality
        
    except Exception as e:
        # Fallback to simple scaling if normalization fails
        warnings.warn(f"Continuum normalization failed: {e}. Using median scaling.")
        
        median_flux = np.median(flux[np.isfinite(flux)])
        if median_flux > 0:
            normalized_flux = flux / median_flux
            normalized_err = err / median_flux if err is not None else None
        else:
            normalized_flux = flux.copy()
            normalized_err = err.copy() if err is not None else None
        
        metadata['method'] = 'median_scaling'
        metadata['success'] = False
        metadata['error'] = str(e)
    
    return normalized_flux, normalized_err, metadata
