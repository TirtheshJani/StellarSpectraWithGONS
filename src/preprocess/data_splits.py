"""
Data splitting utilities for train/validation/test sets.
"""
import os
import h5py
import numpy as np
from typing import Tuple, Optional, Dict, List
import argparse


def create_data_splits(h5_path: str,
                      train_frac: float = 0.7,
                      val_frac: float = 0.15,
                      test_frac: float = 0.15,
                      random_seed: int = 42,
                      stratify_by: Optional[str] = None,
                      min_snr: Optional[float] = None,
                      quality_threshold: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Create train/validation/test splits from HDF5 dataset.
    
    Args:
        h5_path: Path to HDF5 dataset file
        train_frac: Fraction for training set
        val_frac: Fraction for validation set  
        test_frac: Fraction for test set
        random_seed: Random seed for reproducibility
        stratify_by: Column to stratify by ('survey', 'snr_median', etc.)
        min_snr: Minimum SNR threshold for inclusion
        quality_threshold: Minimum quality score threshold
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing indices
    """
    # Validate fractions
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")
    
    np.random.seed(random_seed)
    
    with h5py.File(h5_path, 'r') as h5:
        n_spectra = h5['spectra/flux'].shape[0]
        
        # Create quality filter
        quality_mask = np.ones(n_spectra, dtype=bool)
        
        # SNR filtering
        if min_snr is not None and 'metadata/snr_median' in h5:
            snr_values = h5['metadata/snr_median'][:]
            quality_mask &= (snr_values >= min_snr) & np.isfinite(snr_values)
        
        # Quality score filtering
        if quality_threshold is not None and 'metadata/quality_score' in h5:
            quality_scores = h5['metadata/quality_score'][:]
            quality_mask &= (quality_scores >= quality_threshold) & np.isfinite(quality_scores)
        
        # Get valid indices
        valid_indices = np.where(quality_mask)[0]
        n_valid = len(valid_indices)
        
        print(f"Total spectra: {n_spectra}")
        print(f"Valid spectra (after filtering): {n_valid}")
        
        if n_valid == 0:
            raise ValueError("No valid spectra after filtering")
        
        # Stratified splitting if requested
        if stratify_by is not None and f'metadata/{stratify_by}' in h5:
            stratify_values = h5[f'metadata/{stratify_by}'][valid_indices]
            
            if stratify_by == 'survey':
                # Handle string data
                unique_surveys = np.unique(stratify_values)
                splits = {'train': [], 'val': [], 'test': []}
                
                for survey in unique_surveys:
                    survey_mask = stratify_values == survey
                    survey_indices = valid_indices[survey_mask]
                    n_survey = len(survey_indices)
                    
                    # Shuffle within survey
                    np.random.shuffle(survey_indices)
                    
                    # Calculate splits
                    n_train = int(n_survey * train_frac)
                    n_val = int(n_survey * val_frac)
                    
                    splits['train'].extend(survey_indices[:n_train])
                    splits['val'].extend(survey_indices[n_train:n_train + n_val])
                    splits['test'].extend(survey_indices[n_train + n_val:])
                    
                    print(f"  {survey}: {len(survey_indices[:n_train])}/{len(survey_indices[n_train:n_train + n_val])}/{len(survey_indices[n_train + n_val:])} (train/val/test)")
                
                # Convert to arrays and shuffle
                for split in splits:
                    splits[split] = np.array(splits[split])
                    np.random.shuffle(splits[split])
            
            elif stratify_by == 'snr_median':
                # Stratify by SNR quartiles
                snr_values = stratify_values
                quartiles = np.percentile(snr_values[np.isfinite(snr_values)], [25, 50, 75])
                
                splits = {'train': [], 'val': [], 'test': []}
                
                for i in range(4):  # 4 quartiles
                    if i == 0:
                        quartile_mask = snr_values <= quartiles[0]
                    elif i == 3:
                        quartile_mask = snr_values > quartiles[2]
                    else:
                        quartile_mask = (snr_values > quartiles[i-1]) & (snr_values <= quartiles[i])
                    
                    quartile_indices = valid_indices[quartile_mask]
                    n_quartile = len(quartile_indices)
                    
                    if n_quartile == 0:
                        continue
                    
                    np.random.shuffle(quartile_indices)
                    
                    n_train = int(n_quartile * train_frac)
                    n_val = int(n_quartile * val_frac)
                    
                    splits['train'].extend(quartile_indices[:n_train])
                    splits['val'].extend(quartile_indices[n_train:n_train + n_val])
                    splits['test'].extend(quartile_indices[n_train + n_val:])
                
                # Convert to arrays
                for split in splits:
                    splits[split] = np.array(splits[split])
                    np.random.shuffle(splits[split])
        
        else:
            # Simple random splitting
            shuffled_indices = valid_indices.copy()
            np.random.shuffle(shuffled_indices)
            
            n_train = int(n_valid * train_frac)
            n_val = int(n_valid * val_frac)
            
            splits = {
                'train': shuffled_indices[:n_train],
                'val': shuffled_indices[n_train:n_train + n_val],
                'test': shuffled_indices[n_train + n_val:]
            }
    
    print(f"Final splits: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} (train/val/test)")
    
    return splits


def save_splits_to_hdf5(h5_path: str, splits: Dict[str, np.ndarray], overwrite: bool = False) -> None:
    """Save data splits to the HDF5 file.
    
    Args:
        h5_path: Path to HDF5 dataset file
        splits: Dictionary with split indices
        overwrite: Whether to overwrite existing splits
    """
    with h5py.File(h5_path, 'a') as h5:
        # Create splits group
        if 'splits' in h5:
            if overwrite:
                del h5['splits']
            else:
                raise ValueError("Splits already exist. Use overwrite=True to replace.")
        
        splits_grp = h5.create_group('splits')
        
        for split_name, indices in splits.items():
            splits_grp.create_dataset(split_name, data=indices, dtype='i8')
        
        # Store metadata
        splits_grp.attrs['train_size'] = len(splits['train'])
        splits_grp.attrs['val_size'] = len(splits['val'])
        splits_grp.attrs['test_size'] = len(splits['test'])
        splits_grp.attrs['total_size'] = sum(len(indices) for indices in splits.values())
        
        print(f"Saved splits to {h5_path}:/splits/")


def load_splits_from_hdf5(h5_path: str) -> Dict[str, np.ndarray]:
    """Load data splits from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 dataset file
        
    Returns:
        Dictionary with split indices
    """
    splits = {}
    
    with h5py.File(h5_path, 'r') as h5:
        if 'splits' not in h5:
            raise ValueError("No splits found in HDF5 file")
        
        splits_grp = h5['splits']
        
        for split_name in ['train', 'val', 'test']:
            if split_name in splits_grp:
                splits[split_name] = splits_grp[split_name][:]
    
    return splits


def analyze_dataset_composition(h5_path: str, splits: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Analyze the composition of the dataset and splits.
    
    Args:
        h5_path: Path to HDF5 dataset file
        splits: Optional pre-computed splits to analyze
    """
    print(f"\n=== Dataset Analysis: {h5_path} ===")
    
    with h5py.File(h5_path, 'r') as h5:
        n_spectra = h5['spectra/flux'].shape[0]
        n_wavelengths = h5['spectra/flux'].shape[1]
        
        print(f"Total spectra: {n_spectra:,}")
        print(f"Wavelength points: {n_wavelengths:,}")
        
        if 'spectra/wavelength' in h5:
            wave = h5['spectra/wavelength'][:]
            print(f"Wavelength range: {wave[0]:.1f} - {wave[-1]:.1f} Å")
        
        # Survey composition
        if 'metadata/survey' in h5:
            surveys = h5['metadata/survey'][:]
            unique_surveys, counts = np.unique(surveys, return_counts=True)
            print(f"\nSurvey composition:")
            for survey, count in zip(unique_surveys, counts):
                print(f"  {survey.decode() if isinstance(survey, bytes) else survey}: {count:,} ({count/n_spectra:.1%})")
        
        # SNR statistics  
        if 'metadata/snr_median' in h5:
            snr_values = h5['metadata/snr_median'][:]
            finite_snr = snr_values[np.isfinite(snr_values)]
            if len(finite_snr) > 0:
                print(f"\nSNR statistics:")
                print(f"  Mean: {np.mean(finite_snr):.1f}")
                print(f"  Median: {np.median(finite_snr):.1f}")
                print(f"  Range: {np.min(finite_snr):.1f} - {np.max(finite_snr):.1f}")
                print(f"  Percentiles (10/90): {np.percentile(finite_snr, [10, 90])}")
        
        # Quality statistics
        if 'metadata/quality_score' in h5:
            quality_scores = h5['metadata/quality_score'][:]
            finite_quality = quality_scores[np.isfinite(quality_scores)]
            if len(finite_quality) > 0:
                print(f"\nQuality score statistics:")
                print(f"  Mean: {np.mean(finite_quality):.3f}")
                print(f"  Median: {np.median(finite_quality):.3f}")
                print(f"  Range: {np.min(finite_quality):.3f} - {np.max(finite_quality):.3f}")
        
        # Continuum methods
        if 'metadata/continuum_method' in h5:
            methods = h5['metadata/continuum_method'][:]
            unique_methods, counts = np.unique(methods, return_counts=True)
            print(f"\nContinuum normalization methods:")
            for method, count in zip(unique_methods, counts):
                method_str = method.decode() if isinstance(method, bytes) else method
                print(f"  {method_str}: {count:,} ({count/n_spectra:.1%})")
        
        # Analyze splits if provided
        if splits is not None:
            print(f"\n=== Split Analysis ===")
            for split_name, indices in splits.items():
                print(f"\n{split_name.upper()} set ({len(indices):,} spectra):")
                
                # Survey composition in split
                if 'metadata/survey' in h5:
                    split_surveys = surveys[indices]
                    unique_surveys, counts = np.unique(split_surveys, return_counts=True)
                    for survey, count in zip(unique_surveys, counts):
                        survey_str = survey.decode() if isinstance(survey, bytes) else survey
                        print(f"  {survey_str}: {count:,} ({count/len(indices):.1%})")


def create_processed_directory() -> None:
    """Create the processed data directory if it doesn't exist."""
    processed_dir = os.path.join('data', 'common', 'processed')
    os.makedirs(processed_dir, exist_ok=True)


def main(argv: List[str] = None) -> None:
    """Main function for creating data splits."""
    p = argparse.ArgumentParser(description='Create train/validation/test splits for HDF5 dataset')
    p.add_argument('h5_file', help='Path to HDF5 dataset file')
    p.add_argument('--train-frac', type=float, default=0.7, help='Training set fraction')
    p.add_argument('--val-frac', type=float, default=0.15, help='Validation set fraction')
    p.add_argument('--test-frac', type=float, default=0.15, help='Test set fraction')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--stratify-by', choices=['survey', 'snr_median'], help='Column to stratify by')
    p.add_argument('--min-snr', type=float, help='Minimum SNR threshold')
    p.add_argument('--quality-threshold', type=float, help='Minimum quality score threshold')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing splits')
    p.add_argument('--analyze-only', action='store_true', help='Only analyze dataset, don\'t create splits')
    args = p.parse_args(argv)
    
    if not os.path.exists(args.h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5_file}")
    
    # Analyze dataset
    analyze_dataset_composition(args.h5_file)
    
    if not args.analyze_only:
        # Create splits
        splits = create_data_splits(
            args.h5_file,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_seed=args.seed,
            stratify_by=args.stratify_by,
            min_snr=args.min_snr,
            quality_threshold=args.quality_threshold
        )
        
        # Save splits
        save_splits_to_hdf5(args.h5_file, splits, overwrite=args.overwrite)
        
        # Analyze splits
        analyze_dataset_composition(args.h5_file, splits)


if __name__ == '__main__':
    main()
