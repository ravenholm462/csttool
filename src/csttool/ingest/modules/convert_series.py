"""
convert_series.py

Convert DICOM series to NIfTI format with gradient files.

This module provides:
- DICOM to NIfTI conversion using dicom2nifti
- Gradient file (bval/bvec) handling
- Validation of conversion outputs
- Fallback conversion strategies
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import os
import shutil

try:
    from dicom2nifti import convert_dicom
except ImportError:
    convert_dicom = None


def convert_dicom_to_nifti(
    dicom_dir: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    reorient: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Convert a DICOM series to NIfTI format.
    
    Uses dicom2nifti for conversion, which handles:
    - Multi-frame DICOM
    - Siemens mosaic format
    - Gradient table extraction (bval/bvec)
    
    Parameters
    ----------
    dicom_dir : Path
        Path to directory containing DICOM files
    output_dir : Path
        Directory for output NIfTI and gradient files
    output_name : str, optional
        Base name for output files (without extension).
        If None, uses the DICOM directory name.
    reorient : bool, optional
        Reorient to standard orientation. Default is True.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    result : Dict
        Dictionary containing:
        - nifti_path: Path to output .nii.gz file
        - bval_path: Path to .bval file (or None)
        - bvec_path: Path to .bvec file (or None)
        - success: bool indicating conversion success
        - warnings: List of any warnings
        
    Raises
    ------
    ImportError
        If dicom2nifti is not installed
    RuntimeError
        If conversion fails
    """
    if convert_dicom is None:
        raise ImportError(
            "dicom2nifti is required for DICOM conversion. "
            "Install with: pip install dicom2nifti"
        )
    
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)
    
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    if output_name is None:
        output_name = _sanitize_filename(dicom_dir.name)
    
    output_nii = output_dir / f"{output_name}.nii.gz"
    
    if verbose:
        print("    → Converting: DICOM to NIfTI...")
        print(f"    • Input:  {dicom_dir}")
        print(f"    • Output: {output_nii}")
    
    result = {
        'nifti_path': None,
        'bval_path': None,
        'bvec_path': None,
        'success': False,
        'warnings': []
    }
    
    try:
        # Run conversion
        conversion_result = convert_dicom.dicom_series_to_nifti(
            str(dicom_dir),
            output_file=str(output_nii),
            reorient_nifti=reorient
        )
        
        # Extract paths from result
        nii_path = conversion_result.get("NII_FILE")
        bval_path = conversion_result.get("BVAL_FILE")
        bvec_path = conversion_result.get("BVEC_FILE")
        
        # Validate NIfTI was created
        if nii_path and Path(nii_path).exists():
            result['nifti_path'] = Path(nii_path)
            result['success'] = True
            
            if verbose:
                print(f"    ✓ NIfTI created: {nii_path}")
        else:
            result['warnings'].append("NIfTI file not created")
            raise RuntimeError("Conversion produced no output file")
        
        # Validate gradient files
        if bval_path and Path(bval_path).exists():
            result['bval_path'] = Path(bval_path)
            if verbose:
                print(f"    ✓ bval file: {bval_path}")
        else:
            result['warnings'].append("No .bval file generated")
            if verbose:
                print("    ⚠️ No .bval file generated")
        
        if bvec_path and Path(bvec_path).exists():
            result['bvec_path'] = Path(bvec_path)
            if verbose:
                print(f"    ✓ bvec file: {bvec_path}")
        else:
            result['warnings'].append("No .bvec file generated")
            if verbose:
                print("    ⚠️ No .bvec file generated")
        
        # Check if gradient files are needed
        if result['nifti_path'] and (not result['bval_path'] or not result['bvec_path']):
            _check_if_gradients_needed(result, verbose)
        
    except Exception as e:
        result['success'] = False
        result['warnings'].append(f"Conversion error: {str(e)}")
        print(f"  ✗ Conversion failed: {e}")
        raise RuntimeError(f"DICOM to NIfTI conversion failed: {e}")
    
    return result


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.
    
    Parameters
    ----------
    name : str
        Original name
        
    Returns
    -------
    sanitized : str
        Safe filename with problematic characters replaced
    """
    # Replace problematic characters
    replacements = {
        ' ': '_',
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
    }
    
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Remove leading/trailing underscores and dots
    result = result.strip('_.')
    
    # Ensure not empty
    if not result:
        result = "converted"
    
    return result


def _check_if_gradients_needed(result: Dict, verbose: bool) -> None:
    """
    Check if gradient files are actually needed for this dataset.
    
    For structural images (T1, T2), gradient files are not expected.
    For diffusion data, missing gradients is a problem.
    """
    import nibabel as nib
    
    try:
        img = nib.load(str(result['nifti_path']))
        shape = img.shape
        
        # Check if 4D (diffusion) or 3D (structural)
        if len(shape) == 3:
            # 3D image - likely structural, gradients not needed
            result['warnings'] = [w for w in result['warnings']
                                  if 'bval' not in w.lower() and 'bvec' not in w.lower()]
            if verbose:
                print("    • 3D volume detected - gradient files not required")
        elif len(shape) == 4 and shape[3] > 1:
            # 4D image - definitely needs gradients
            if not result['bval_path'] or not result['bvec_path']:
                result['warnings'].append(
                    "4D diffusion data detected but gradient files missing. "
                    "Tractography will not be possible without bval/bvec files."
                )
                if verbose:
                    print(f"    ⚠️ 4D data ({shape[3]} volumes) requires gradient files")
    except Exception as e:
        if verbose:
            print(f"    ⚠️ Could not verify image dimensions: {e}")


def validate_conversion(
    nifti_path: Path,
    bval_path: Optional[Path] = None,
    bvec_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """
    Validate converted NIfTI file and gradient files.
    
    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file
    bval_path : Path, optional
        Path to .bval file
    bvec_path : Path, optional
        Path to .bvec file
    verbose : bool
        Print validation results
        
    Returns
    -------
    validation : Dict
        Dictionary containing:
        - valid: bool - overall validity
        - nifti_valid: bool
        - gradients_valid: bool
        - data_shape: tuple
        - n_volumes: int
        - n_gradients: int
        - issues: list of any problems found
    """
    import nibabel as nib
    import numpy as np
    
    validation = {
        'valid': False,
        'nifti_valid': False,
        'gradients_valid': False,
        'data_shape': None,
        'n_volumes': 0,
        'n_gradients': 0,
        'issues': []
    }
    
    # Validate NIfTI
    try:
        img = nib.load(str(nifti_path))
        data_shape = img.shape
        validation['nifti_valid'] = True
        validation['data_shape'] = data_shape
        
        if len(data_shape) == 4:
            validation['n_volumes'] = data_shape[3]
        elif len(data_shape) == 3:
            validation['n_volumes'] = 1
        
        if verbose:
            print("    • NIfTI Validation:")
            print(f"    ├─ Shape: {data_shape}")
            print(f"    ├─ Voxel size: {img.header.get_zooms()[:3]}")
            print(f"    └─ Data type: {img.get_data_dtype()}")
    except Exception as e:
        validation['issues'].append(f"Invalid NIfTI: {e}")
        print(f"  ✗ Invalid NIfTI: {e}")
        return validation
    
    # Validate gradient files (if this is 4D diffusion data)
    if validation['n_volumes'] > 1 and bval_path and bvec_path:
        try:
            # Load bvals
            bvals = np.loadtxt(str(bval_path))
            if bvals.ndim == 0:
                bvals = np.array([bvals])
            bvals = bvals.flatten()
            
            # Load bvecs
            bvecs = np.loadtxt(str(bvec_path))
            if bvecs.shape[0] == 3:
                bvecs = bvecs.T  # Transpose if needed
            
            validation['n_gradients'] = len(bvals)
            
            # Check consistency
            if len(bvals) != validation['n_volumes']:
                validation['issues'].append(
                    f"Volume/gradient mismatch: {validation['n_volumes']} volumes "
                    f"but {len(bvals)} b-values"
                )
            elif bvecs.shape[0] != len(bvals):
                validation['issues'].append(
                    f"bval/bvec mismatch: {len(bvals)} b-values but "
                    f"{bvecs.shape[0]} gradient vectors"
                )
            else:
                validation['gradients_valid'] = True
            
            if verbose:
                print("    • Gradient Validation:")
                print(f"    ├─ B-values: {len(bvals)}")
                print(f"    ├─ Unique b-values: {sorted(set(bvals.astype(int)))}")
                print(f"    └─ Gradient vectors: {bvecs.shape}")
                
        except Exception as e:
            validation['issues'].append(f"Error reading gradient files: {e}")
            if verbose:
                print(f"    ⚠️ Error reading gradient files: {e}")
    
    elif validation['n_volumes'] == 1:
        # Single volume - gradients not required
        validation['gradients_valid'] = True
        if verbose:
            print("    • Single volume - gradient files not required")
    
    # Overall validity
    validation['valid'] = (validation['nifti_valid'] and 
                           validation['gradients_valid'] and
                           len(validation['issues']) == 0)
    
    if verbose:
        if validation['valid']:
            print("  ✓ Conversion validation passed")
        else:
            print("  ✗ Conversion validation failed")
            for issue in validation['issues']:
                print(f"    • {issue}")
    
    return validation


def convert_with_fallback(
    dicom_dir: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Convert DICOM to NIfTI with fallback strategies.
    
    Tries dicom2nifti first, then falls back to dcm2niix if available.
    
    Parameters
    ----------
    dicom_dir : Path
        Path to DICOM directory
    output_dir : Path
        Output directory
    output_name : str, optional
        Base output filename
    verbose : bool
        Print progress
        
    Returns
    -------
    result : Dict
        Conversion result dictionary
    """
    
    # Try dicom2nifti first
    try:
        result = convert_dicom_to_nifti(
            dicom_dir, output_dir, output_name, verbose=verbose
        )
        if result['success']:
            return result
    except Exception as e:
        if verbose:
            print(f"    ⚠️ dicom2nifti failed: {e}")
            print("    → Attempting fallback...")
    
    # Try dcm2niix as fallback
    try:
        result = _convert_with_dcm2niix(
            dicom_dir, output_dir, output_name, verbose
        )
        return result
    except Exception as e:
        if verbose:
            print(f"    ⚠️ dcm2niix fallback also failed: {e}")
    
    # Both failed
    raise RuntimeError(
        "DICOM conversion failed with all available methods. "
        "Ensure dicom2nifti or dcm2niix is properly installed."
    )


def _convert_with_dcm2niix(
    dicom_dir: Path,
    output_dir: Path,
    output_name: Optional[str],
    verbose: bool
) -> Dict:
    """
    Fallback conversion using dcm2niix command-line tool.
    """
    import subprocess
    
    # Check if dcm2niix is available
    try:
        subprocess.run(['dcm2niix', '-h'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("dcm2niix not found in PATH")
    
    if output_name is None:
        output_name = _sanitize_filename(dicom_dir.name)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("    → Using dcm2niix for conversion...")
    
    # Run dcm2niix
    cmd = [
        'dcm2niix',
        '-z', 'y',           # gzip output
        '-f', output_name,   # output filename
        '-o', str(output_dir),
        str(dicom_dir)
    ]
    
    result_proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if result_proc.returncode != 0:
        raise RuntimeError(f"dcm2niix failed: {result_proc.stderr}")
    
    # Find output files
    result = {
        'nifti_path': None,
        'bval_path': None,
        'bvec_path': None,
        'success': False,
        'warnings': []
    }
    
    nii_files = list(output_dir.glob(f"{output_name}*.nii.gz"))
    if nii_files:
        result['nifti_path'] = nii_files[0]
        result['success'] = True
    
    bval_files = list(output_dir.glob(f"{output_name}*.bval"))
    if bval_files:
        result['bval_path'] = bval_files[0]
    
    bvec_files = list(output_dir.glob(f"{output_name}*.bvec"))
    if bvec_files:
        result['bvec_path'] = bvec_files[0]
    
    if verbose:
        if result['success']:
            print("    ✓ dcm2niix conversion successful")
        else:
            print("    ✗ dcm2niix produced no output")
    
    return result