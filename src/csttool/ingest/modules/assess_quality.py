from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from csttool.tracking.modules.estimate_directions import get_max_sh_order


def count_unique_directions(bvecs: np.ndarray, bvals: np.ndarray, b0_threshold: int = 50) -> int:
    """Count unique gradient directions (excluding b=0 volumes).
    
    Args:
        bvecs: (N, 3) or (3, N) array of gradient directions
        bvals: (N,) array of b-values
        b0_threshold: b-value threshold for b=0 volumes
        
    Returns:
        Number of unique gradient directions
    """
    if bvals.ndim == 2:
        bvals = bvals.flatten()
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
        
    dwi_mask = bvals > b0_threshold
    if not np.any(dwi_mask):
        return 0
        
    dwi_bvecs = bvecs[dwi_mask]
    norms = np.linalg.norm(dwi_bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_bvecs = dwi_bvecs / norms
    
    unique_dirs = np.unique(np.round(normalized_bvecs, decimals=4), axis=0)
    return len(unique_dirs)


def extract_acquisition_metadata(
    bvecs: np.ndarray,
    bvals: np.ndarray,
    voxel_size: Tuple[float, float, float],
    bids_json: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract acquisition parameters for reporting.
    
    Args:
        bvecs: (N, 3) or (3, N) array of gradient directions
        bvals: (N,) array of b-values
        voxel_size: (x, y, z) voxel dimensions in mm
        bids_json: Optional BIDS JSON metadata dict
        overrides: Optional dict of CLI-provided values (take precedence)
        
    Returns:
        Dict with acquisition metadata for reports
    """
    overrides = overrides or {}
    bids_json = bids_json or {}
    
    if bvals.ndim == 2:
        bvals = bvals.flatten()
    
    # Extract from BIDS JSON (convert units where needed)
    field_strength = None
    if 'MagneticFieldStrength' in bids_json:
        try:
            field_strength = float(bids_json['MagneticFieldStrength'])
        except (ValueError, TypeError):
            pass
    
    echo_time_ms = None
    if 'EchoTime' in bids_json:
        try:
            # BIDS stores in seconds, convert to ms
            echo_time_ms = float(bids_json['EchoTime']) * 1000
        except (ValueError, TypeError):
            pass
    
    acq = {
        'field_strength_T': field_strength,
        'echo_time_ms': echo_time_ms,
        'b_values': sorted(set(bvals.astype(int).tolist())),
        'n_directions': count_unique_directions(bvecs, bvals),
        'n_volumes': len(bvals),
        'resolution_mm': list(voxel_size),
    }
    
    # CLI overrides take precedence
    acq.update(overrides)
    
    return acq

def assess_acquisition_quality(
    bvecs: np.ndarray,
    bvals: np.ndarray,
    voxel_size: Tuple[float, float, float],
    bids_json: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, str]]:
    """
    Assess DWI acquisition quality for CST tractography.
    
    Returns list of (severity, message) tuples.
    Severity levels: "CRITICAL", "WARNING", "INFO"
    
    Args:
        bvecs: (N, 3) or (3, N) array of gradient directions
        bvals: (N,) or (1, N) array of b-values
        voxel_size: (x, y, z) voxel dimensions in mm
        bids_json: Optional dictionary of BIDS metadata
        
    Returns:
        List of (severity, message) tuples
    """
    warnings = []
    
    # Ensure correct shapes
    if bvals.ndim == 2:
        bvals = bvals.flatten()
    
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
        
    # Count gradient directions (exclude b=0)
    b0_threshold = 50  # s/mm²
    dwi_mask = bvals > b0_threshold
    
    # Check if we have any DWI volumes
    if not np.any(dwi_mask):
         warnings.append((
            "CRITICAL",
            "No diffusion-weighted volumes found (all b-values <= 50)."
        ))
         return warnings

    # Count unique directions
    # Normalize bvecs to handle slight variations
    dwi_bvecs = bvecs[dwi_mask]
    norms = np.linalg.norm(dwi_bvecs, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    normalized_bvecs = dwi_bvecs / norms
    
    # Round to avoid floating point precision issues
    unique_dirs = np.unique(np.round(normalized_bvecs, decimals=4), axis=0)
    n_directions = len(unique_dirs)
    
    if n_directions < 15:
        warnings.append((
            "CRITICAL",
            f"Only {n_directions} gradient directions detected. "
            f"Minimum 15 required for basic tractography, 28+ recommended."
        ))
    elif n_directions < 28:
        warnings.append((
            "WARNING",
            f"{n_directions} gradient directions limits SH order to 4. "
            f"Consider acquiring >=28 directions for SH order 6."
        ))
    
    # Check b-value
    max_bval = np.max(bvals)
    if max_bval < 800:
        warnings.append((
            "WARNING",
            f"Maximum b-value ({max_bval:.0f} s/mm²) may underestimate FA "
            f"and reduce diffusion contrast."
        ))
    elif max_bval > 3000:
        warnings.append((
            "WARNING",
            f"High b-value ({max_bval:.0f} s/mm²) may have SNR limitations. "
            f"Ensure adequate denoising."
        ))
    
    # Check voxel size
    max_voxel = max(voxel_size)
    if max_voxel > 2.5:
        warnings.append((
            "WARNING",
            f"Large voxel size ({voxel_size[0]:.1f}x{voxel_size[1]:.1f}x"
            f"{voxel_size[2]:.1f} mm) may cause partial volume effects "
            f"in internal capsule and brainstem."
        ))
    
    # Check for anisotropic voxels
    voxel_ratio = max(voxel_size) / min(voxel_size)
    if voxel_ratio > 1.5:
        warnings.append((
            "WARNING",
            f"Anisotropic voxels (ratio {voxel_ratio:.1f}) may cause "
            f"directional bias in tractography."
        ))
    
    # Check JSON-derived fields if available
    if bids_json:
        # Check echo time
        echo_time = bids_json.get("EchoTime")
        if echo_time:
            # Handle potential string/float differences and units
            # BIDS specifies EchoTime in seconds
            try:
                echo_time_ms = float(echo_time) * 1000
                if echo_time_ms > 100:
                    warnings.append((
                        "WARNING",
                        f"Long echo time ({echo_time_ms:.0f} ms) reduces SNR due to T2 decay."
                    ))
            except (ValueError, TypeError):
                pass
        
        # Check multiband factor
        mb_factor = bids_json.get("MultibandAccelerationFactor")
        if mb_factor:
            try:
                if float(mb_factor) > 4:
                    warnings.append((
                        "INFO",
                        f"High multiband factor ({mb_factor}) may reduce SNR. "
                        f"Verify image quality."
                    ))
            except (ValueError, TypeError):
                pass
        
        # Check parallel imaging factor
        parallel_factor = bids_json.get("ParallelReductionFactorInPlane")
        if parallel_factor:
            try:
                if float(parallel_factor) > 3:
                    warnings.append((
                        "INFO",
                        f"High parallel imaging factor ({parallel_factor}) may reduce SNR."
                    ))
            except (ValueError, TypeError):
                pass
    
    return warnings
