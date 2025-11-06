import nibabel as nib
import numpy as np
from pathlib import Path

from dipy.align.reslice import reslice
from dipy.align import motion_correction
from dipy.io.image import save_nifti
from dipy.segment.mask import median_otsu

from .import_data import load_data


def check_voxel_size(voxel_size: tuple[float, float, float], tol: float = 1e-3) -> bool:
    """Check if voxel size is isotropic.

    Args:
        voxel_size (tuple[float, float, float]): Voxel size in mm (x, y, z).
        tol (float, optional): Tolerance for differences. Defaults to 1e-3.

    Returns:
        bool: True if voxel sizes are isotropic within tolerance.
    """
    return np.allclose(voxel_size, voxel_size[0], atol=tol)


def reslice_data(
    data: np.ndarray,
    affine: np.ndarray,
    voxel_size: tuple[float, float, float],
    target_voxel_size: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Reslice data to isotropic voxel size.

    Args:
        data (np.ndarray): 3D or 4D image array.
        affine (np.ndarray): 4x4 affine matrix.
        voxel_size (tuple[float, float, float]): Voxel dimensions (mm).
        target_voxel_size (float, optional): Target voxel size in mm.
            Defaults to 2.0.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
            Resliced data, new affine, and new voxel size.
    """
    new_voxel_size = target_voxel_size
    new_zoom = (new_voxel_size, new_voxel_size, new_voxel_size)
    data_resliced, affine_resliced = reslice(data, affine, voxel_size, new_zoom)
    return data_resliced, affine_resliced, new_zoom


def compute_brain_mask(
    data: np.ndarray,
    affine: np.ndarray,
    gtab,
    mask_path: str | Path,
    median_radius: int = 2,
    numpass: int = 1,
) -> np.ndarray:
    """Compute a brain mask from the mean b0 image and save as NIfTI.

    Args:
        data: 4D DWI array (x, y, z, n_volumes).
        affine: 4x4 affine matrix.
        gtab: DIPY GradientTable for the dataset.
        mask_path: Where to save the binary mask NIfTI.
        median_radius: Radius for median_otsu.
        numpass: Number of passes for median_otsu.

    Returns:
        mask: 3D boolean array (x, y, z) with brain voxels = True.
    """
    mask_path = Path(mask_path)

    # Use all b0 volumes to form a higher SNR reference
    b0_idx = np.where(gtab.b0s_mask)[0]
    if b0_idx.size == 0:
        raise ValueError("No b0 volumes found in gradient table – cannot build brain mask.")

    mean_b0 = np.mean(data[..., b0_idx], axis=-1)

    # DIPY's median_otsu: returns (segmented_image, mask)
    _, mask = median_otsu(mean_b0, median_radius=median_radius, numpass=numpass)

    # Save mask as NIfTI (float32 or uint8)
    save_nifti(str(mask_path), mask.astype(np.float32), affine)
    print(f"Brain mask saved → {mask_path}")

    return mask


def motion_correct_data(
    data: np.ndarray,
    affine: np.ndarray,
    gtab,
    static_mask: np.ndarray | None = None,
    b0_ref: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Between-volumes motion correction using DIPY."""

    import io
    import contextlib
    
    if static_mask is not None:
        if static_mask.ndim != 3:
            raise ValueError(
                f"static_mask must be 3D, got shape {static_mask.shape}"
            )
        if static_mask.shape != data.shape[:3]:
            raise ValueError(
                f"static_mask shape {static_mask.shape} does not match "
                f"data spatial shape {data.shape[:3]}"
            )
        static_mask = static_mask.astype(np.float32)

    print("Running DIPY motion correction...")

    if verbose:
        # normal noisy behavior
        mc_img, affines = motion_correction(
            data,
            gtab,
            affine=affine,
            b0_ref=b0_ref,
            pipeline=['center_of_mass', 'translation', 'rigid', 'affine'],
            static_mask=static_mask,
        )
    else:
        # swallow all the "Optimizing level ..." prints
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mc_img, affines = motion_correction(
                data,
                gtab,
                affine=affine,
                b0_ref=b0_ref,
                pipeline=['center_of_mass', 'translation', 'rigid', 'affine'],
                static_mask=static_mask,
            )

    data_mc = mc_img.get_fdata(dtype=np.float32)
    affine_mc = mc_img.affine
    print("Motion correction done.")
    return data_mc, affine_mc

def estimate_noise(
    data: np.ndarray,
    num_coils: int,
    return_mask: bool = True
) -> tuple[float, np.ndarray | None]:
    """Estimate STD of noise in data using PIESNO.

    Args:
        data: 4D DWI array.
        num_coils: The number of phase array coils of the MRI scanner.
        return_mask: If True, return a mask identifying pure noise voxels.

    Returns:
        sigma: Estimated noise standard deviation (scalar).
        mask: Noise mask (if requested and available), otherwise None.
    """
    from dipy.denoise.noise_estimate import piesno

    result = piesno(data, num_coils, return_mask=return_mask)

    # Be robust to different DIPY versions / return signatures
    if isinstance(result, (tuple, list)):
        # Always treat first element as sigma
        sigma = result[0]
        mask = result[1] if len(result) > 1 else None
    else:
        sigma = result
        mask = None

    # Ensure sigma is a scalar
    if isinstance(sigma, np.ndarray):
        sigma_value = float(np.mean(sigma))
    else:
        sigma_value = float(sigma)

    if return_mask and mask is not None:
        background_std = np.std(data[mask.astype(bool)])
        print(f"Background noise STD: {background_std:.4f}")

    print(f"Estimated noise sigma (PIESNO): {sigma_value:.4f}")
    return sigma_value, mask

def denoise_nlmeans(
    data: np.ndarray,
    sigma: float | None = None,
    mask: np.ndarray | None = None,
    num_coils: int = 1,
    patch_radius: int = 1,
    block_radius: int = 2,
    rician: bool = True,
) -> np.ndarray:
    """Perform non-local means denoising.

    Args:
        data: 4D DWI array.
        sigma: Noise standard deviation. If None, will be estimated.
        mask: 3D brain mask. Optional.
        num_coils: Number of receiver coils for sigma estimation.
        patch_radius: Patch radius for NLMeans.
        block_radius: Block radius for NLMeans.
        rician: If True, assumes Rician noise distribution.

    Returns:
        Denoised 4D array.
    """
    from dipy.denoise.nlmeans import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma

    if sigma is None:
        print("Estimating noise sigma for denoising...")
        sigma = estimate_sigma(data, N=num_coils)
        print(f"Estimated sigma: {sigma if np.isscalar(sigma) else np.mean(sigma):.4f}")
    
    # Ensure sigma is a scalar - take mean if it's an array
    if isinstance(sigma, np.ndarray):
        sigma = float(np.mean(sigma))
    
    print("Running NLMeans denoising...")
    den = nlmeans(
        data,
        sigma=sigma,
        mask=mask,
        patch_radius=patch_radius,
        block_radius=block_radius,
        rician=rician,
        num_threads=-1  # use all available threads
    )
    print("Denoising complete.")
    
    return den

def suppress_gibbs(
    data: np.ndarray,
    slice_axis: int = 2,
    num_processes: int = 1,
) -> np.ndarray:
    """Suppress Gibbs ringing artifacts using total variation.

    Args:
        data: 4D DWI array.
        slice_axis: Axis along which to apply Gibbs suppression (0, 1, or 2).
            Default is 2 (axial slices).
        num_processes: Number of processes for parallel processing.
            Use -1 for all available cores, 1 for single-threaded.

    Returns:
        Data with suppressed Gibbs artifacts.
    """
    from dipy.denoise.gibbs import gibbs_removal

    print("Running Gibbs artifact suppression...")
    data_corrected = gibbs_removal(
        data,
        slice_axis=slice_axis,
        num_processes=num_processes
    )
    print("Gibbs suppression complete.")
    
    return data_corrected

def process_and_save(
    nifti_path: str | Path,
    bval_path: str | Path,
    bvec_path: str | Path,
    output_path: str | Path,
    target_voxel_size: float = 2.0,
    b0_threshold: int = 50,
    num_coils: int = 1,
    denoise: bool = True,
    suppress_gibbs_artifacts: bool = True,
    gibbs_slice_axis: int = 2,
    save_intermediate: bool = False,
) -> None:
    """Complete preprocessing pipeline with noise estimation.
    
    Pipeline: load → estimate noise → denoise → suppress Gibbs → reslice → 
              skull strip → motion correct → estimate noise again → save
    
    Args:
        nifti_path: Path to input NIfTI.
        bval_path: Path to b-values file.
        bvec_path: Path to b-vectors file.
        output_path: Path for final output.
        target_voxel_size: Desired isotropic voxel size in mm.
        b0_threshold: Maximum b-value considered a b0 image.
        num_coils: Number of receiver coils (1 for SENSE, N for GRAPPA).
        denoise: Whether to perform denoising.
        suppress_gibbs_artifacts: Whether to suppress Gibbs ringing artifacts.
        gibbs_slice_axis: Axis for Gibbs suppression (0=sagittal, 1=coronal, 2=axial).
        save_intermediate: Whether to save intermediate results.
    """
    nifti_path = Path(nifti_path)
    bval_path = Path(bval_path)
    bvec_path = Path(bvec_path)
    output_path = Path(output_path)

    print("="*70)
    print("STEP 1: Loading data")
    print("="*70)
    
    # 1) Load data + gradient table
    data, affine, hdr, gtab = load_data(
        nifti_path,
        bval_path=bval_path,
        bvec_path=bvec_path,
        b0_threshold=b0_threshold,
    )
    voxel_size = hdr.get_zooms()[:3]
    print(f"Loaded data: shape={data.shape}, voxel_size={voxel_size}")
    print(f"Number of gradients: {len(gtab.bvals)}")
    print(f"B-values range: {gtab.bvals.min():.0f} - {gtab.bvals.max():.0f}")

    print("\n" + "="*70)
    print("STEP 2: Pre-processing noise estimation")
    print("="*70)
    
    # 2) Estimate noise before preprocessing
    sigma_pre, noise_mask = estimate_noise(data, num_coils, return_mask=True)
    
    if save_intermediate and noise_mask is not None:
        noise_mask_path = output_path.with_name(output_path.stem + "_noise_mask_pre.nii.gz")
        save_nifti(str(noise_mask_path), noise_mask.astype(np.uint8), affine)
        print(f"Pre-processing noise mask saved → {noise_mask_path}")

    # 3) Denoise if requested
    current_data = data
    
    if denoise:
        print("\n" + "="*70)
        print("STEP 3: Denoising")
        print("="*70)
        current_data = denoise_nlmeans(current_data, sigma=sigma_pre, num_coils=num_coils)
        
        if save_intermediate:
            denoised_path = output_path.with_name(output_path.stem + "_denoised.nii.gz")
            save_nifti(str(denoised_path), current_data, affine)
            print(f"Denoised data saved → {denoised_path}")
    else:
        print("\n" + "="*70)
        print("STEP 3: Denoising - SKIPPED")
        print("="*70)

    # 4) Suppress Gibbs artifacts if requested
    if suppress_gibbs_artifacts:
        print("\n" + "="*70)
        print("STEP 4: Gibbs artifact suppression")
        print("="*70)
        current_data = suppress_gibbs(current_data, slice_axis=gibbs_slice_axis, num_processes=-1)
        
        if save_intermediate:
            gibbs_path = output_path.with_name(output_path.stem + "_gibbs_corrected.nii.gz")
            save_nifti(str(gibbs_path), current_data, affine)
            print(f"Gibbs-corrected data saved → {gibbs_path}")
    else:
        print("\n" + "="*70)
        print("STEP 4: Gibbs artifact suppression - SKIPPED")
        print("="*70)

    print("\n" + "="*70)
    print("STEP 5: Reslicing (if needed)")
    print("="*70)
    
    # 5) Reslice if needed
    data_rs, affine_rs = current_data, affine
    if not check_voxel_size(voxel_size):
        print(f"Voxel sizes not isotropic ({voxel_size}). Reslicing...")
        data_rs, affine_rs, new_vox = reslice_data(
            current_data, affine, voxel_size, target_voxel_size
        )
        print(f"Resliced to {new_vox} mm voxels.")
    else:
        print(f"Voxel sizes already isotropic: {voxel_size}")

    print("\n" + "="*70)
    print("STEP 6: Brain mask computation")
    print("="*70)
    
    # 6) Brain segmentation / skull stripping (on resliced data)
    mask_path = output_path.with_name(output_path.stem + "_brain_mask.nii.gz")
    brain_mask = compute_brain_mask(data_rs, affine_rs, gtab, mask_path)

    print("\n" + "="*70)
    print("STEP 7: Motion correction")
    print("="*70)
    
    # 7) Motion correction using that mask
    data_mc, affine_mc = motion_correct_data(
        data_rs,
        affine_rs,
        gtab,
        static_mask=brain_mask,
    )

    print("\n" + "="*70)
    print("STEP 8: Post-processing noise estimation")
    print("="*70)
    
    # 8) Estimate noise after preprocessing
    sigma_post, _ = estimate_noise(data_mc, num_coils, return_mask=False)
    
    print("\n" + "="*70)
    print("Noise comparison:")
    print(f"  Pre-processing sigma:  {sigma_pre:.4f}")
    print(f"  Post-processing sigma: {sigma_post:.4f}")
    print(f"  Noise reduction:       {((sigma_pre - sigma_post) / sigma_pre * 100):.2f}%")
    print("="*70)

    print("\n" + "="*70)
    print("STEP 9: Saving final output")
    print("="*70)
    
    # 9) Save final preprocessed data
    save_nifti(str(output_path), data_mc, affine_mc)
    print(f"Preprocessing complete → {output_path}")
    print("="*70)