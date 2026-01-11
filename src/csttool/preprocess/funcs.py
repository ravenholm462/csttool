"""
funcs.py

Utility functions for csttool's preprocessing pipeline.

1. Load NIfTI + bvals/bvecs and build a gradient table
2. Estimate noise and denoise with NLMEANS
3. Compute a brain mask with median Otsu on b0 volumes
4. Perform between volume motion correction
5. Save the preprocessed data to disk
"""

import os
from os.path import join
from pathlib import Path
from time import time

import numpy as np
import matplotlib.pyplot as plt

from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dicom2nifti import convert_dicom
from dipy.core.gradients import gradient_table

from dipy.denoise.noise_estimate import piesno
from dipy.denoise.nlmeans import nlmeans

from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq

from dipy.align import motion_correction

def is_dicom_dir(p):
    """Check whether the given path is a directory containing DICOM files.

    Args:
        p (Path): Path to the directory to check.

    Returns:
        bool: True if the directory exists and contains at least one *.dcm file, otherwise False.
    """

    if not p.is_dir():
        return False
    if any(f.suffix.lower() == ".dcm" for f in p.iterdir()):
        return True
    return False
    
def convert_to_nifti(dicom_dir, out_dir):
    """Convert a directory of DICOM files to NIfTI format using dicom2nifti.

    Args:
        dicom_dir (str): str to the input DICOM directory.
        out_dir (str): str to the output directory where NIfTI and sidecar files will be saved.

    Returns:
        tuple[str, str, str]: Paths to the generated .nii(.gz), .bval, and .bvec files.
    """

    nifti_dir = os.path.join(out_dir, "nifti")
    os.makedirs(nifti_dir, exist_ok=True)

    # Convert inputs to strings for dicom2nifti
    dicom_dir_str = str(dicom_dir)
    study_name = os.path.basename(dicom_dir_str)
    output_nii = os.path.join(nifti_dir, study_name + ".nii.gz")
    
    result = convert_dicom.dicom_series_to_nifti(
        dicom_dir_str,
        output_file=output_nii,
        reorient_nifti=True
    )
    
    nii = result["NII_FILE"]
    
    # Get bval and bvec paths if they exist
    bval = result.get("BVAL_FILE")
    bvec = result.get("BVEC_FILE")

    # Check if files actually exist
    if bval is not None and not os.path.exists(bval):
        print(f"Warning: bval file doesn't exist: {bval}")
        bval = None
    if bvec is not None and not os.path.exists(bvec):
        print(f"Warning: bvec file doesn't exist: {bvec}")
        bvec = None

    if bval is None or bvec is None:
        print("Warning: dicom2nifti did not create .bval/.bvec for this series.")
        print(f"Generated NIfTI: {nii}")
        if bval:
            print(f"bval: {bval}")
        if bvec:
            print(f"bvec: {bvec}")

    return nii, bval, bvec

def load_dataset(
        nifti_path: str,
        fname: str,
        visualize: bool = False
):
    """
    Load a DWI dataset and its gradient information.

    Parameters
    ----------
    nifti_path : str
        Folder that contains .nii.gz, .bval and .bvec files.
    fname : str
        Base filename without extension.
    visualize : bool
        If True, show example slices from two volumes.

    Returns
    -------
    data : np.ndarray
        4D DWI data array.
    affine : np.ndarray
        4x4 affine matrix from the NIfTI header.
    img : nibabel.Nifti1Image
        NIfTI image object. Contains NIfTI object metadata.
    gtab : dipy.core.gradients.GradientTable
        Gradient table built from bvals and bvecs.
    bvals : np.ndarray
        1D array of b values.
    bvecs : np.ndarray
        2D array of b vectors.
    """
    if nifti_path is None or fname is None:
        raise ValueError("nifti_path and fname must be provided")

    fdwi = join(nifti_path, fname + ".nii.gz")
    print(f"Found NIfTI dataset: {fdwi}")

    # Try to find gradient files with flexible extension support (.bval/.bvec or .bvals/.bvecs)
    fbval = None
    fbvec = None
    
    for bval_ext, bvec_ext in [('.bval', '.bvec'), ('.bvals', '.bvecs')]:
        test_bval = join(nifti_path, fname + bval_ext)
        test_bvec = join(nifti_path, fname + bvec_ext)
        
        if os.path.exists(test_bval) and os.path.exists(test_bvec):
            fbval = test_bval
            fbvec = test_bvec
            break
    
    if fbval is None or fbvec is None:
        raise FileNotFoundError(
            f"Could not find gradient files for '{fname}' in {nifti_path}. "
            f"Tried: {fname}.bval/.bvec and {fname}.bvals/.bvecs"
        )
    
    print(f"Found gradient files: {fbval}")
    print(f"                      {fbvec}")

    # Load dataset, show shape and voxel size
    data, affine, img = load_nifti(fdwi, return_img=True)
    print("\n" + "=" * 70)
    print(f"Loaded study {fname}")
    print(f"Data shape: {data.shape}")
    print(f"Voxel size [mm]: {img.header.get_zooms()[:3]}")
    print("=" * 70 + "\n")

    if visualize:
        axial_middle = data.shape[2] // 2
        plt.figure("DWI dataset overview")
        plt.suptitle(f"Study: {fname}")

        plt.subplot(1, 2, 1)
        plt.axis("off")
        volume = 0
        plt.imshow(data[:, :, axial_middle, volume].T,
                   cmap="gray", origin="lower")
        plt.title(f"Axial slice, volume {volume}")

        plt.subplot(1, 2, 2)
        plt.axis("off")
        volume = min(10, data.shape[3] - 1)
        plt.imshow(data[:, :, axial_middle, volume].T,
                   cmap="gray", origin="lower")
        plt.title(f"Axial slice, volume {volume}")

        plt.show()

    # Read bvalues and bvectors, build a gradient table
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    num_of_gradients = len(gtab)

    print(gtab.info)
    print(f"Number of gradients: {num_of_gradients}")
    print("\n" + "=" * 70 + "\n")

    return data, affine, img, gtab


def denoise_nlmeans(
        data: np.ndarray,
        N: int = 4,
        brain_mask: np.ndarray | None = None,
        visualize: bool = False
):
    """
    Denoise DWI data using PIESNO noise estimation and NLMEANS.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    N : int
        Number of receiver coils for PIESNO. Use 4 for GRAPPA, 1 for SENSE.
    brain_mask : np.ndarray or None
        Optional brain mask. If None, a mask is derived from PIESNO.
    visualize : bool
        If True, show before and after denoising for a single slice and volume.

    Returns
    -------
    den : np.ndarray
        Denoised 4D DWI data.
    brain_mask : np.ndarray
        Brain mask used during denoising.
    """
    print("Estimating noise using PIESNO...")
    sigma_map, noise_mask = piesno(data, N, return_mask=True)
    sigma = float(np.mean(sigma_map))

    if brain_mask is None:
        brain_mask = ~noise_mask

    print(f"Mean noise std: {sigma}")
    print("Background std =", float(np.std(data[noise_mask])))
    print("Brain std      =", float(np.std(data[brain_mask])))

    print("\nDenoising using NLMEANS...")
    t = time()

    den = nlmeans(
        data.astype(np.float32),
        sigma=sigma,
        mask=brain_mask,
        patch_radius=1,
        block_radius=2,
        rician=False,  # debatable
        num_threads=-1,
    )

    print(f"Denoising complete. Total time elapsed: {time() - t:.2f} s\n")

    if visualize:
        axial_middle = brain_mask.shape[2] // 2
        vol_idx = 0  # show first volume, often a b0

        before = data[:, :, axial_middle, vol_idx]
        after = den[:, :, axial_middle, vol_idx]

        difference = np.abs(
            after.astype(np.float64) - before.astype(np.float64)
        )

        slice_mask = brain_mask[:, :, axial_middle]
        difference[~slice_mask] = 0

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(before, cmap="gray", origin="lower")
        ax[0].set_title("Noisy")
        ax[0].axis("off")

        ax[1].imshow(after, cmap="gray", origin="lower")
        ax[1].set_title("Denoised")
        ax[1].axis("off")

        ax[2].imshow(difference, cmap="gray", origin="lower")
        ax[2].set_title("Difference inside brain")
        ax[2].axis("off")

        plt.show()
        print("=" * 70 + "\n")

    return den, brain_mask


def background_segmentation(
        data: np.ndarray,
        gtab,
        median_radius: int = 2,
        numpass: int = 1,
        visualize: bool = False
):
    """
    Compute a brain mask using median Otsu on b0 volumes and apply it.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array. Denoised data is recommended here.
    gtab : GradientTable
        Gradient table used to identify b0 volumes.
    median_radius : int
        Radius of the median filter.
    numpass : int
        Number of passes of the median filter.
    visualize : bool
        If True, show a central slice before and after masking.

    Returns
    -------
    masked_data : np.ndarray
        Brain masked 4D DWI data.
    brain_mask : np.ndarray
        Binary brain mask.
    """
    print("Applying median Otsu segmentation...")

    b0_idx = np.where(gtab.bvals < 50)[0]
    if b0_idx.size == 0:
        raise RuntimeError("No b0 volumes found with bvals < 50")

    masked_data, brain_mask = median_otsu(
        data,
        vol_idx=b0_idx,
        median_radius=median_radius,
        numpass=numpass
    )

    if visualize:
        sli = data.shape[2] // 2
        b0_vol0 = data[:, :, sli, b0_idx[0]]
        b0_masked_vol0 = masked_data[:, :, sli, 0]

        plt.figure("Brain segmentation")
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.imshow(histeq(b0_vol0.astype(float)).T,
                   cmap="gray", origin="lower")
        plt.title("b0 slice")

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.imshow(histeq(b0_masked_vol0.astype(float)).T,
                   cmap="gray", origin="lower")
        plt.title("Median Otsu brain masked b0")

        plt.show()

    print("Median Otsu segmentation complete\n")

    return masked_data, brain_mask


def perform_motion_correction(
        data: np.ndarray,
        gtab,
        affine: np.ndarray,
        brain_mask: np.ndarray | None = None
):
     print("Performing between volume motion correction...")
     t = time()

     if brain_mask is not None:
         # Ensure the mask is binary and uint8
         brain_mask = brain_mask.astype(np.uint8)

     try:
         if brain_mask is not None:
             data_corrected, reg_affines = motion_correction(
                 data,
                 gtab,
                 affine=affine,
                 static_mask=brain_mask
             )
         else:
             data_corrected, reg_affines = motion_correction(
                 data,
                 gtab,
                 affine=affine
             )
     except Exception as e:
         print(f"Motion correction failed: {e}")
         # Try without the mask
         print("Trying without mask...")
         try:
             data_corrected, reg_affines = motion_correction(
                 data,
                 gtab,
                 affine=affine
             )
         except Exception as e2:
             print(f"Motion correction without mask also failed: {e2}")
             print("Returning original data.")
             data_corrected = data
             reg_affines = [affine] * data.shape[-1]

     print(f"Motion correction complete. Total time elapsed: {time() - t:.2f} s\n")

     return data_corrected, reg_affines


def save_output(data, affine, out_dir, stem, save_intermediates=True, motion_correction_applied=False):
    """
    Save preprocessed data with organized structure and clear naming.
    """
    from pathlib import Path
    import nibabel as nib
    import json
    from datetime import datetime

     # SAFETY CHECK: Ensure data is a numpy array, not a Nifti1Image
    if hasattr(data, 'get_fdata'):
        print("Converting Nifti1Image to numpy array for saving...")
        data = data.get_fdata()
    
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Data must be a numpy array, got {type(data)}")
    
    out_dir = Path(out_dir)
    
    # Create directory structure
    preproc_dir = out_dir / "preprocessed"
    intermediate_dir = out_dir / "intermediate" 
    log_dir = out_dir / "logs"
    
    preproc_dir.mkdir(parents=True, exist_ok=True)
    if save_intermediates:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main preprocessed output - include motion correction status in filename
    motion_status = "mc" if motion_correction_applied else "nomc"
    preproc_path = preproc_dir / f"{stem}_dwi_preproc_{motion_status}.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), preproc_path)
    
    # File paths dictionary - CONVERT ALL PATHS TO STRINGS
    file_paths = {
        'preprocessed_dwi': str(preproc_path),  # Convert to string
        'timestamp': timestamp,
        'motion_correction_applied': motion_correction_applied,
        'output_structure': {
            'preprocessed': str(preproc_dir),      # Convert to string
            'intermediate': str(intermediate_dir), # Convert to string  
            'logs': str(log_dir)                   # Convert to string
        }
    }
    
    # Create processing report
    report = {
        'processing_date': timestamp,
        'input_stem': stem,
        'output_files': file_paths,
        'data_shape': data.shape,
        'data_dtype': str(data.dtype),
        'voxel_size': np.sqrt(np.sum(affine[:3, :3]**2, axis=0)).tolist(),
        'motion_correction_applied': motion_correction_applied,
        'motion_correction_status': 'success' if motion_correction_applied else 'skipped_or_failed'
    }
    
    report_path = log_dir / f"{stem}_preprocessing_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    file_paths['processing_report'] = str(report_path)  # Convert to string
    
    print(f"✓ Preprocessed data saved to: {preproc_path}")
    print(f"✓ Processing report saved to: {report_path}")
    print(f"✓ Motion correction: {'APPLIED' if motion_correction_applied else 'NOT APPLIED'}")
    
    return file_paths


def copy_gradient_files(original_nii_path, out_dir, stem, motion_correction_applied=False):
    """Simply copy .bval and .bvec files to output directory."""
    from pathlib import Path
    import shutil
    
    original_dir = Path(original_nii_path).parent
    out_dir = Path(out_dir) / "preprocessed"
    
    motion_status = "mc" if motion_correction_applied else "nomc"
    
    # Detect which extension is used (.bval/.bvec or .bvals/.bvecs)
    bval_src = None
    bvec_src = None
    
    for bval_ext, bvec_ext in [('.bval', '.bvec'), ('.bvals', '.bvecs')]:
        test_bval = original_dir / f"{stem}{bval_ext}"
        test_bvec = original_dir / f"{stem}{bvec_ext}"
        
        if test_bval.exists() and test_bvec.exists():
            bval_src = test_bval
            bvec_src = test_bvec
            break
    
    if bval_src is None or bvec_src is None:
        print(f"⚠️  Warning: Could not find gradient files for {stem}")
        return
    
    # Always save as .bval/.bvec (singular) for consistency
    bval_dest = out_dir / f"{stem}_dwi_preproc_{motion_status}.bval"
    bvec_dest = out_dir / f"{stem}_dwi_preproc_{motion_status}.bvec"
    
    shutil.copy2(bval_src, bval_dest)
    print(f"✓ Copied .bval to: {bval_dest}")
    
    shutil.copy2(bvec_src, bvec_dest)
    print(f"✓ Copied .bvec to: {bvec_dest}")


def save_brain_mask(mask, affine, out_dir, stem):
    """Save brain mask with consistent naming."""
    import nibabel as nib
    from pathlib import Path
    
    out_dir = Path(out_dir) / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mask_path = out_dir / f"{stem}_brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)
    
    print(f"✓ Brain mask saved to: {mask_path}")
    return mask_path


def save_denoised_data(data, affine, out_dir, stem):
    """Save denoised data (intermediate file)."""
    import nibabel as nib
    from pathlib import Path
    
    out_dir = Path(out_dir) / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    denoised_path = out_dir / f"{stem}_dwi_denoised.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), denoised_path)
    
    return denoised_path

def save_preprocessing_report(stem, out_dir, motion_correction_applied, params=None):
    """Save preprocessing report as JSON."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    out_dir = Path(out_dir) / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'subject_id': stem,
        'timestamp': datetime.now().isoformat(),
        'motion_correction_applied': motion_correction_applied,
        'parameters': params or {}
    }
    
    report_path = out_dir / f"{stem}_preprocessing_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Preprocessing report saved to: {report_path}")
    return report_path
