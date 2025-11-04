import nibabel as nib
import numpy as np
import os

from pathlib import Path
from dipy.align.reslice import reslice
# from dipy.align import motion_correction
from multiprocessing import cpu_count
from dipy.io.image import save_nifti
from .import_data import load_data
from dipy.segment.mask import median_otsu


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
        target_voxel_size (float | None, optional): Target voxel size in mm.
            If None, the function defaults to 2.0 mm.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
            Resliced data, new affine, and new voxel size.
    """
    new_voxel_size = target_voxel_size
    new_zoom = (new_voxel_size, new_voxel_size, new_voxel_size)
    data_resliced, affine_resliced = reslice(data, affine, voxel_size, new_zoom)
    return data_resliced, affine_resliced, new_zoom

def motion_correct_data(
    nifti_path: str | Path,
    bval_path: str | Path,
    bvec_path: str | Path,
    out_path: str | Path,
    n_jobs: int | None = None,
    omp_nthreads: int | None = None,
    seed: int = 42
) -> Path:
    """Between volumes motion correction using nifreeze.

    Args:
        nifti_path: Path to input NIfTI file
        bval_path: Path to b-values file
        bvec_path: Path to b-vectors file
        out_path: Path for output corrected NIfTI
        n_jobs: Number of parallel jobs
        omp_nthreads: Number of OpenMP threads
        seed: Random seed for reproducibility

    Returns:
        Path: Path to the saved motion-corrected NIfTI file
    """
    from nifreeze.model import DTIModel
    from nifreeze.estimator import Estimator
    from nifreeze.data import dmri
    os.environ["OMP_NUM_THREADS"] = "1"

    nifti_path = Path(nifti_path)
    bval_path = Path(bval_path)
    bvec_path = Path(bvec_path)
    out_path = Path(out_path)

    # 1) Build NiFreeze DWI dataset from disk
    # NiFreeze handles bvals/bvecs internally; no DIPY GradientTable needed.
    dataset = dmri.from_nii(
        str(nifti_path),
        bval_file=str(bval_path),
        bvec_file=str(bvec_path),
    )

    # Create model instance with the dataset
    model = DTIModel(dataset)
    
    # Load estimator with the model
    estimator = Estimator(
        model=model,
        strategy="random",
    )
    
    # Multithreading options
    if omp_nthreads is None:
        omp_nthreads = cpu_count()
    if n_jobs is None:
        n_jobs = 1
    
    # Run the estimator
    _ = estimator.run(
        dataset,
        omp_nthreads=omp_nthreads,
        n_jobs=n_jobs,
        seed=seed,
    )
    
    # Save corrected data as NIfTI
    dataset.to_nifti(str(out_path))
    
    return out_path

def process_and_save(
    nifti_path: str | Path,
    bval_path: str | Path,
    bvec_path: str | Path,
    output_path: str | Path,
    target_voxel_size: float = 2.0,
    use_mask: bool = True,
    b0_threshold: int = 50,
    n_jobs: int | None = None,
    omp_nthreads: int | None = None,
    seed: int = 42,
) -> None:
    """Implements the preprocessing pipeline.

    Args:
        nifti_path: Path to input NIfTI (.nii or .nii.gz)
        bval_path: Path to b-values file
        bvec_path: Path to b-vectors file
        output_path: Path where the preprocessed NIfTI is saved
        target_voxel_size: Desired isotropic voxel size in mm. Defaults to 2.0
        b0_threshold: Maximum b-value considered a b0 image. Defaults to 50
        n_jobs: Number of parallel jobs for motion correction
        omp_nthreads: Number of OpenMP threads for motion correction
        seed: Random seed for reproducibility

    Returns:
        None
    """    
    nifti_path = Path(nifti_path)
    bval_path = Path(bval_path)
    bvec_path = Path(bvec_path)
    output_path = Path(output_path)
    
    # 1) Load data + gradient table using your existing helper
    data, affine, hdr, gtab = load_data(
        nifti_path,
        bval_path=bval_path,
        bvec_path=bvec_path,
        b0_threshold=b0_threshold,
    )
    
    voxel_size = hdr.get_zooms()[:3]
    
    # Default: no change
    data_rs = data
    affine_rs = affine
    new_vox = voxel_size
    
    # 2) Check voxel size and reslice if needed
    if not check_voxel_size(voxel_size):
        print(f"Voxel sizes not isotropic ({voxel_size}). Reslicing...")
        data_rs, affine_rs, new_vox = reslice_data(
            data_rs, affine_rs, voxel_size, target_voxel_size
        )
        print(f"Resliced to {new_vox} mm voxels")
        
        # Save resliced data to temporary file for motion correction
        temp_resliced_path = output_path.parent / f"{output_path.stem}_temp_resliced.nii.gz"
        save_nifti(str(temp_resliced_path), data_rs, affine_rs)
        input_for_mc = temp_resliced_path
    else:
        print(f"Voxel sizes already isotropic: {voxel_size}")
        input_for_mc = nifti_path
    
    # Apply brain mask
    static_mask = None
    if use_mask:
        # Quick brain mask from mean b0
        b0_indices = np.where(gtab.b0s_mask)[0]
        mean_b0 = np.mean(data[..., b0_indices], axis=-1)
        
        # Fast mask using median_otsu
        _, static_mask = median_otsu(mean_b0, median_radius=2, numpass=1)

    mask_path = output_path.parent / "brainmask.nii.gz"
    nib.save(nib.Nifti1Image(static_mask.astype(np.uint8), img.affine), str(mask_path))

    # 3) Motion correction
    print("Starting motion correction...")
    mc_output_path = output_path.parent / f"{output_path.stem}_mc.nii.gz"
    
    motion_correct_data(
        nifti_path=input_for_mc,
        bval_path=bval_path,
        bvec_path=bvec_path,
        out_path=mc_output_path,
        
        n_jobs=n_jobs,
        omp_nthreads=omp_nthreads,
        seed=seed,
    )
    print("Motion correction complete.")
    
    # 4) Load the motion-corrected data for final save
    data_mc, affine_mc, hdr_mc, _ = load_data(
        mc_output_path,
        bval_path=bval_path,
        bvec_path=bvec_path,
        b0_threshold=b0_threshold,
    )
    
    # TODO: Denoise

    # TODO: Brain segmentation

    # TODO: Suppress Gibbs oscillations
    
    # 5) Save final preprocessed data
    save_nifti(str(output_path), data_mc, affine_mc)
    print(f"Preprocessing complete → {output_path}")
    
    # Clean up temporary files if created
    if input_for_mc != nifti_path:
        temp_resliced_path.unlink(missing_ok=True)
    mc_output_path.unlink(missing_ok=True)

