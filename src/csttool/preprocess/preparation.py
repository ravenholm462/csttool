import nibabel as nib
import numpy as np

from pathlib import Path
from dipy.align.reslice import reslice
from dipy.align import motion_correction
from dipy.io.image import save_nifti

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
        data: np.ndarray,
        affine: np.ndarray,
        gtab
) -> tuple[np.ndarray, np.ndarray]:
    """Between volumes motion correction using dipy.align.motion_correct

    Args:
        data (np.ndarray): 4D DWI array (x, y, z, n_volumes)
        affine (np.ndarray): 4x4 affine matrix for the data.
        gtab (_type_): DIPY GradientTable for the dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - data_mc: motion-corrected 4D DWI array.
            - affine_mc: affine matrix for the corrected data.
    """

    data_corrected, reg_affines = motion_correction(data, gtab, affine=affine)

    if isinstance(data_corrected, nib.Nifti1Image):
        data_mc = data_corrected.get_fdata(dtype=np.float32)
        affine_mc = data_corrected.affine
    else:
        data_mc = np.asarray(data_corrected, dtype=np.float32)
        affine_mc = affine

    return data_mc, affine_mc


def process_and_save(
    nifti_path: str | Path,
    output_path: str | Path,
    target_voxel_size: float = 2.0,
    b0_threshold: int = 50,
) -> None:
    """Minimal preprocessing: load, reslice if needed, motion correct, save.

    Args:
        nifti_path (str | Path): Path to input NIfTI (.nii or .nii.gz).
        output_path (str | Path): Path where the preprocessed NIfTI is saved.
        target_voxel_size (float, optional): Desired isotropic voxel size in mm.
            Defaults to 2.0.
        b0_threshold (int, optional): Maximum b-value considered a b0 image when
            building the gradient table. Defaults to 50.

    Returns:
        None
    """
    nifti_path = Path(nifti_path)
    output_path = Path(output_path)

    # 1) Load data + gradient table using your existing helper
    data, affine, hdr, gtab = load_data(
        nifti_path,
        bval_path=None,
        bvec_path=None,
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
    else:
        print(f"Voxel sizes already isotropic: {voxel_size}")

    # 3) Motion correction
    data_mc, affine_mc = motion_correct_data(data_rs, affine_rs, gtab, n_jobs = 4)
    print("Motion correction complete.")

    # TODO: Denoise

    # TODO: Brain segmentation

    # TODO: Suppress Gibbs oscillations

    # 4) Save final preprocessed data (for now: motion-corrected, resliced)
    save_nifti(str(output_path), data_mc, affine_mc)
    print(f"Preprocessing complete → {output_path}")

