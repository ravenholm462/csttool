import nibabel as nib
import numpy as np
from pathlib import Path

from dipy.align.reslice import reslice
from dipy.align import motion_correction        # << re-enable this
from dipy.io.image import save_nifti
from dipy.segment.mask import median_otsu       # << keep for masking

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

    # Save mask as NIfTI (float32 or uint8 – here float32 like DIPY examples)
    save_nifti(str(mask_path), mask.astype(np.float32), affine)
    print(f"Brain mask saved → {mask_path}")

    return mask

def motion_correct_data(
    data: np.ndarray,
    affine: np.ndarray,
    gtab,
    static_mask: np.ndarray | None = None,
    b0_ref: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Between-volumes motion correction using DIPY.

    Args:
        data: 4D DWI array (x, y, z, n_volumes).
        affine: 4x4 affine matrix.
        gtab: DIPY GradientTable.
        static_mask: 3D brain mask (x, y, z). If provided, used as static_mask.
        b0_ref: Index of b0 volume to use as reference. Defaults to 0.

    Returns:
        data_mc: motion-corrected 4D DWI array.
        affine_mc: affine of the corrected image.
    """
    if static_mask is not None:
        # Ensure mask is 3D and matches the spatial shape of data
        if static_mask.ndim != 3:
            raise ValueError(
                f"static_mask must be 3D, got shape {static_mask.shape}"
            )
        if static_mask.shape != data.shape[:3]:
            raise ValueError(
                f"static_mask shape {static_mask.shape} does not match "
                f"data spatial shape {data.shape[:3]}"
            )
        # DIPY's affine map code expects float arrays, not bool
        static_mask = static_mask.astype(np.float32)

    print("Running DIPY motion correction...")
    mc_img, affines = motion_correction(
        data,
        gtab,
        affine=affine,
        b0_ref=b0_ref,
        pipeline=['center_of_mass', 'translation'],
        static_mask=static_mask,
    )

    data_mc = mc_img.get_fdata(dtype=np.float32)
    affine_mc = mc_img.affine
    print("Motion correction done.")
    return data_mc, affine_mc

def process_and_save(
    nifti_path: str | Path,
    bval_path: str | Path,
    bvec_path: str | Path,
    output_path: str | Path,
    target_voxel_size: float = 2.0,
    b0_threshold: int = 50,
) -> None:
    """Preprocessing pipeline: load → reslice → skull strip → motion correct → save."""

    nifti_path = Path(nifti_path)
    bval_path = Path(bval_path)
    bvec_path = Path(bvec_path)
    output_path = Path(output_path)

    # 1) Load data + gradient table
    data, affine, hdr, gtab = load_data(
        nifti_path,
        bval_path=bval_path,
        bvec_path=bvec_path,
        b0_threshold=b0_threshold,
    )
    voxel_size = hdr.get_zooms()[:3]
    print(f"Loaded data: shape={data.shape}, voxel_size={voxel_size}, "
          f"n_gradients={len(gtab.bvals)}")

    # 2) Reslice if needed
    data_rs, affine_rs = data, affine
    if not check_voxel_size(voxel_size):
        print(f"Voxel sizes not isotropic ({voxel_size}). Reslicing...")
        data_rs, affine_rs, new_vox = reslice_data(
            data, affine, voxel_size, target_voxel_size
        )
        print(f"Resliced to {new_vox} mm voxels.")
    else:
        print("Voxel sizes already isotropic.")

    # 3) Brain segmentation / skull stripping (on resliced data)
    mask_path = output_path.with_name(output_path.stem + "_mask.nii.gz")
    print("Computing brain mask...")
    brain_mask = compute_brain_mask(data_rs, affine_rs, gtab, mask_path)

    # 4) Motion correction using that mask
    data_mc, affine_mc = motion_correct_data(
        data_rs,
        affine_rs,
        gtab,
        static_mask=brain_mask,
    )


    # TODO: Denoise

    # TODO: Brain segmentation refinement (if needed)

    # TODO: Suppress Gibbs oscillations

    # 5) Save final preprocessed data
    save_nifti(str(output_path), data_mc, affine_mc)
    print(f"Preprocessing complete → {output_path}")
