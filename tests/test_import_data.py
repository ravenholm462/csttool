from pathlib import Path
from csttool.preprocess.import_data import is_dicom_dir, convert_to_nifti
from csttool.preprocess.preparation import process_and_save


def main():
    # debian
    dicom_dir = Path("/home/alem/Documents/thesis/data/anom/cmrr_mbep2d_diff_AP_TDI_Series0017/")
    out_dir = Path("/home/alem/Documents/thesis/data/nifti/")
    preproc_dir = Path("/home/alem/Documents/thesis/data/preproc/")

    # ubuntu
    # dicom_dir = Path("/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017")
    # out_dir = Path("/home/alemnalo/anom/outtest")
    # preproc_dir = Path("/home/alemnalo/anom/preproc")

    out_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Is DICOM directory: {is_dicom_dir(dicom_dir)}")

    print("\nConverting DICOM to NIfTI...")
    nii, bval, bvec = convert_to_nifti(dicom_dir, out_dir)

    print("\nStarting preprocessing pipeline...")
    out_nii = preproc_dir / (nii.stem + "_preproc.nii.gz")
    process_and_save(
        nifti_path=nii,
        bval_path=bval,
        bvec_path=bvec,
        output_path=out_nii,  # (see note below)
        target_voxel_size=2.0,
        num_coils=1,
        denoise=True,
        suppress_gibbs_artifacts=True,
        gibbs_slice_axis=2,
        save_intermediate=True,
    )


if __name__ == "__main__":
    main()
