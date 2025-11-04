from pathlib import Path
from csttool.preprocess.import_data import is_dicom_dir, convert_to_nifti, load_data
from csttool.preprocess.preparation import process_and_save

dicom_dir = Path("/home/alem/Documents/thesis/data/anom/cmrr_mbep2d_diff_AP_TDI_Series0017/")
out_dir = Path("/home/alem/Documents/thesis/data/nifti/")
preproc_dir = Path("/home/alem/Documents/thesis/data/preproc/")

out_dir.mkdir(parents=True, exist_ok=True)
preproc_dir.mkdir(parents=True, exist_ok=True)

print(f"Is DICOM directory: {is_dicom_dir(dicom_dir)}")

print("\nConverting DICOM to NIfTI...")
nii, bval, bvec = convert_to_nifti(dicom_dir, out_dir)

print("\nStarting preprocessing pipeline...")
out_nii = preproc_dir / (nii.stem + "_preproc.nii.gz")
process_and_save(nii, bval, bvec, out_nii)