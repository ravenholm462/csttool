from pathlib import Path
from csttool.preprocess.import_data import is_dicom_dir, convert_to_nifti, load_data

dicom_dir = Path("/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017")
out_dir = Path("/home/alemnalo/anom/nifti")

nii, bval, bvec = convert_to_nifti(dicom_dir, out_dir)
data, affine, hdr, gtab = load_data(nii, bval, bvec)

print("Data shape:", data.shape)
print("Gradients:", len(gtab.bvals))
print(is_dicom_dir(dicom_dir))
