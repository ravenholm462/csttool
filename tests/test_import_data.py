from pathlib import Path
from csttool.preprocess.import_data import is_dicom_dir, convert_to_nifti, load_data
from csttool.preprocess.preparation import process_and_save

# Define paths
dicom_dir = Path("/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017")
out_dir = Path("/home/alemnalo/anom/nifti")
preproc_dir = Path("/home/alemnalo/anom/preproc")

# Create output directories if they don't exist
out_dir.mkdir(parents=True, exist_ok=True)
preproc_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Check if it's a DICOM directory
print(f"Is DICOM directory: {is_dicom_dir(dicom_dir)}")

# Step 2: Convert DICOM to NIfTI
print("\nConverting DICOM to NIfTI...")
nii, bval, bvec = convert_to_nifti(dicom_dir, out_dir)
print(f"NIfTI file: {nii}")
print(f"Bval file: {bval}")
print(f"Bvec file: {bvec}")

# Step 3: Load and inspect the data
print("\nLoading data...")
data, affine, hdr, gtab = load_data(nii, bval, bvec)
print(f"Data shape: {data.shape}")
print(f"Number of gradients: {len(gtab.bvals)}")
print(f"Voxel size: {hdr.get_zooms()[:3]}")
print(f"B-values range: {gtab.bvals.min():.0f} - {gtab.bvals.max():.0f}")

# Step 4: Preprocess the data
print("\n" + "="*60)
print("Starting preprocessing pipeline...")
print("="*60)

output_path = preproc_dir / "17_cmrr_mbep2d_diff_ap_tdi_preproc.nii.gz"

process_and_save(
    nifti_path=nii,
    bval_path=bval,
    bvec_path=bvec,
    output_path=output_path,
    target_voxel_size=2.0,
    b0_threshold=50,
    n_jobs=None,
    omp_nthreads=None,
    seed=42,
)

print("\n" + "="*60)
print(f"Preprocessing complete!")
print(f"Output saved to: {output_path}")
print("="*60)

# Step 5: Verify the output
if output_path.exists():
    print("\nVerifying preprocessed data...")
    data_preproc, affine_preproc, hdr_preproc, gtab_preproc = load_data(
        output_path, bval, bvec
    )
    print(f"Preprocessed data shape: {data_preproc.shape}")
    print(f"Preprocessed voxel size: {hdr_preproc.get_zooms()[:3]}")
else:
    print("\nWarning: Output file was not created!")