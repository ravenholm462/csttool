"""
load_dataset.py

Load dataset from DICOM directory or NIfTI file and build gradient table.

"""

import os
from pathlib import Path

import dicom2nifti

from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti

def load_dataset(dir_path: str, fname: str):
    """
    Load dataset from DICOM directory or NIfTI file and build gradient table.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the dataset.
    fname : str
        Name of the file to load.
    
    Returns
    -------
    nii : Nifti1Image
        NIfTI image.
    bval : str
        Path to the bval file.
    bvec : str
        Path to the bvec file.
    gtab : GradientTable
        Gradient table.
    """
    # Check if the directory exists
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist")

    # Check if DICOM directory
    if any(f.suffix == ".dcm" for f in dir_path.iterdir()):
        print(f"DICOM directory detected: {dir_path}")
        # Convert DICOM to NIfTI
        # Save NIfTI, bval and bvec files to a directory called nifti one level up from dir_path
        print(f"Converting DICOM to NIfTI...")
        nifti_dir = dir_path.parent / "nifti"
        nifti_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        result = dicom2nifti.dicom_series_to_nifti(
            str(dir_path),
            str(nifti_dir / (fname + ".nii.gz")),
            reorient_nifti=True
        )
        nii = load_nifti(result["NII_FILE"])[0]
        bval_path = result.get("BVAL_FILE")
        bvec_path = result.get("BVEC_FILE") 
    else:
        print(f"NIfTI directory detected: {dir_path}")
        nifti_dir = dir_path
        nii_path = os.path.join(dir_path, fname + ".nii.gz")
        
        # Try .bval first, then .bvals for gradient files
        bval_path = os.path.join(dir_path, fname + ".bval")
        if not os.path.exists(bval_path):
            bval_path = os.path.join(dir_path, fname + ".bvals")
        
        bvec_path = os.path.join(dir_path, fname + ".bvec")
        if not os.path.exists(bvec_path):
            bvec_path = os.path.join(dir_path, fname + ".bvecs")
        
        nii = load_nifti(nii_path)[0]

    # Read bvalues and bvectors, build a gradient table
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    num_of_gradients = len(gtab)

    print(gtab.info)
    print(f"Number of gradients: {num_of_gradients}")
    print("\n" + "=" * 70 + "\n")
  
    return nii, gtab, nifti_dir