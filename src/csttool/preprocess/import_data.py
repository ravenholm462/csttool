#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 18:35:38 2025

@author: alem
"""

import numpy as np
import nibabel as nib

from pathlib import Path

import matplotlib.pyplot as plt

from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

import dicom2nifti

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
        dicom_dir (Path): Path to the input DICOM directory.
        out_dir (Path): Path to the output directory where NIfTI and sidecar files will be saved.

    Returns:
        tuple[Path, Path, Path]: Paths to the generated .nii(.gz), .bval, and .bvec files.
    """
  
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom2nifti.convert_directory(dicom_dir, out_dir, reorient=True)

    out_dir = Path(out_dir)
    nii = next(iter(list(out_dir.glob("*.nii.gz")) + list(out_dir.glob("*.nii"))))
    stem = nii.name.replace(".nii.gz", "").replace(".nii", "")
    bval = out_dir / f"{stem}.bval"
    bvec = out_dir / f"{stem}.bvec"
    
    return nii, bval, bvec

def load_data(nifti_path, bval_path=None, bvec_path=None, b0_threshold=50):
    """Load NIfTI data and build a diffusion gradient table.

    If no .bval or .bvec paths are provided, they are assumed to be located
    next to the NIfTI file with the same filename stem.

    Args:
        nifti_path (Path): Path to the .nii or .nii.gz file.
        bval_path (Path, optional): Path to the .bval file. Defaults to None.
        bvec_path (Path, optional): Path to the .bvec file. Defaults to None.
        b0_threshold (int, optional): Maximum b-value considered a b0 image. Defaults to 50.

    Returns:
        tuple[np.ndarray, np.ndarray, nib.Nifti1Header, dipy.core.gradients.GradientTable]:
            - data: 4D diffusion image array (x, y, z, n_volumes)
            - affine: 4x4 affine matrix
            - hdr: NIfTI header
            - gtab: DIPY GradientTable object
    """

    p = Path(nifti_path)

    if bval_path is None or bvec_path is None:
        stem = p.name.replace(".nii.gz", "").replace(".nii", "")
        bval_path = bval_path or p.with_name(f"{stem}.bval")
        bvec_path = bvec_path or p.with_name(f"{stem}.bvec")

    img = nib.load(str(p))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    hdr = img.header

    gtab = None
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    return data, affine, hdr, gtab



    
