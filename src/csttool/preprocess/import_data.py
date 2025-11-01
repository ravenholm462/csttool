#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 18:35:38 2025

@author: alem
"""

import numpy as np
import nibabel as nib

from os.path import expanduser, join

import matplotlib.pyplot as plt

from dipy.core.gradients import gradient_table
from dipy.data import fetch_sherbrooke_3shell
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

import dicom2nifti

def is_dicom_dir(p):
    """
    If data is DICOMS, check if folder exists and if so, if populated by .dcm
    """
    if not p.is_dir():
        return False
    if any(f.suffix.lower() == ".dcm" for f in p.iterdir()):
        return True
    return False
    
def convert_to_nifti(dicom_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom2nifti.convert_directory(dicom_dir, out_dir)

    
