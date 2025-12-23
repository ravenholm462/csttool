"""
Registration module for csttool's pipeline.

Given an FA map as a NIfTI file and a template (MNI 152 used here), the module handles spatial registration between
MNI template space (static image) and subject native space (moving image).
"""

# Imports

import numpy as np
import nibabel as nib
from pathlib import Path

def load_mni_template(contrast="T1"):
    
    from dipy.data import fetch_mni_template, read_mni_template

    print(f"Fetching MNI template (contrast: {contrast})")

    fetch_mni_template()
    template_img = read_mni_template(contrast=contrast)

    template_data = template_img.get_fdata()
    template_affine = template_img.affine


    print(f"MNI template loaded: shape {template_data.shape}")

    return template_img, template_data, template_affine


def compute_affine_registration():

    return


def compute_syn_registration():

    return


def register_mni_to_subject():

    return


def save_registration_report():

    return


def visualize():

    return


