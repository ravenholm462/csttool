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

# use dipy.viz.regtools.overlay_slices to show comparison before and after registration
# https://docs.dipy.org/1.0.0/examples_built/affine_registration_3d.html

def compute_affine_registration(
    static_image,
    static_affine,
    moving_image,
    moving_affine,
    nbins=32,
    sampling_prop=None,
    level_iters=[10000, 1000, 100],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1],
    verbose=True
):
    from dipy.align.imaffine import (
        transform_centers_of_mass,
        MutualInformationMetric,
        AffineRegistration
    )
    from dipy.align.transforms import (
        TranslationTransform3D,
        RigidTransform3D,
        AffineTransform3D
    )

    if verbose:
        print("Computing affine registration...")
        print(f"    Static (subject): {static_image.shape}")
        print(f"    Moving (template): {moving_image.shape}")

    # Stage 0: Center of mass alignment
    if verbose:
        print("    Stage 0/3: Aligning centers of mass...")
    c_of_mass = transform_centers_of_mass(
        static_image, static_affine,
        moving_image, moving_affine
    )

    # Setup registration
    if verbose:
        print("    Setting up affine registration...")
        print(f"        Level iters: {level_iters}")
        print(f"        Sigmas: {sigmas}")
        print(f"        Factors: {factors}")

    metric = MutualInformationMetric(nbins=nbins, sampling_proportion=sampling_prop)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors
    )

    # Stage 1: Translation
    if verbose:
        print("    Stage 1/3: Translation...")
    translation = affreg.optimize(
        static_image, moving_image,
        TranslationTransform3D(), None,
        static_affine, moving_affine,
        starting_affine=c_of_mass.affine
    )

    # Stage 2: Rigid
    if verbose:
        print("    Stage 2/3: Rigid...")
    rigid = affreg.optimize(
        static_image, moving_image,
        RigidTransform3D(), None,
        static_affine, moving_affine,
        starting_affine=translation.affine
    )

    # Stage 3: Full affine
    if verbose:
        print("    Stage 3/3: Affine...")
    affine = affreg.optimize(
        static_image, moving_image,
        AffineTransform3D(), None,
        static_affine, moving_affine,
        starting_affine=rigid.affine
    )

    if verbose:
        print("✓ Affine registration complete")

    return affine


def compute_syn_registration():

    return


def register_mni_to_subject():

    return


def save_registration_report():

    return


def visualize():

    return


