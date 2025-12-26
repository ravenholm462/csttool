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
    num_iters=[10000, 1000, 100],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1],
    verbose=True
):

    from dipy.align.imaffine import (transform_centers_of_mass,
                                    AffineMap,
                                    MutualInformationMetric,
                                    AffineRegistration)

    from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

    if verbose:
        print(f"Computing affine registration...")
        print(f"    Static (subject): {moving_image.shape}")
        print(f"    Moving: {static_image.shape}")

    if verbose:
        print("    Aligning centers of mass...")

    centers_of_mass = transform_centers_of_mass(
        static_image,
        static_affine,
        moving_image,
        moving_affine
    )

    if verbose:
        print("    Center of mass alignment complete")
        print(f"   Commencing affine registration...")
        print(f"   Num of iters (coarse, medium, fine): {num_iters}")
        print(f"   Sigmas: {sigmas}")
        print(f"   Factors: {factors}")
    
    metric = MutualInformationMetric(nbins=nbins, sampling_proportion=sampling_prop)

    affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = center_of_mass.affine

    translation = affreg.optimize(
        static_image,
        moving_image,
        transform,
        params0,
        static_affine,
        moving_affine,
        starting_affine
    )

    transformed = translation.transform(moving_image)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(
        static_image, 
        moving_image, 
        transform, 
        params0,
        static_affine, 
        moving_affine,
        starting_affine=starting_affine
    )

    transformed = rigid.transform(moving_image)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(
        static_image, 
        moving_image, 
        transform, 
        params0,
        static_affine, 
        moving_affine,
        starting_affine=starting_affine
    )

    transformed = affine.transform(moving_image)
    
    return transformed


def compute_syn_registration():

    return


def register_mni_to_subject():

    return


def save_registration_report():

    return


def visualize():

    return


