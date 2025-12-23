def load_and_mask(nii_dirname, nii_fname, visualize=False, verbose=False):
    """This function loads a NIfTI dataset with its gradient table, then applies median Otsu threshold segmentation to generate a brainmask.

    Args:
        nii_dirname (str): Directory of your NIfTI file.
        nii_fname (str): Name of your NIfTI file.
        visualize (bool): Set to True for data visualization. Defaults to False.
        verbose (bool): Set to True for verbose output. Defaults to False.

    Returns:
        tuple: (data, affine, img, gtab, masked_data, brain_mask)
            - data: 4D DWI array (X, Y, Z, N)
            - affine: 4x4 transformation matrix
            - img: nibabel Nifti1Image object
            - gtab: DIPY GradientTable
            - masked_data: Brain-masked DWI data
            - brain_mask: Binary brain mask array
    """    
    from csttool.preprocess.funcs import load_dataset, background_segmentation

    data, affine, img, gtab = load_dataset(
    nifti_path=nii_dirname,
    fname=nii_fname,
    visualize=visualize
    )

    if verbose:
        print(f"Loaded dataset: {nii_dirname + nii_fname}")
        print(f"Data shape: {data.shape}")
        print(f"Affine:\n{affine}")
        print(f"Gradient table: {len(gtab.bvals)} volumes")

    masked_data, brain_mask = background_segmentation(
    data,
    gtab,
    visualize=visualize
    )

    if verbose:
        print("Brain mask generated.")
        mask_coverage = brain_mask.sum() / brain_mask.size * 100
        print(f"Brain mask: {brain_mask.sum():,} voxels ({mask_coverage:.1f}%)")

    return data, affine, img, gtab, masked_data, brain_mask