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
    from csttool.preprocess.modules.load_dataset import load_dataset
    from csttool.preprocess.modules.background_segmentation import background_segmentation

    # Remove extension if present in fname, as modules.load_dataset expects stem or handles it differently
    # Actually modules.load_dataset expects fname without extension for nifti construction in some paths,
    # but let's check how it uses it.
    # It constructs: os.path.join(dir_path, fname + ".nii.gz")
    # So we should pass fname WITHOUT extension if it's not there.
    # The original caller likely passed it without extension based on usage in cli.py or tracking.
    
    # modules.load_dataset returns: nii, gtab, nifti_dir, metadata
    nii, gtab, _, _ = load_dataset(
        dir_path=nii_dirname,
        fname=nii_fname
    )
    
    data = nii.get_fdata()
    affine = nii.affine
    img = nii

    if verbose:
        print(f"  → Loaded dataset: {nii_dirname}/{nii_fname}")
        print(f"    • Data shape: {data.shape}")
        print(f"    • Gradient table: {len(gtab.bvals)} volumes")

    # background_segmentation(data, gtab=None, median_radius=2, numpass=1, autocrop=False)
    # Original used visualize=visualize, but new one doesn't support visualize arg directly in computation 
    # (it seems visualizion is handled separately or removed).
    # We will ignore visualize arg for segmentation as it's not in the new signature.
    masked_data, brain_mask = background_segmentation(
        data,
        gtab
    )

    if verbose:
        mask_coverage = brain_mask.sum() / brain_mask.size * 100
        print(f"    • Brain mask: {brain_mask.sum():,} voxels ({mask_coverage:.1f}%)")

    return data, affine, img, gtab, masked_data, brain_mask