
import argparse
from pathlib import Path
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

def cmd_extract(args: argparse.Namespace) -> dict | None:
    """
    Extract bilateral CST using atlas-based ROI filtering.
    
    Supports two methods:
    - endpoint: Filter by streamline endpoints (original)
    - passthrough: Filter by streamlines passing through ROIs (more permissive)
    
    Note: roi-seeded method requires raw DWI data and is only available via cmd_run.
    """
    
    verbose = getattr(args, 'verbose', True)
    
    # Check extraction method
    extraction_method = getattr(args, 'extraction_method', 'passthrough')
    
    if extraction_method == "roi-seeded":
        print("Error: roi-seeded method requires raw DWI data.")
        print("       Use 'csttool run' for roi-seeded extraction, or")
        print("       use --extraction-method endpoint|passthrough with cmd_extract.")
        return None
    
    # Validate inputs
    if not args.tractogram.exists():
        print(f"Error: Tractogram not found: {args.tractogram}")
        return None
    
    if not args.fa.exists():
        print(f"Error: FA map not found: {args.fa}")
        return None
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Import extraction modules
    try:
        from csttool.extract.modules.registration import register_mni_to_subject
        from csttool.extract.modules.warp_atlas_to_subject import (
            warp_harvard_oxford_to_subject,
            CST_ROI_CONFIG
        )
        from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
        from csttool.extract.modules.endpoint_filtering import (
            extract_bilateral_cst,
            save_cst_tractograms,
            save_extraction_report
        )
        from csttool.extract.modules.passthrough_filtering import extract_cst_passthrough
    except ImportError as e:
        print(f"Error importing extraction modules: {e}")
        return None
    
    # Load inputs
    print(f"Loading tractogram: {args.tractogram}")
    try:
        sft = load_tractogram(str(args.tractogram), 'same')
        streamlines = sft.streamlines
        print(f"  Loaded {len(streamlines):,} streamlines")
    except Exception as e:
        print(f"Error loading tractogram: {e}")
        return None
    
    print(f"Loading FA map: {args.fa}")
    try:
        fa_data, fa_affine = load_nifti(str(args.fa))
    except Exception as e:
        print(f"Error loading FA map: {e}")
        return None
    
    # Step 1: Registration
    print("\n" + "="*60)
    print("Step 1: Registering MNI template to subject space")
    print("="*60)
    
    level_iters_affine = [1000, 100, 10] if args.fast_registration else [10000, 1000, 100]
    level_iters_syn = [5, 5, 3] if args.fast_registration else [10, 10, 5]
    
    try:
        reg_result = register_mni_to_subject(
            subject_fa_path=args.fa,
            output_dir=args.out,
            level_iters_affine=level_iters_affine,
            level_iters_syn=level_iters_syn,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during registration: {e}")
        return None
    
    # Step 2: Warp atlases
    print("\n" + "="*60)
    print("Step 2: Warping Harvard-Oxford atlases to subject space")
    print("="*60)
    
    try:
        warped = warp_harvard_oxford_to_subject(
            registration_result=reg_result,
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error warping atlases: {e}")
        return None
    
    # Step 3: Create ROI masks
    print("\n" + "="*60)
    print("Step 3: Creating CST ROI masks")
    print("="*60)
    
    try:
        masks = create_cst_roi_masks(
            warped_cortical=warped['cortical_warped'],
            warped_subcortical=warped['subcortical_warped'],
            subject_affine=warped['subject_affine'],
            roi_config=CST_ROI_CONFIG,
            dilate_brainstem=args.dilate_brainstem,
            dilate_motor=args.dilate_motor,
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose,
            original_subject_affine=warped.get('original_subject_affine'),
            reorientation_transform=warped.get('reorientation_transform')
        )
    except Exception as e:
        print(f"Error creating ROI masks: {e}")
        return None
    
    # Step 4: Extract CST
    print("\n" + "="*60)
    print(f"Step 4: Extracting bilateral CST (method: {extraction_method})")
    print("="*60)
    
    try:
        if extraction_method == "passthrough":
            cst_result = extract_cst_passthrough(
                streamlines=streamlines,
                masks=masks,
                affine=warped['subject_affine'],  # Use affine that matches the ROI masks
                min_length=args.min_length,
                max_length=args.max_length,
                verbose=verbose
            )
        else:  # "endpoint"
            cst_result = extract_bilateral_cst(
                streamlines=streamlines,
                masks=masks,
                affine=warped['subject_affine'],  # Use affine that matches the ROI masks
                min_length=args.min_length,
                max_length=args.max_length,
                verbose=verbose
            )
    except Exception as e:
        print(f"Error during CST extraction: {e}")
        return None
    
    # Step 5: Save outputs
    print("\n" + "="*60)
    print("Step 5: Saving extracted tractograms")
    print("="*60)
    
    try:
        reference_img = nib.load(str(args.fa))
        output_paths = save_cst_tractograms(
            cst_result=cst_result,
            reference_img=reference_img,
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose
        )
        
        save_extraction_report(cst_result, output_paths, args.out, args.subject_id)
    except Exception as e:
        print(f"Error saving outputs: {e}")
        return None
    
    if getattr(args, 'save_visualizations', False):
        from csttool.extract.modules import save_all_extraction_visualizations
        save_all_extraction_visualizations(
            cst_result=cst_result,
            fa=fa_data,
            masks=masks,
            affine=warped['subject_affine'],
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Subject: {args.subject_id}")
    print(f"Left CST:  {cst_result['stats']['cst_left_count']:,} streamlines")
    print(f"Right CST: {cst_result['stats']['cst_right_count']:,} streamlines")
    print(f"Total:     {cst_result['stats']['cst_total_count']:,} streamlines")
    print(f"Extraction rate: {cst_result['stats']['extraction_rate']:.2f}%")
    print(f"{'='*60}")
    
    return {
        'cst_left_path': output_paths.get('cst_left'),
        'cst_right_path': output_paths.get('cst_right'),
        'cst_combined_path': output_paths.get('cst_combined'),
        'stats': cst_result['stats']
    }


def run_roi_seeded_extraction(
    preproc_path: Path,
    fa_path: Path,
    output_dir: Path,
    subject_id: str,
    args: argparse.Namespace,
    verbose: bool = True
) -> dict | None:
    """
    Run ROI-seeded CST extraction.
    
    This method seeds streamlines directly from motor cortex ROIs
    and filters by brainstem traversal. Requires raw DWI data.
    """
    from csttool.extract.modules.registration import register_mni_to_subject
    from csttool.extract.modules.warp_atlas_to_subject import (
        warp_harvard_oxford_to_subject,
        CST_ROI_CONFIG
    )
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
    from csttool.extract.modules.roi_seeded_tracking import extract_cst_roi_seeded
    from csttool.extract.modules.endpoint_filtering import (
        save_cst_tractograms,
        save_extraction_report
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preprocessed DWI data
    if verbose:
        print(f"Loading preprocessed data: {preproc_path}")
    
    dwi_img = nib.load(str(preproc_path))
    data = dwi_img.get_fdata()
    affine = dwi_img.affine
    
    # Load gradient information
    bval_path = preproc_path.with_suffix('').with_suffix('.bval')
    bvec_path = preproc_path.with_suffix('').with_suffix('.bvec')
    
    # Handle .nii.gz extension
    if not bval_path.exists():
        stem = preproc_path.name.replace('.nii.gz', '').replace('.nii', '')
        bval_path = preproc_path.parent / f"{stem}.bval"
        bvec_path = preproc_path.parent / f"{stem}.bvec"
    
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs)
    
    # Load FA map
    fa_img = nib.load(str(fa_path))
    fa_data = fa_img.get_fdata()
    fa_affine = fa_img.affine
    
    # Create brain mask from FA
    brain_mask = fa_data > 0
    
    # Step 1: Registration
    if verbose:
        print("\n" + "="*60)
        print("Step 1: Registering MNI template to subject space")
        print("="*60)
    
    level_iters_affine = [1000, 100, 10] if getattr(args, 'fast_registration', False) else [10000, 1000, 100]
    level_iters_syn = [5, 5, 3] if getattr(args, 'fast_registration', False) else [10, 10, 5]
    
    reg_result = register_mni_to_subject(
        subject_fa_path=fa_path,
        output_dir=output_dir,
        level_iters_affine=level_iters_affine,
        level_iters_syn=level_iters_syn,
        verbose=verbose
    )
    
    # Step 2: Warp atlases
    if verbose:
        print("\n" + "="*60)
        print("Step 2: Warping Harvard-Oxford atlases to subject space")
        print("="*60)
    
    warped = warp_harvard_oxford_to_subject(
        registration_result=reg_result,
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose
    )
    
    # Step 3: Create ROI masks
    if verbose:
        print("\n" + "="*60)
        print("Step 3: Creating CST ROI masks")
        print("="*60)
    
    masks = create_cst_roi_masks(
        warped_cortical=warped['cortical_warped'],
        warped_subcortical=warped['subcortical_warped'],
        subject_affine=fa_affine,
        roi_config=CST_ROI_CONFIG,
        dilate_brainstem=getattr(args, 'dilate_brainstem', 2),
        dilate_motor=getattr(args, 'dilate_motor', 1),
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose,
        original_subject_affine=warped.get('original_subject_affine'),
        reorientation_transform=warped.get('reorientation_transform')
    )
    
    # Step 4: ROI-seeded extraction
    if verbose:
        print("\n" + "="*60)
        print("Step 4: ROI-seeded CST extraction")
        print("="*60)
    
    cst_result = extract_cst_roi_seeded(
        data=data,
        gtab=gtab,
        affine=affine,
        brain_mask=brain_mask,
        motor_left_mask=masks['motor_left'],
        motor_right_mask=masks['motor_right'],
        brainstem_mask=masks['brainstem'],
        fa_map=fa_data,
        fa_threshold=getattr(args, 'seed_fa_threshold', 0.15),
        seed_density=getattr(args, 'seed_density', 2),
        step_size=0.5,
        min_length=getattr(args, 'min_length', 30.0),
        max_length=getattr(args, 'max_length', 200.0),
        verbose=verbose
    )
    
    # Step 5: Save outputs
    if verbose:
        print("\n" + "="*60)
        print("Step 5: Saving extracted tractograms")
        print("="*60)
    
    output_paths = save_cst_tractograms(
        cst_result=cst_result,
        reference_img=fa_img,
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose
    )
    
    save_extraction_report(cst_result, output_paths, output_dir, subject_id)
    
    # Visualizations
    if getattr(args, 'save_visualizations', False):
        from csttool.extract.modules import save_all_extraction_visualizations
        save_all_extraction_visualizations(
            cst_result=cst_result,
            fa=fa_data,
            masks=masks,
            affine=fa_affine,
            output_dir=output_dir,
            subject_id=subject_id,
            verbose=verbose
        )
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("ROI-SEEDED EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Subject: {subject_id}")
        print(f"Left CST:  {cst_result['stats']['cst_left_count']:,} streamlines")
        print(f"Right CST: {cst_result['stats']['cst_right_count']:,} streamlines")
        print(f"Total:     {cst_result['stats']['cst_total_count']:,} streamlines")
        print(f"{'='*60}")
    
    return {
        'cst_left_path': output_paths.get('cst_left'),
        'cst_right_path': output_paths.get('cst_right'),
        'cst_combined_path': output_paths.get('cst_combined'),
        'stats': cst_result['stats']
    }
