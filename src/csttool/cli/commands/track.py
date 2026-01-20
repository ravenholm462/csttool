
import argparse
from pathlib import Path

from dipy.io.image import load_nifti
from ..utils import extract_stem_from_filename, get_gtab_for_preproc

from csttool.tracking.modules import (
    fit_tensors,
    estimate_directions,
    validate_sh_order,
    seed_and_stop,
    run_tractography,
    save_tracking_outputs,
)

def cmd_track(args: argparse.Namespace) -> dict | None:
    """
    Run whole-brain deterministic tractography on preprocessed data.
    """
    preproc_nii = args.nifti
    verbose = getattr(args, 'verbose', False)
    
    if not preproc_nii.exists():
        print(f"Error: preprocessed NIfTI not found: {preproc_nii}")
        return None

    args.out.mkdir(parents=True, exist_ok=True)

    # Determine subject ID / stem
    if args.subject_id:
        stem = args.subject_id
    else:
        stem = extract_stem_from_filename(str(preproc_nii))
    
    print(f"Subject ID: {stem}")
    print(f"Loading preprocessed data from {preproc_nii}")
    
    try:
        data, affine, img = load_nifti(str(preproc_nii), return_img=True)
    except Exception as e:
        print(f"Error loading NIfTI: {e}")
        return None

    print("Building gradient table for preprocessed data")
    try:
        gtab = get_gtab_for_preproc(preproc_nii)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Step 1: Brain masking
    print("\nStep 1: Brain masking with median Otsu")

    from csttool.preprocess.modules.background_segmentation import background_segmentation
    masked_data, brain_mask = background_segmentation(
        data,
        gtab
    )

    # Step 2: Tensor fitting
    print("\nStep 2: Tensor fit and scalar measures (FA, MD, RD, AD)")
    try:
        tenfit, fa, md, rd, ad, white_matter = fit_tensors(
            masked_data, 
            gtab, 
            brain_mask,
            fa_thresh=args.fa_thr,
            visualize=getattr(args, 'show_plots', False),
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during tensor fitting: {e}")
        return None

    # Step 3: Direction field estimation
    print("\nStep 3: Direction field estimation (CSA ODF model)")
    try:
        csapeaks = estimate_directions(
            masked_data,
            gtab,
            white_matter,
            sh_order=args.sh_order,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during direction estimation: {e}")
        return None

    # Step 4: Stopping criterion and seeds
    print("\nStep 4: Stopping criterion and seed generation")
    try:
        seeds, stopping_criterion = seed_and_stop(
            fa,
            affine,
            white_matter=white_matter,
            fa_thresh=args.fa_thr,
            density=args.seed_density,
            use_binary=False,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during seed generation: {e}")
        return None

    # Step 5: Deterministic tracking
    print("\nStep 5: Deterministic tracking")
    try:
        streamlines = run_tractography(
            csapeaks,
            stopping_criterion,
            seeds,
            affine,
            step_size=args.step_size,
            verbose=verbose,
            visualize=getattr(args, 'show_plots', False)
        )
    except Exception as e:
        print(f"Error during tractography: {e}")
        return None

    # Step 6: Save outputs
    print("\nStep 6: Saving tractogram, scalar maps, and report")
    
    # Get the validated SH order (may have been reduced due to insufficient directions)
    validated_sh_order = validate_sh_order(gtab, args.sh_order, verbose=False)
    
    tracking_params = {
        'step_size': args.step_size,
        'fa_thresh': args.fa_thr,
        'seed_density': args.seed_density,
        'sh_order': validated_sh_order,
        'sphere': 'symmetric362',
        'stopping_criterion': 'fa_threshold',
        'relative_peak_threshold': 0.8,
        'min_separation_angle': 45,
    }
    
    try:
        outputs = save_tracking_outputs(
            streamlines,
            img,
            fa,
            md,
            affine,
            out_dir=args.out,
            stem=stem,
            rd=rd,
            ad=ad,
            tracking_params=tracking_params,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error saving outputs: {e}")
        return None
    
    # After save_tracking_outputs() and before the summary print
    if getattr(args, 'save_visualizations', False):
        from csttool.tracking.modules import save_all_tracking_visualizations
        save_all_tracking_visualizations(
            streamlines=streamlines,
            fa=fa,
            md=md,
            white_matter=white_matter,
            brain_mask=brain_mask,
            seeds=seeds,
            affine=affine,
            output_dir=args.out,
            stem=stem,
            tenfit=tenfit,
            fa_thresh=args.fa_thr,
            tracking_params=tracking_params,
            verbose=verbose
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"TRACKING COMPLETE - {stem}")
    print(f"{'='*60}")
    print(f"Whole-brain streamlines: {len(streamlines):,}")
    print(f"\nOutputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    print(f"{'='*60}")
    
    return {
        'tractogram_path': outputs['tractogram'],
        'fa_path': outputs['fa_map'],
        'md_path': outputs['md_map'],
        'rd_path': outputs.get('rd_map'),  # May be None if not computed
        'ad_path': outputs.get('ad_map'),  # May be None if not computed
        'n_streamlines': len(streamlines),
        'stem': stem
    }
