"""
cli.py

Command-line interface for csttool - CST assessment using DTI data.

Commands:
    check       - Run environment checks
    import      - Import DICOMs or load NIfTI data
    preprocess  - Run preprocessing pipeline
    track       - Run whole-brain tractography
    extract     - Extract bilateral CST from tractogram
    metrics     - Compute CST metrics and generate reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
import csttool.preprocess.funcs as preproc
from csttool.tracking.modules import (
    fit_tensors,
    estimate_directions,
    seed_and_stop,
    run_tractography,
    save_tracking_outputs,
)

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti


def main() -> None:
    """Entrypoint for the csttool CLI."""
    parser = argparse.ArgumentParser(
        prog="csttool",
        description="CST assessment tool using DTI data."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command")

    # -------------------------------------------------------------------------
    # check subtool
    # -------------------------------------------------------------------------
    p_check = subparsers.add_parser("check", help="Run environment checks")
    p_check.set_defaults(func=cmd_check)

    # -------------------------------------------------------------------------
    # import subtool
    # -------------------------------------------------------------------------
    p_import = subparsers.add_parser(
        "import",
        help="Import DICOMs or load NIfTI data and report basic info"
    )
    add_io_arguments(p_import)
    p_import.set_defaults(func=cmd_import)

    # -------------------------------------------------------------------------
    # preprocess subtool
    # -------------------------------------------------------------------------
    p_preproc = subparsers.add_parser(
        "preprocess",
        help="Run preprocessing pipeline (denoise, skull strip, motion correction)"
    )
    add_io_arguments(p_preproc)
    p_preproc.add_argument(
        "--coil-count",
        type=int,
        default=4,
        help="Number of receiver coils for PIESNO noise estimation (N)."
    )
    p_preproc.add_argument(
        "--show-plots",
        action="store_true",
        help="Enable interactive QC plots (displayed, not saved)."
    )
    p_preproc.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save QC visualizations to output directory."
    )
    p_preproc.add_argument(
        "--perform-motion-correction",
        action="store_true",
        help="Enable between volume motion correction (disabled by default)."
    )
    p_preproc.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    p_preproc.set_defaults(func=cmd_preprocess)

    # -------------------------------------------------------------------------
    # track subtool
    # -------------------------------------------------------------------------
    p_track = subparsers.add_parser(
        "track",
        help="Run whole-brain deterministic tractography on preprocessed data"
    )
    p_track.add_argument(
        "--nifti",
        type=Path,
        required=True,
        help="Path to preprocessed NIfTI (.nii or .nii.gz).",
    )
    p_track.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject identifier for output naming (default: extracted from filename)."
    )
    p_track.add_argument(
        "--fa-thr",
        type=float,
        default=0.2,
        help="FA threshold for stopping and seeding (default 0.2).",
    )
    p_track.add_argument(
        "--seed-density",
        type=int,
        default=1,
        help="Seeds per voxel in the seed mask (default 1).",
    )
    p_track.add_argument(
        "--step-size",
        type=float,
        default=0.5,
        help="Tracking step size in millimetres (default 0.5).",
    )
    p_track.add_argument(
        "--sh-order",
        type=int,
        default=6,
        help="Maximum spherical harmonic order for CSA ODF model (default 6).",
    )
    p_track.add_argument(
        "--show-plots",
        action="store_true",
        help="Enable interactive QC plots (displayed, not saved).",
    )
    p_track.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save QC visualizations to output directory."
    )
    p_track.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    p_track.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for tractogram and scalar maps.",
    )
    p_track.set_defaults(func=cmd_track)

    # -------------------------------------------------------------------------
    # extract subtool
    # -------------------------------------------------------------------------
    p_extract = subparsers.add_parser(
        "extract",
        help="Extract bilateral CST from whole-brain tractogram using atlas-based ROI filtering"
    )
    p_extract.add_argument(
        "--tractogram",
        type=Path,
        required=True,
        help="Path to whole-brain tractogram (.trk)"
    )
    p_extract.add_argument(
        "--fa",
        type=Path,
        required=True,
        help="Path to subject FA map for registration"
    )
    p_extract.add_argument(
        "--subject-id",
        type=str,
        default="subject",
        help="Subject identifier for output naming"
    )
    p_extract.add_argument(
        "--min-length",
        type=float,
        default=20.0,
        help="Minimum streamline length in mm (default: 20)"
    )
    p_extract.add_argument(
        "--max-length",
        type=float,
        default=200.0,
        help="Maximum streamline length in mm (default: 200)"
    )
    p_extract.add_argument(
        "--dilate-brainstem",
        type=int,
        default=2,
        help="Dilation iterations for brainstem ROI (default: 2)"
    )
    p_extract.add_argument(
        "--dilate-motor",
        type=int,
        default=1,
        help="Dilation iterations for motor cortex ROIs (default: 1)"
    )
    p_extract.add_argument(
        "--fast-registration",
        action="store_true",
        help="Use reduced iterations for faster registration (testing mode)"
    )
    p_extract.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save QC visualizations to output directory."
    )
    p_extract.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    p_extract.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for extracted CST tractograms"
    )
    p_extract.set_defaults(func=cmd_extract)

    # -------------------------------------------------------------------------
    # metrics subtool
    # -------------------------------------------------------------------------
    p_metrics = subparsers.add_parser(
        "metrics",
        help="Compute bilateral CST metrics and generate reports"
    )
    p_metrics.add_argument(
        "--cst-left",
        type=Path,
        required=True,
        help="Path to left CST tractogram (.trk)"
    )
    p_metrics.add_argument(
        "--cst-right",
        type=Path,
        required=True,
        help="Path to right CST tractogram (.trk)"
    )
    p_metrics.add_argument(
        "--fa",
        type=Path,
        help="FA map for microstructural analysis"
    )
    p_metrics.add_argument(
        "--md",
        type=Path,
        help="MD map for microstructural analysis"
    )
    p_metrics.add_argument(
        "--subject-id",
        type=str,
        default="subject",
        help="Subject identifier for reports"
    )
    p_metrics.add_argument(
        "--generate-pdf",
        action="store_true",
        help="Generate PDF clinical report"
    )
    p_metrics.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save QC visualizations to output directory (enabled by default for metrics)."
    )
    p_metrics.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    p_metrics.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for metrics and reports"
    )
    p_metrics.set_defaults(func=cmd_metrics)

    # -------------------------------------------------------------------------
    # Parse and execute
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


# =============================================================================
# Helper functions
# =============================================================================

def add_io_arguments(p: argparse.ArgumentParser) -> None:
    """Add common IO arguments for import and preprocess."""
    p.add_argument("--dicom", type=Path, help="Path to DICOM directory.")
    p.add_argument(
        "--nifti",
        type=Path,
        help="Path to NIfTI file (.nii or .nii.gz). "
             "Sidecars (.bval/.bvec) must be in the same folder with the same stem."
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (also used for converted NIfTI)."
    )


def resolve_nifti(args: argparse.Namespace) -> Path:
    """Resolve a NIfTI path from DICOM or explicit NIfTI."""
    nii: Path | None = None

    # DICOM case
    if args.dicom and preproc.is_dicom_dir(args.dicom):
        print("Valid DICOM directory. Converting to NIfTI...")
        nii_str, _bval, _bvec = preproc.convert_to_nifti(args.dicom, args.out)
        nii = Path(nii_str)  # Convert string to Path

    else:
        # Provided DICOM path is invalid
        if args.dicom and not preproc.is_dicom_dir(args.dicom):
            print(f"Warning: {args.dicom} is not a valid DICOM directory. "
                  "Falling back to existing NIfTI.")

        # Use explicit NIfTI if given
        if args.nifti:
            nii = args.nifti if isinstance(args.nifti, Path) else Path(args.nifti)
        else:
            # Try to find a NIfTI in the output directory
            print(f"Attempting to find NIfTI in {args.out}...")
            candidates = list(args.out.glob("*.nii.gz")) + list(args.out.glob("*.nii"))
            if not candidates:
                raise FileNotFoundError(
                    "No NIfTI file found. Provide --nifti or a valid --dicom directory."
                )
            nii = candidates[0]

    if not nii.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nii}")

    return nii


def load_with_preproc(nii: Path):
    """Load data and gradients using preproc.load_dataset."""
    nifti_dir = str(nii.parent)
    name = nii.name

    if name.endswith(".nii.gz"):
        fname = name[:-7]
    elif name.endswith(".nii"):
        fname = name[:-4]
    else:
        raise ValueError("NIfTI must end with .nii or .nii.gz")

    data, affine, img, gtab = preproc.load_dataset(
        nifti_path=nifti_dir,
        fname=fname,
        visualize=False,
    )
    hdr = img.header
    return data, affine, hdr, gtab


def get_gtab_for_preproc(preproc_nii: Path):
    """
    Given a preprocessed NIfTI path like <stem>_preproc.nii.gz,
    find the original .bval and .bvec next to it and build gtab.
    """
    name = preproc_nii.name

    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        raise ValueError("Preprocessed NIfTI must end with .nii or .nii.gz")

    # Try to find matching bval/bvec files
    # First try exact stem match
    bval = preproc_nii.with_name(f"{stem}.bval")
    bvec = preproc_nii.with_name(f"{stem}.bvec")
    
    # If not found, try removing _preproc suffix variations
    if not bval.exists() or not bvec.exists():
        for suffix in ["_preproc", "_dwi_preproc_nomc", "_dwi_preproc_mc", "_dwi_preproc"]:
            if stem.endswith(suffix):
                orig_stem = stem[:-len(suffix)]
                bval = preproc_nii.with_name(f"{orig_stem}.bval")
                bvec = preproc_nii.with_name(f"{orig_stem}.bvec")
                if bval.exists() and bvec.exists():
                    break

    print(f"Using .bval: {bval}")
    print(f"Using .bvec: {bvec}")

    if not bval.exists() or not bvec.exists():
        raise FileNotFoundError(
            f"Missing .bval or .bvec for {stem}. "
            "Expected them next to the NIfTI file."
        )

    bvals, bvecs = read_bvals_bvecs(str(bval), str(bvec))
    gtab = gradient_table(bvals, bvecs=bvecs)
    return gtab


def extract_stem_from_filename(filename: str) -> str:
    """Extract clean stem from filename, removing common suffixes."""
    path = Path(filename)
    stem = path.stem  # Removes .gz if present
    
    # Handle .nii.gz case
    if stem.endswith('.nii'):
        stem = stem[:-4]
    
    # Remove common suffixes
    suffixes_to_remove = [
        '_dwi_preproc_nomc',
        '_dwi_preproc_mc', 
        '_preproc_nomc',
        '_preproc_mc',
        '_preproc',
        '_dwi'
    ]
    
    for suffix in suffixes_to_remove:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    
    # If stem is empty or very short, use original name
    if len(stem) < 3:
        stem = path.stem.replace('.nii', '')
    
    return stem


# =============================================================================
# Command implementations
# =============================================================================

def cmd_check(args: argparse.Namespace) -> None:
    """Runs environment checks."""
    print("csttool environment OK")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Version: {__version__}")
    
    # Check key dependencies
    try:
        import dipy
        print(f"DIPY: {dipy.__version__}")
    except ImportError:
        print("DIPY: NOT FOUND")
    
    try:
        import nibabel
        print(f"NiBabel: {nibabel.__version__}")
    except ImportError:
        print("NiBabel: NOT FOUND")
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except ImportError:
        print("NumPy: NOT FOUND")
    
    try:
        import matplotlib
        print(f"Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("Matplotlib: NOT FOUND")


def cmd_import(args: argparse.Namespace) -> None:
    """Import DICOM data or load an existing NIfTI dataset and print basic info."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    data, _affine, hdr, gtab = load_with_preproc(nii)

    print(f"\nDataset Information:")
    print(f"  File: {nii}")
    print(f"  Data shape: {data.shape}")
    print(f"  Gradient directions: {len(gtab.bvals)}")
    voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
    print(f"  Voxel size (mm): {voxel_size}")
    print(f"  B-values: {sorted(set(gtab.bvals.astype(int)))}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run the preprocessing pipeline on the input data."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    args.out.mkdir(parents=True, exist_ok=True)

    data, affine, _hdr, gtab = load_with_preproc(nii)
    
    # Keep original data for visualization comparison
    data_original = data.copy() if args.save_visualizations else None

    stem = nii.name.replace(".nii.gz", "").replace(".nii", "")

    print("Step 1: Denoising with NLMeans")
    denoised, brain_mask_piesno = preproc.denoise_nlmeans(
        data,
        N=args.coil_count,
        brain_mask=None,
        visualize=args.show_plots,
    )
    
    # Keep denoised data for visualization
    data_denoised = denoised.copy() if args.save_visualizations else None

    print("Step 2: Brain masking with median Otsu")
    masked_data, brain_mask = preproc.background_segmentation(
        denoised,
        gtab,
        visualize=args.show_plots,
    )

    reg_affines = None
    if args.perform_motion_correction:
        print("Step 3: Between volume motion correction")
        try:
            preprocessed, reg_affines = preproc.perform_motion_correction(
                masked_data,
                gtab,
                affine,
                brain_mask=brain_mask,
            )
            motion_correction_applied = True
            print("✓ Motion correction successful")
        except Exception as e:
            print(f"Motion correction failed: {e}")
            print("Continuing without motion correction")
            preprocessed = masked_data
            motion_correction_applied = False
    else:
        print("Skipping motion correction (default behavior)")
        preprocessed = masked_data
        motion_correction_applied = False

    print("Step 4: Saving output")
    output_path = preproc.save_output(
        preprocessed,
        affine,
        str(args.out),
        stem,
        motion_correction_applied=motion_correction_applied
    )

    print("Step 5: Copying gradient files")
    preproc.copy_gradient_files(
        nii, str(args.out), stem, motion_correction_applied
    )

    # Generate visualizations if requested
    if args.save_visualizations:
        print("\nStep 6: Generating QC visualizations")
        try:
            from csttool.preprocess.modules.visualizations import (
                save_all_preprocessing_visualizations
            )
            
            viz_paths = save_all_preprocessing_visualizations(
                data_original=data_original,
                data_denoised=data_denoised,
                data_preprocessed=preprocessed,
                brain_mask=brain_mask,
                gtab=gtab,
                output_dir=args.out,
                stem=stem,
                reg_affines=reg_affines,
                motion_correction_applied=motion_correction_applied,
                verbose=True
            )
            
            print(f"\nVisualizations saved:")
            for name, path in viz_paths.items():
                if path:
                    print(f"  {name}: {path}")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

    print(f"\n✓ Preprocessing complete. Output: {output_path}")


def cmd_track(args: argparse.Namespace) -> None:
    """
    Run whole-brain deterministic tractography on preprocessed data.
    
    Outputs:
        - Whole-brain tractogram (.trk)
        - FA map (.nii.gz)
        - MD map (.nii.gz)
        - Processing report (.json)
        - Visualizations (optional, with --save-visualizations)
    """
    preproc_nii = args.nifti
    verbose = getattr(args, 'verbose', False)
    
    if not preproc_nii.exists():
        print(f"Error: preprocessed NIfTI not found: {preproc_nii}")
        return

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
        return

    print("Building gradient table for preprocessed data")
    try:
        gtab = get_gtab_for_preproc(preproc_nii)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # -------------------------------------------------------------------------
    # Step 1: Brain masking
    # -------------------------------------------------------------------------
    print("\nStep 1: Brain masking with median Otsu")
    masked_data, brain_mask = preproc.background_segmentation(
        data,
        gtab,
        visualize=args.show_plots,
    )

    # -------------------------------------------------------------------------
    # Step 2: Tensor fitting
    # -------------------------------------------------------------------------
    print("\nStep 2: Tensor fit and scalar measures (FA, MD)")
    try:
        tenfit, fa, md, white_matter = fit_tensors(
            masked_data, 
            gtab, 
            brain_mask,
            fa_thresh=args.fa_thr,
            visualize=args.show_plots,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during tensor fitting: {e}")
        return

    # -------------------------------------------------------------------------
    # Step 3: Direction field estimation
    # -------------------------------------------------------------------------
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
        return

    # -------------------------------------------------------------------------
    # Step 4: Stopping criterion and seeds
    # -------------------------------------------------------------------------
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
        return

    # -------------------------------------------------------------------------
    # Step 5: Deterministic tracking
    # -------------------------------------------------------------------------
    print("\nStep 5: Deterministic tracking")
    try:
        streamlines = run_tractography(
            csapeaks,
            stopping_criterion,
            seeds,
            affine,
            step_size=args.step_size,
            verbose=verbose,
            visualize=args.show_plots
        )
    except Exception as e:
        print(f"Error during tractography: {e}")
        return

    # -------------------------------------------------------------------------
    # Step 6: Save outputs
    # -------------------------------------------------------------------------
    print("\nStep 6: Saving tractogram, scalar maps, and report")
    
    tracking_params = {
        'step_size': args.step_size,
        'fa_thresh': args.fa_thr,
        'seed_density': args.seed_density,
        'sh_order': args.sh_order,
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
            tracking_params=tracking_params,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error saving outputs: {e}")
        return

    # -------------------------------------------------------------------------
    # Step 7: Generate visualizations (optional)
    # -------------------------------------------------------------------------
    if args.save_visualizations:
        print("\nStep 7: Generating QC visualizations")
        try:
            from csttool.tracking.modules.visualizations import (
                save_all_tracking_visualizations
            )
            
            viz_paths = save_all_tracking_visualizations(
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
                verbose=True
            )
            
            outputs['visualizations'] = viz_paths
            
            print(f"\nVisualizations saved:")
            for name, path in viz_paths.items():
                if path:
                    print(f"  {name}: {path}")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TRACKING COMPLETE - {stem}")
    print(f"{'='*60}")
    print(f"Whole-brain streamlines: {len(streamlines):,}")
    print(f"\nOutputs:")
    for key, path in outputs.items():
        if key != 'visualizations':
            print(f"  {key}: {path}")
    if 'visualizations' in outputs:
        print(f"  visualizations: {args.out / 'visualizations'}")
    print(f"{'='*60}")


def cmd_extract(args: argparse.Namespace) -> None:
    """
    Extract bilateral CST using atlas-based ROI filtering.
    
    Pipeline:
        1. Register MNI template to subject space (Affine + SyN)
        2. Warp Harvard-Oxford atlases to subject space
        3. Create ROI masks (brainstem, motor cortex L/R)
        4. Filter streamlines by endpoint connectivity
        5. Save bilateral CST tractograms
    """
    import nibabel as nib
    
    verbose = getattr(args, 'verbose', True)
    
    # Validate inputs
    if not args.tractogram.exists():
        print(f"Error: Tractogram not found: {args.tractogram}")
        return
    
    if not args.fa.exists():
        print(f"Error: FA map not found: {args.fa}")
        return
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Import extraction modules
    try:
        from csttool.extract.modules.registration import register_mni_to_subject
        from csttool.extract.modules.warp_atlas_to_subject import (
            fetch_harvard_oxford, 
            warp_harvard_oxford_to_subject,
            CST_ROI_CONFIG
        )
        from csttool.extract.modules.create_roi_masks import (
            create_cst_roi_masks,
            visualize_roi_masks
        )
        from csttool.extract.modules.endpoint_filtering import (
            extract_bilateral_cst,
            save_cst_tractograms,
            save_extraction_report
        )
        from dipy.io.streamline import load_tractogram
    except ImportError as e:
        print(f"Error importing extraction modules: {e}")
        print("Make sure all extraction module dependencies are installed.")
        return
    
    # -------------------------------------------------------------------------
    # Step 1: Load inputs
    # -------------------------------------------------------------------------
    print(f"Loading tractogram: {args.tractogram}")
    try:
        sft = load_tractogram(str(args.tractogram), 'same')
        streamlines = sft.streamlines
        print(f"  Loaded {len(streamlines):,} streamlines")
    except Exception as e:
        print(f"Error loading tractogram: {e}")
        return
    
    print(f"Loading FA map: {args.fa}")
    try:
        fa_data, fa_affine = load_nifti(str(args.fa))
    except Exception as e:
        print(f"Error loading FA map: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Step 2: Registration (MNI → Subject)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Step 1: Registering MNI template to subject space")
    print("="*60)
    
    level_iters_affine = [1000, 100, 10] if args.fast_registration else [10000, 1000, 100]
    level_iters_syn = [5, 5, 3] if args.fast_registration else [10, 10, 5]
    
    if args.fast_registration:
        print("⚡ Fast registration mode enabled (reduced iterations)")
    
    try:
        reg_result = register_mni_to_subject(
            subject_fa_path=args.fa,
            output_dir=args.out,
            subject_id=args.subject_id,
            level_iters_affine=level_iters_affine,
            level_iters_syn=level_iters_syn,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during registration: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Step 3: Warp Harvard-Oxford atlases
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Step 2: Warping Harvard-Oxford atlases to subject space")
    print("="*60)
    
    try:
        atlases = fetch_harvard_oxford(verbose=verbose)
        warped = warp_harvard_oxford_to_subject(
            cortical_img=atlases['cortical_img'],
            subcortical_img=atlases['subcortical_img'],
            mapping=reg_result['mapping'],
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error warping atlases: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Step 4: Create ROI masks
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Step 3: Creating CST ROI masks")
    print("="*60)
    
    try:
        masks = create_cst_roi_masks(
            warped_cortical=warped['cortical'],
            warped_subcortical=warped['subcortical'],
            subject_affine=fa_affine,
            roi_config=CST_ROI_CONFIG,
            dilate_brainstem=args.dilate_brainstem,
            dilate_motor=args.dilate_motor,
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error creating ROI masks: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Step 5: Extract bilateral CST
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Step 4: Extracting bilateral CST")
    print("="*60)
    
    try:
        cst_result = extract_bilateral_cst(
            streamlines=streamlines,
            masks=masks,
            affine=fa_affine,
            min_length=args.min_length,
            max_length=args.max_length,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during CST extraction: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Step 6: Save outputs
    # -------------------------------------------------------------------------
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
        
        # Save extraction report
        save_extraction_report(cst_result, output_paths, args.out, args.subject_id)
    except Exception as e:
        print(f"Error saving outputs: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Step 7: Generate visualizations (optional)
    # -------------------------------------------------------------------------
    if args.save_visualizations:
        print("\n" + "="*60)
        print("Step 6: Generating QC visualizations")
        print("="*60)
        
        try:
            # ROI mask visualization
            viz_path = visualize_roi_masks(
                masks=masks,
                subject_fa=fa_data,
                output_dir=args.out,
                subject_id=args.subject_id,
                verbose=verbose
            )
            print(f"✓ ROI visualization: {viz_path}")
            
            # CST extraction visualization
            try:
                from csttool.extract.modules.visualizations import plot_cst_extraction
                cst_viz_path = plot_cst_extraction(
                    cst_result=cst_result,
                    fa=fa_data,
                    affine=fa_affine,
                    output_dir=args.out,
                    subject_id=args.subject_id,
                    verbose=verbose
                )
                print(f"✓ CST extraction visualization: {cst_viz_path}")
            except ImportError:
                # Fallback if visualization module doesn't have this function yet
                pass
                
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Subject: {args.subject_id}")
    print(f"Left CST:  {cst_result['stats']['cst_left_count']:,} streamlines")
    print(f"Right CST: {cst_result['stats']['cst_right_count']:,} streamlines")
    print(f"Total:     {cst_result['stats']['cst_total_count']:,} streamlines")
    print(f"Extraction rate: {cst_result['stats']['extraction_rate']:.2f}%")
    print(f"\nOutputs saved to: {args.out}")
    print(f"{'='*60}")


def cmd_metrics(args: argparse.Namespace) -> None:
    """
    Compute bilateral CST metrics and generate reports.
    
    Outputs:
        - JSON report (complete metrics)
        - CSV summary (for group analysis)
        - PDF report (optional, with visualizations)
        - Visualization plots
    """
    verbose = getattr(args, 'verbose', True)
    
    # Validate inputs
    if not args.cst_left.exists():
        print(f"Error: Left CST tractogram not found: {args.cst_left}")
        return
    
    if not args.cst_right.exists():
        print(f"Error: Right CST tractogram not found: {args.cst_right}")
        return
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Import metrics modules
    try:
        from dipy.io.streamline import load_tractogram
        from csttool.metrics import (
            analyze_cst_hemisphere,
            compare_bilateral_cst,
            print_hemisphere_summary,
            print_bilateral_summary,
            plot_tract_profiles,
            plot_bilateral_comparison,
        )
        from csttool.metrics.modules.reports import (
            save_json_report,
            save_csv_summary,
            save_pdf_report,
        )
    except ImportError as e:
        print(f"Error importing metrics modules: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Load tractograms
    # -------------------------------------------------------------------------
    print(f"Loading left CST: {args.cst_left}")
    try:
        sft_left = load_tractogram(str(args.cst_left), 'same')
        streamlines_left = sft_left.streamlines
        print(f"  Loaded {len(streamlines_left):,} streamlines")
    except Exception as e:
        print(f"Error loading left CST: {e}")
        return
    
    print(f"Loading right CST: {args.cst_right}")
    try:
        sft_right = load_tractogram(str(args.cst_right), 'same')
        streamlines_right = sft_right.streamlines
        print(f"  Loaded {len(streamlines_right):,} streamlines")
    except Exception as e:
        print(f"Error loading right CST: {e}")
        return
    
    # Load scalar maps
    fa_map, fa_affine = None, None
    md_map = None
    
    if args.fa:
        if args.fa.exists():
            print(f"Loading FA map: {args.fa}")
            fa_map, fa_affine = load_nifti(str(args.fa))
        else:
            print(f"Warning: FA map not found: {args.fa}")
    
    if args.md:
        if args.md.exists():
            print(f"Loading MD map: {args.md}")
            md_map, _ = load_nifti(str(args.md))
        else:
            print(f"Warning: MD map not found: {args.md}")
    
    # Use tractogram affine if FA affine not available
    affine = fa_affine if fa_affine is not None else sft_left.affine
    
    # -------------------------------------------------------------------------
    # Analyze each hemisphere
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analyzing LEFT CST")
    print("="*60)
    try:
        left_metrics = analyze_cst_hemisphere(
            streamlines=streamlines_left,
            fa_map=fa_map,
            md_map=md_map,
            affine=affine,
            hemisphere='left'
        )
        if verbose:
            print_hemisphere_summary(left_metrics)
    except Exception as e:
        print(f"Error analyzing left CST: {e}")
        return
    
    print("\n" + "="*60)
    print("Analyzing RIGHT CST")
    print("="*60)
    try:
        right_metrics = analyze_cst_hemisphere(
            streamlines=streamlines_right,
            fa_map=fa_map,
            md_map=md_map,
            affine=affine,
            hemisphere='right'
        )
        if verbose:
            print_hemisphere_summary(right_metrics)
    except Exception as e:
        print(f"Error analyzing right CST: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Bilateral comparison
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Computing bilateral comparison")
    print("="*60)
    try:
        comparison = compare_bilateral_cst(left_metrics, right_metrics)
    except Exception as e:
        print(f"Error during bilateral comparison: {e}")
        return
    
    # -------------------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Generating reports")
    print("="*60)
    
    try:
        json_path = save_json_report(comparison, args.out, args.subject_id)
        print(f"✓ JSON report: {json_path}")
    except Exception as e:
        print(f"Error saving JSON report: {e}")
    
    try:
        csv_path = save_csv_summary(comparison, args.out, args.subject_id)
        print(f"✓ CSV summary: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV summary: {e}")
    
    # -------------------------------------------------------------------------
    # Generate visualizations (always enabled for metrics, or with flag)
    # -------------------------------------------------------------------------
    # Metrics always generates visualizations by default
    save_viz = getattr(args, 'save_visualizations', True) or True
    
    viz_dir = args.out / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    viz_paths = {}
    
    try:
        if fa_map is not None and 'fa' in left_metrics:
            viz_paths['tract_profiles'] = plot_tract_profiles(
                left_metrics, right_metrics, viz_dir, args.subject_id, scalar='fa'
            )
            print(f"✓ Tract profiles: {viz_paths['tract_profiles']}")
    except Exception as e:
        print(f"Warning: Could not generate tract profiles: {e}")
    
    try:
        viz_paths['bilateral_comparison'] = plot_bilateral_comparison(
            comparison, viz_dir, args.subject_id
        )
        print(f"✓ Bilateral comparison: {viz_paths['bilateral_comparison']}")
    except Exception as e:
        print(f"Warning: Could not generate bilateral comparison: {e}")
    
    # -------------------------------------------------------------------------
    # Generate PDF if requested
    # -------------------------------------------------------------------------
    if args.generate_pdf:
        try:
            pdf_path = save_pdf_report(comparison, viz_paths, args.out, args.subject_id)
            print(f"✓ PDF report: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not generate PDF report: {e}")
            print("  (You may need to install reportlab: pip install reportlab)")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("METRICS COMPLETE")
    print(f"{'='*60}")
    print(f"Subject: {args.subject_id}")
    print(f"\nMorphology:")
    print(f"  Left CST:  {left_metrics['morphology']['n_streamlines']:,} streamlines, "
          f"volume = {left_metrics['morphology']['tract_volume']:.0f} mm³")
    print(f"  Right CST: {right_metrics['morphology']['n_streamlines']:,} streamlines, "
          f"volume = {right_metrics['morphology']['tract_volume']:.0f} mm³")
    
    if 'fa' in left_metrics:
        print(f"\nFA:")
        print(f"  Left:  {left_metrics['fa']['mean']:.3f} ± {left_metrics['fa']['std']:.3f}")
        print(f"  Right: {right_metrics['fa']['mean']:.3f} ± {right_metrics['fa']['std']:.3f}")
        if 'fa' in comparison['asymmetry']:
            print(f"  LI:    {comparison['asymmetry']['fa']['laterality_index']:.3f}")
    
    print(f"\nOutputs saved to: {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()