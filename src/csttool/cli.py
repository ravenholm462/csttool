"""
cli.py

Command-line interface for csttool - CST assessment using DTI data.

Commands:
    check       - Run environment checks
    import      - Import DICOMs or load NIfTI data (with series selection)
    preprocess  - Run preprocessing pipeline
    track       - Run whole-brain tractography
    extract     - Extract bilateral CST from tractogram
    metrics     - Compute CST metrics and generate reports
    run         - Run complete pipeline (check → import → preprocess → track → extract → metrics)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from time import time
import json

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
    # import subtool (with ingest module support)
    # -------------------------------------------------------------------------
    p_import = subparsers.add_parser(
        "import",
        help="Import DICOM data or load NIfTI with series selection and validation"
    )
    p_import.add_argument(
        "--dicom",
        type=Path,
        help="Path to DICOM directory (can be study root with multiple series)"
    )
    p_import.add_argument(
        "--nifti",
        type=Path,
        help="Path to existing NIfTI file (skip DICOM conversion)"
    )
    p_import.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for converted files"
    )
    p_import.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject identifier for output naming"
    )
    p_import.add_argument(
        "--series",
        type=int,
        default=None,
        help="Series number to convert (as shown in scan output, 1-indexed)"
    )
    p_import.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan and analyze series, don't convert"
    )
    p_import.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
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
        help="Enable QC plots for denoising and segmentation."
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
        help="Enable QC plots for segmentation and tractography.",
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
    # run subtool - COMPLETE PIPELINE
    # -------------------------------------------------------------------------
    p_run = subparsers.add_parser(
        "run",
        help="Run complete pipeline: check → import → preprocess → track → extract → metrics"
    )
    
    # Input options
    p_run.add_argument(
        "--dicom",
        type=Path,
        help="Path to DICOM directory"
    )
    p_run.add_argument(
        "--nifti",
        type=Path,
        help="Path to existing NIfTI file (skip import step)"
    )
    p_run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (subdirectories created for each step)"
    )
    p_run.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject identifier for all outputs"
    )
    
    # Import options
    p_run.add_argument(
        "--series",
        type=int,
        default=None,
        help="DICOM series number to convert (1-indexed)"
    )
    
    # Preprocessing options
    p_run.add_argument(
        "--coil-count",
        type=int,
        default=4,
        help="Number of receiver coils for PIESNO (default: 4)"
    )
    p_run.add_argument(
        "--perform-motion-correction",
        action="store_true",
        help="Enable motion correction during preprocessing"
    )
    
    # Tracking options
    p_run.add_argument(
        "--fa-thr",
        type=float,
        default=0.2,
        help="FA threshold for tracking (default: 0.2)"
    )
    p_run.add_argument(
        "--seed-density",
        type=int,
        default=1,
        help="Seeds per voxel (default: 1)"
    )
    p_run.add_argument(
        "--step-size",
        type=float,
        default=0.5,
        help="Tracking step size in mm (default: 0.5)"
    )
    p_run.add_argument(
        "--sh-order",
        type=int,
        default=6,
        help="Spherical harmonic order (default: 6)"
    )
    
    # Extraction options
    p_run.add_argument(
        "--min-length",
        type=float,
        default=20.0,
        help="Minimum streamline length in mm (default: 20)"
    )
    p_run.add_argument(
        "--max-length",
        type=float,
        default=200.0,
        help="Maximum streamline length in mm (default: 200)"
    )
    p_run.add_argument(
        "--dilate-brainstem",
        type=int,
        default=2,
        help="Brainstem ROI dilation (default: 2)"
    )
    p_run.add_argument(
        "--dilate-motor",
        type=int,
        default=1,
        help="Motor cortex ROI dilation (default: 1)"
    )
    p_run.add_argument(
        "--fast-registration",
        action="store_true",
        help="Use fast registration (reduced accuracy)"
    )
    
    # Metrics options
    p_run.add_argument(
        "--generate-pdf",
        action="store_true",
        help="Generate PDF clinical report"
    )
    
    # Pipeline control options
    p_run.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip environment check step"
    )
    p_run.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails"
    )
    p_run.add_argument(
        "--show-plots",
        action="store_true",
        help="Show QC plots during processing"
    )
    p_run.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    
    p_run.set_defaults(func=cmd_run)

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
        nii, _bval, _bvec = preproc.convert_to_nifti(args.dicom, args.out)

    else:
        # Provided DICOM path is invalid
        if args.dicom and not preproc.is_dicom_dir(args.dicom):
            print(f"Warning: {args.dicom} is not a valid DICOM directory. "
                  "Falling back to existing NIfTI.")

        # Use explicit NIfTI if given
        if args.nifti:
            nii = args.nifti
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


def find_files_recursive(directory: Path, pattern: str) -> list[Path]:
    """Recursively find files matching a pattern."""
    return list(directory.rglob(pattern))


# =============================================================================
# Command implementations
# =============================================================================

def cmd_check(args: argparse.Namespace) -> bool:
    """Runs environment checks. Returns True if all checks pass."""
    print("csttool environment check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Version: {__version__}")
    
    all_ok = True
    
    # Check key dependencies
    try:
        import dipy
        print(f"✓ DIPY: {dipy.__version__}")
    except ImportError:
        print("✗ DIPY: NOT FOUND")
        all_ok = False
    
    try:
        import nibabel
        print(f"✓ NiBabel: {nibabel.__version__}")
    except ImportError:
        print("✗ NiBabel: NOT FOUND")
        all_ok = False
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("✗ NumPy: NOT FOUND")
        all_ok = False
    
    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError:
        print("✗ SciPy: NOT FOUND")
        all_ok = False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib: NOT FOUND")
        all_ok = False
    
    try:
        import nilearn
        print(f"✓ Nilearn: {nilearn.__version__}")
    except ImportError:
        print("✗ Nilearn: NOT FOUND (needed for atlas)")
        all_ok = False
    
    # Check optional dependencies
    try:
        import reportlab
        print(f"✓ ReportLab: {reportlab.Version}")
    except ImportError:
        print("○ ReportLab: NOT FOUND (optional, for PDF reports)")
    
    # Check ingest module
    try:
        from csttool.ingest import run_ingest_pipeline
        print("✓ Ingest module: available")
    except ImportError:
        print("○ Ingest module: not installed (legacy import will be used)")
    
    if all_ok:
        print("\n✓ All required dependencies available")
    else:
        print("\n✗ Some dependencies missing - install with: pip install -e .")
    
    return all_ok


def cmd_import(args: argparse.Namespace) -> dict | None:
    """Import DICOM data or load an existing NIfTI dataset."""
    
    # Try to use the new ingest module
    try:
        from csttool.ingest import run_ingest_pipeline, scan_study
        USE_INGEST = True
    except ImportError:
        USE_INGEST = False
        print("Note: Ingest module not available, using legacy import")
    
    if USE_INGEST and args.dicom:
        # Use new ingest pipeline
        args.out.mkdir(parents=True, exist_ok=True)
        
        # Scan-only mode
        if getattr(args, 'scan_only', False):
            print(f"Scanning DICOM directory: {args.dicom}")
            series_list = scan_study(args.dicom)
            
            if not series_list:
                print("No valid DICOM series found.")
                return None
            
            print(f"\nFound {len(series_list)} series.")
            return {'series': series_list, 'scan_only': True}
        
        # Full conversion
        result = run_ingest_pipeline(
            study_dir=args.dicom,
            output_dir=args.out,
            series_index=args.series - 1 if args.series else None,
            subject_id=args.subject_id,
            verbose=getattr(args, 'verbose', False)
        )
        
        if result and result.get('nifti_path'):
            print(f"\n✓ Import complete: {result['nifti_path']}")
            return result
        else:
            print("Import failed.")
            return None
    
    else:
        # Legacy import behavior
        return cmd_import_legacy(args)


def cmd_import_legacy(args: argparse.Namespace) -> dict | None:
    """Legacy import using preproc functions."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    data, _affine, hdr, gtab = load_with_preproc(nii)

    print(f"\nDataset Information:")
    print(f"  File: {nii}")
    print(f"  Data shape: {data.shape}")
    print(f"  Gradient directions: {len(gtab.bvals)}")
    voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
    print(f"  Voxel size (mm): {voxel_size}")
    print(f"  B-values: {sorted(set(gtab.bvals.astype(int)))}")
    
    return {
        'nifti_path': nii,
        'data_shape': data.shape,
        'n_gradients': len(gtab.bvals)
    }


def cmd_preprocess(args: argparse.Namespace) -> dict | None:
    """Run the preprocessing pipeline on the input data."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    args.out.mkdir(parents=True, exist_ok=True)

    data, affine, _hdr, gtab = load_with_preproc(nii)

    stem = nii.name.replace(".nii.gz", "").replace(".nii", "")

    print("Step 1: Denoising with NLMeans")
    denoised, brain_mask_piesno = preproc.denoise_nlmeans(
        data,
        N=args.coil_count,
        brain_mask=None,
        visualize=getattr(args, 'show_plots', False),
    )

    print("Step 2: Brain masking with median Otsu")
    masked_data, brain_mask = preproc.background_segmentation(
        denoised,
        gtab,
        visualize=getattr(args, 'show_plots', False),
    )

    motion_correction_applied = False
    if getattr(args, 'perform_motion_correction', False):
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
    else:
        print("Skipping motion correction (default behavior)")
        preprocessed = masked_data

    print("Step 4: Saving output")
    output_paths = preproc.save_output(
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
    
    # Save brain mask
    mask_path = preproc.save_brain_mask(brain_mask, affine, str(args.out), stem)

    print(f"\n✓ Preprocessing complete.")
    
    return {
        'preprocessed_path': output_paths['preprocessed_dwi'],
        'motion_correction': motion_correction_applied,
        'stem': stem
    }


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
    masked_data, brain_mask = preproc.background_segmentation(
        data,
        gtab,
        visualize=getattr(args, 'show_plots', False),
    )

    # Step 2: Tensor fitting
    print("\nStep 2: Tensor fit and scalar measures (FA, MD)")
    try:
        tenfit, fa, md, white_matter = fit_tensors(
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
        return None

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
        'n_streamlines': len(streamlines),
        'stem': stem
    }


def cmd_extract(args: argparse.Namespace) -> dict | None:
    """
    Extract bilateral CST using atlas-based ROI filtering.
    """
    import nibabel as nib
    
    verbose = getattr(args, 'verbose', True)
    
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
        from dipy.io.streamline import load_tractogram
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
        return None
    
    # Step 4: Extract CST
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


def cmd_metrics(args: argparse.Namespace) -> dict | None:
    """
    Compute bilateral CST metrics and generate reports.
    """
    verbose = getattr(args, 'verbose', True)
    
    # Validate inputs
    if not args.cst_left.exists():
        print(f"Error: Left CST tractogram not found: {args.cst_left}")
        return None
    
    if not args.cst_right.exists():
        print(f"Error: Right CST tractogram not found: {args.cst_right}")
        return None
    
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
        return None
    
    # Load tractograms
    print(f"Loading left CST: {args.cst_left}")
    try:
        sft_left = load_tractogram(str(args.cst_left), 'same')
        streamlines_left = sft_left.streamlines
        print(f"  Loaded {len(streamlines_left):,} streamlines")
    except Exception as e:
        print(f"Error loading left CST: {e}")
        return None
    
    print(f"Loading right CST: {args.cst_right}")
    try:
        sft_right = load_tractogram(str(args.cst_right), 'same')
        streamlines_right = sft_right.streamlines
        print(f"  Loaded {len(streamlines_right):,} streamlines")
    except Exception as e:
        print(f"Error loading right CST: {e}")
        return None
    
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
    
    affine = fa_affine if fa_affine is not None else sft_left.affine
    
    # Analyze hemispheres
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
        return None
    
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
        return None
    
    # Bilateral comparison
    print("\n" + "="*60)
    print("Computing bilateral comparison")
    print("="*60)
    try:
        comparison = compare_bilateral_cst(left_metrics, right_metrics)
    except Exception as e:
        print(f"Error during bilateral comparison: {e}")
        return None
    
    # Save reports
    print("\n" + "="*60)
    print("Generating reports")
    print("="*60)
    
    json_path = None
    csv_path = None
    
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
    
    # Generate visualizations
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
    
    # Generate PDF if requested
    pdf_path = None
    if getattr(args, 'generate_pdf', False):
        try:
            pdf_path = save_pdf_report(comparison, viz_paths, args.out, args.subject_id)
            print(f"✓ PDF report: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not generate PDF report: {e}")
    
    # Summary
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
    
    print(f"{'='*60}")
    
    return {
        'json_path': json_path,
        'csv_path': csv_path,
        'pdf_path': pdf_path,
        'comparison': comparison
    }


def cmd_run(args: argparse.Namespace) -> None:
    """
    Run complete CST analysis pipeline.
    
    Steps:
        1. check     - Verify environment and dependencies
        2. import    - Convert DICOM to NIfTI (or validate existing NIfTI)
        3. preprocess - Denoise, skull strip, (optional) motion correct
        4. track     - Whole-brain deterministic tractography
        5. extract   - Atlas-based bilateral CST extraction
        6. metrics   - Compute metrics and generate reports
    """
    
    verbose = getattr(args, 'verbose', False)
    continue_on_error = getattr(args, 'continue_on_error', False)
    
    # Create output directory structure
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Determine subject ID
    subject_id = args.subject_id
    if not subject_id:
        if args.dicom:
            subject_id = args.dicom.name
        elif args.nifti:
            subject_id = extract_stem_from_filename(str(args.nifti))
        else:
            subject_id = "subject"
    
    # Initialize pipeline tracking
    pipeline_start = time()
    step_times = {}
    step_results = {}
    failed_steps = []
    
    print("\n" + "="*70)
    print("CSTTOOL - COMPLETE CST ANALYSIS PIPELINE")
    print("="*70)
    print(f"Subject ID:     {subject_id}")
    print(f"Output:         {args.out}")
    print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # =========================================================================
    # STEP 1: CHECK
    # =========================================================================
    if not getattr(args, 'skip_check', False):
        print("\n" + "▶"*3 + " STEP 1/6: ENVIRONMENT CHECK " + "◀"*3)
        t0 = time()
        
        try:
            # Create a mock args object for cmd_check
            check_args = argparse.Namespace()
            check_ok = cmd_check(check_args)
            step_results['check'] = {'success': check_ok}
            
            if not check_ok and not continue_on_error:
                print("\n✗ Environment check failed. Fix dependencies and retry.")
                return
                
        except Exception as e:
            print(f"✗ Check failed: {e}")
            failed_steps.append('check')
            step_results['check'] = {'success': False, 'error': str(e)}
            if not continue_on_error:
                return
        
        step_times['check'] = time() - t0
    else:
        print("\n" + "⏭ STEP 1/6: SKIPPING ENVIRONMENT CHECK")
        step_results['check'] = {'success': True, 'skipped': True}
    
    # =========================================================================
    # STEP 2: IMPORT
    # =========================================================================
    print("\n" + "▶"*3 + " STEP 2/6: IMPORT DATA " + "◀"*3)
    t0 = time()
    
    nifti_path = None
    
    try:
        if args.nifti and args.nifti.exists():
            # Use provided NIfTI directly
            nifti_path = args.nifti
            print(f"Using existing NIfTI: {nifti_path}")
            step_results['import'] = {'success': True, 'nifti_path': str(nifti_path)}
        elif args.dicom:
            # Run import from DICOM
            import_args = argparse.Namespace(
                dicom=args.dicom,
                nifti=None,
                out=args.out,
                subject_id=subject_id,
                series=getattr(args, 'series', None),
                scan_only=False,
                verbose=verbose
            )
            
            import_result = cmd_import(import_args)
            
            if import_result and import_result.get('nifti_path'):
                nifti_path = Path(import_result['nifti_path'])
                step_results['import'] = {'success': True, 'result': import_result}
            else:
                raise RuntimeError("Import failed to produce NIfTI file")
        else:
            raise ValueError("Must provide --dicom or --nifti")
            
    except Exception as e:
        print(f"✗ Import failed: {e}")
        failed_steps.append('import')
        step_results['import'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            _save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['import'] = time() - t0
    
    # =========================================================================
    # STEP 3: PREPROCESS
    # =========================================================================
    print("\n" + "▶"*3 + " STEP 3/6: PREPROCESSING " + "◀"*3)
    t0 = time()
    
    preproc_path = None
    
    try:
        if nifti_path is None:
            raise RuntimeError("No NIfTI available from import step")
        
        preproc_args = argparse.Namespace(
            dicom=None,
            nifti=nifti_path,
            out=args.out,
            coil_count=getattr(args, 'coil_count', 4),
            show_plots=getattr(args, 'show_plots', False),
            perform_motion_correction=getattr(args, 'perform_motion_correction', False),
            verbose=verbose
        )
        
        preproc_result = cmd_preprocess(preproc_args)
        
        if preproc_result and preproc_result.get('preprocessed_path'):
            preproc_path = Path(preproc_result['preprocessed_path'])
            step_results['preprocess'] = {'success': True, 'result': preproc_result}
        else:
            raise RuntimeError("Preprocessing failed")
            
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        failed_steps.append('preprocess')
        step_results['preprocess'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            _save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['preprocess'] = time() - t0
    
    # =========================================================================
    # STEP 4: TRACK
    # =========================================================================
    print("\n" + "▶"*3 + " STEP 4/6: TRACTOGRAPHY " + "◀"*3)
    t0 = time()
    
    tractogram_path = None
    fa_path = None
    md_path = None
    
    try:
        if preproc_path is None:
            raise RuntimeError("No preprocessed data available")
        
        track_out = args.out / "tracking"
        track_args = argparse.Namespace(
            nifti=preproc_path,
            subject_id=subject_id,
            fa_thr=getattr(args, 'fa_thr', 0.2),
            seed_density=getattr(args, 'seed_density', 1),
            step_size=getattr(args, 'step_size', 0.5),
            sh_order=getattr(args, 'sh_order', 6),
            show_plots=getattr(args, 'show_plots', False),
            verbose=verbose,
            out=track_out
        )
        
        track_result = cmd_track(track_args)
        
        if track_result:
            tractogram_path = Path(track_result['tractogram_path'])
            fa_path = Path(track_result['fa_path'])
            md_path = Path(track_result['md_path'])
            step_results['track'] = {'success': True, 'result': track_result}
        else:
            raise RuntimeError("Tracking failed")
            
    except Exception as e:
        print(f"✗ Tracking failed: {e}")
        failed_steps.append('track')
        step_results['track'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            _save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['track'] = time() - t0
    
    # =========================================================================
    # STEP 5: EXTRACT
    # =========================================================================
    print("\n" + "▶"*3 + " STEP 5/6: CST EXTRACTION " + "◀"*3)
    t0 = time()
    
    cst_left_path = None
    cst_right_path = None
    
    try:
        if tractogram_path is None or fa_path is None:
            raise RuntimeError("No tractogram or FA map available")
        
        extract_out = args.out / "extraction"
        extract_args = argparse.Namespace(
            tractogram=tractogram_path,
            fa=fa_path,
            subject_id=subject_id,
            min_length=getattr(args, 'min_length', 20.0),
            max_length=getattr(args, 'max_length', 200.0),
            dilate_brainstem=getattr(args, 'dilate_brainstem', 2),
            dilate_motor=getattr(args, 'dilate_motor', 1),
            fast_registration=getattr(args, 'fast_registration', False),
            verbose=verbose,
            out=extract_out
        )
        
        extract_result = cmd_extract(extract_args)
        
        if extract_result and extract_result.get('cst_left_path') and extract_result.get('cst_right_path'):
            cst_left_path = Path(extract_result['cst_left_path'])
            cst_right_path = Path(extract_result['cst_right_path'])
            step_results['extract'] = {'success': True, 'result': extract_result}
        else:
            raise RuntimeError("CST extraction failed or produced no streamlines")
            
    except Exception as e:
        print(f"✗ CST extraction failed: {e}")
        failed_steps.append('extract')
        step_results['extract'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            _save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['extract'] = time() - t0
    
    # =========================================================================
    # STEP 6: METRICS
    # =========================================================================
    print("\n" + "▶"*3 + " STEP 6/6: METRICS & REPORTS " + "◀"*3)
    t0 = time()
    
    try:
        if cst_left_path is None or cst_right_path is None:
            raise RuntimeError("No CST tractograms available")
        
        metrics_out = args.out / "metrics"
        metrics_args = argparse.Namespace(
            cst_left=cst_left_path,
            cst_right=cst_right_path,
            fa=fa_path,
            md=md_path,
            subject_id=subject_id,
            generate_pdf=getattr(args, 'generate_pdf', False),
            verbose=verbose,
            out=metrics_out
        )
        
        metrics_result = cmd_metrics(metrics_args)
        
        if metrics_result:
            step_results['metrics'] = {'success': True, 'result': metrics_result}
        else:
            raise RuntimeError("Metrics computation failed")
            
    except Exception as e:
        print(f"✗ Metrics failed: {e}")
        failed_steps.append('metrics')
        step_results['metrics'] = {'success': False, 'error': str(e)}
    
    step_times['metrics'] = time() - t0
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time() - pipeline_start
    
    # Save pipeline report
    report_path = _save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Subject ID:     {subject_id}")
    print(f"Total time:     {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"\nStep timing:")
    for step, elapsed in step_times.items():
        status = "✓" if step not in failed_steps else "✗"
        print(f"  {status} {step:12s}: {elapsed:.1f}s")
    
    if failed_steps:
        print(f"\n⚠️  Failed steps: {', '.join(failed_steps)}")
    else:
        print(f"\n✓ All steps completed successfully!")
    
    print(f"\nOutputs:")
    print(f"  Pipeline report: {report_path}")
    if step_results.get('metrics', {}).get('success'):
        print(f"  Metrics:         {args.out / 'metrics'}")
    if step_results.get('extract', {}).get('success'):
        print(f"  CST tractograms: {args.out / 'extraction'}")
    
    print("="*70)


def _save_pipeline_report(
    output_dir: Path,
    subject_id: str,
    step_results: dict,
    step_times: dict,
    failed_steps: list,
    start_time: float
) -> Path:
    """Save pipeline execution report as JSON."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    report = {
        'subject_id': subject_id,
        'pipeline_version': __version__,
        'execution': {
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_seconds': time() - start_time,
            'success': len(failed_steps) == 0
        },
        'step_times': step_times,
        'failed_steps': failed_steps,
        'step_results': {
            step: {
                'success': result.get('success', False),
                'error': result.get('error') if 'error' in result else None
            }
            for step, result in step_results.items()
        }
    }
    
    report_path = log_dir / f"{subject_id}_pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report_path


if __name__ == "__main__":
    main()