from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
import csttool.preprocess.funcs as preproc
import csttool.tracking.funcs as trk

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

    # check subtool
    p_check = subparsers.add_parser("check", help="Run environment checks")
    p_check.set_defaults(func=cmd_check)

    # import subtool
    p_import = subparsers.add_parser(
        "import",
        help="Import DICOMs or load NIfTI data and report basic info"
    )
    add_io_arguments(p_import)
    p_import.set_defaults(func=cmd_import)

    # preprocess subtool
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
        "--skip-motion-correction",
        action="store_true",
        help="Skip between volume motion correction."
    )
    p_preproc.set_defaults(func=cmd_preprocess)

    # track subtool
    p_track = subparsers.add_parser(
        "track",
        help="Run deterministic tractography on preprocessed data"
    )
    p_track.add_argument(
        "--nifti",
        type=Path,
        required=True,
        help="Path to preprocessed NIfTI (.nii or .nii.gz).",
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
        help="Enable QC plots used by background segmentation.",
    )
    p_track.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for tractogram.",
    )
    p_track.set_defaults(func=cmd_track)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


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


def cmd_check(args: argparse.Namespace) -> None:
    """Runs environment checks."""
    import sys

    print("csttool environment OK")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Version: {__version__}")


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

    if stem.endswith("_preproc"):
        orig_stem = stem[:-8]
    else:
        # If user gives a non preproc name, assume sidecars match it
        orig_stem = stem

    bval = preproc_nii.with_name(f"{orig_stem}.bval")
    bvec = preproc_nii.with_name(f"{orig_stem}.bvec")

    print(f"Using .bval: {bval}")
    print(f"Using .bvec: {bvec}")

    if not bval.exists() or not bvec.exists():
        raise FileNotFoundError(
            f"Missing .bval or .bvec for {orig_stem}. "
            "Expected them next to the NIfTI file."
        )

    bvals, bvecs = read_bvals_bvecs(str(bval), str(bvec))
    gtab = gradient_table(bvals, bvecs=bvecs)
    return gtab


def cmd_import(args: argparse.Namespace) -> None:
    """Import DICOM data or load an existing NIfTI dataset and print basic info."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    data, _affine, hdr, gtab = load_with_preproc(nii)

    print(f"Data shape: {data.shape}")
    print(f"Gradient directions: {len(gtab.bvals)}")
    voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
    print(f"Voxel size (mm): {voxel_size}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run the preprocessing pipeline on the input data."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    args.out.mkdir(parents=True, exist_ok=True)

    data, affine, _hdr, gtab = load_with_preproc(nii)

    stem = nii.name.replace(".nii.gz", "").replace(".nii", "")

    print("Step 1: Denoising with NLMeans")
    denoised, brain_mask_piesno = preproc.denoise_nlmeans(
        data,
        N=args.coil_count,
        brain_mask=None,
        visualize=args.show_plots,
    )

    print("Step 2: Brain masking with median Otsu")
    masked_data, brain_mask = preproc.background_segmentation(
        denoised,
        gtab,
        visualize=args.show_plots,
    )

    if args.skip_motion_correction:
        print("Skipping motion correction as requested")
        preprocessed = masked_data
    else:
        print("Step 3: Between volume motion correction")
        preprocessed, reg_affines = preproc.perform_motion_correction(
            masked_data,
            gtab,
            affine,
            brain_mask=brain_mask,
        )
        # reg_affines could be saved in a future extension

    print("Step 4: Saving output")
    output_path = preproc.save_output(
        preprocessed,
        affine,
        str(args.out),
        stem,
    )
    print(f"Preprocessing complete. Output: {output_path}")

def cmd_track(args: argparse.Namespace) -> None:
    """
    Run deterministic tractography on a preprocessed NIfTI dataset.

    Expected workflow:
      1) Run `csttool preprocess` to create <stem>_preproc.nii.gz
      2) Run `csttool track --nifti <stem>_preproc.nii.gz --out <dir>`
    """
    preproc_nii = args.nifti
    if not preproc_nii.exists():
        print(f"Error: preprocessed NIfTI not found: {preproc_nii}")
        return

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading preprocessed data from {preproc_nii}")
    data, affine, img = load_nifti(str(preproc_nii), return_img=True)

    print("Building gradient table for preprocessed data")
    try:
        gtab = get_gtab_for_preproc(preproc_nii)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Step 1: Brain masking with median Otsu")
    # Reuse segmentation on the preprocessed data
    masked_data, brain_mask = preproc.background_segmentation(
        data,
        gtab,
        visualize=args.show_plots,
    )

    print("Step 2: Tensor fit and scalar measures")
    tenfit = trk.fit_tensor(masked_data, gtab, brain_mask=brain_mask)
    fa, md = trk.compute_measures(tenfit)

    print("Step 3: Direction field estimation")
    peaks = trk.get_directions(
        masked_data,
        gtab,
        mask=brain_mask,
        sh_order_max=args.sh_order,
    )

    print("Step 4: Stopping criterion and seeds")
    stopping_criterion, seeds, seed_mask = trk.terminate_and_seed(
        fa,
        affine,
        fa_thr=args.fa_thr,
        seed_density=args.seed_density,
    )

    print("Step 5: Deterministic tracking")
    streamlines = trk.run_deterministic_tracking(
        peaks,
        stopping_criterion,
        seeds,
        affine,
        step_size=args.step_size,
    )

    # Save tractogram
    stem = preproc_nii.name.replace(".nii.gz", "").replace(".nii", "")
    tract_name = f"{stem}_det"
    print("Step 6: Saving tractogram")
    trk.save_tractogram_trk(
        streamlines,
        img,
        args.out,
        fname=tract_name,
    )


if __name__ == "__main__":
    main()