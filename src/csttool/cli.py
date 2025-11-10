from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
import csttool.preprocess.funcs as preproc


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


if __name__ == "__main__":
    main()