from __future__ import annotations
import argparse
from pathlib import Path
from . import __version__
from csttool.preprocess.import_data import is_dicom_dir, convert_to_nifti, load_data

def main() -> None:
    
    # Main parser
    parser = argparse.ArgumentParser(
        prog="csttool",
        description="CST assessment tool using DTI data."
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"        
        )
    
    # Subparser container
    # Defines an environment to add "subtools" to 
    subparsers = parser.add_subparsers(dest="command")
    
    # check subtool
    p_check = subparsers.add_parser("check", help="Run environment checks")
    p_check.set_defaults(func=cmd_check)

    # import subtool
    p_import = subparsers.add_parser("import", help="Import DICOMs or load NIfTI data")
    p_import.add_argument("--dicom", type=Path, help="Path to DICOM directory.")
    p_import.add_argument("--nifti", type=Path, help="Path to NIfTI file (.nii or .nii.gz).")
    p_import.add_argument("--bval", type=Path, help="Optional path to .bval (used with --nifti).")
    p_import.add_argument("--bvec", type=Path, help="Optional path to .bvec (used with --nifti).")
    p_import.add_argument("--out",  type=Path, required=True, help="Output directory.")
    p_import.set_defaults(func=cmd_import)
    
    args = parser.parse_args()    
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        
def cmd_check(args: argparse.Namespace) -> None:
    """Check if env ok."""
    import sys
    print("csttool environment OK")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Version: {__version__}")

def cmd_import(args: argparse.Namespace) -> None:
    """Import DICOMs or load an existing NIfTI dataset."""

    nii = None
    bval = None
    bvec = None

    # If valid DICOM directory:
    if args.dicom and is_dicom_dir(args.dicom):
        print("Valid DICOM directory. Converting to NIfTI...")
        nii, bval, bvec = convert_to_nifti(args.dicom, args.out)

    else:
        # If not a valid DICOM directory:
        if args.dicom and not is_dicom_dir(args.dicom):
            print(f"Warning: {args.dicom} is not a valid DICOM directory. "
                  "Falling back to existing NIfTI.")

        # If NIfTI provided:
        if args.nifti:
            nii = args.nifti
        else:
            # If NIfTI not provided and no DICOM directory:
            print(f"Attempting to find NIfTI in {args.out}...")
            nii_candidates = list(args.out.glob("*.nii*"))
            if not nii_candidates:
                print("Error: no NIfTI file found. "
                      "Provide --nifti or a valid --dicom directory.")
                return
            nii = nii_candidates[0]

        if not nii.exists():
            print(f"Error: NIfTI file not found: {nii}")
            return

        # Derive sidecars next to the NIfTI if not provided
        stem = nii.name.replace(".nii.gz", "").replace(".nii", "")
        bval = args.bval or nii.with_name(f"{stem}.bval")
        bvec = args.bvec or nii.with_name(f"{stem}.bvec")

        if not bval.exists() or not bvec.exists():
            print("Warning: missing .bval or .bvec — attempting to continue.")

    # Report
    data, _affine, hdr, gtab = load_data(nii, bval, bvec)
    print(f"Data shape: {data.shape}")
    print(f"Gradient directions: {len(gtab.bvals)}")
    voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])

    print(f"Voxel size (mm): {voxel_size}")

if __name__ == "__main__":
    main()