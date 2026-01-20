
import argparse
from pathlib import Path
from ..utils import resolve_nifti, load_with_preproc

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
            series_index=args.series,
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

    data, _affine, hdr, gtab, metadata = load_with_preproc(nii)

    print(f"\nDataset Information:")
    print(f"  File: {nii}")
    print(f"  Data shape: {data.shape}")
    print(f"  Gradient directions: {len(gtab.bvals)}")
    voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
    print(f"  Voxel size (mm): {voxel_size}")
    print(f"  B-values: {sorted(set(gtab.bvals.astype(int)))}")
    
    # Merge basic info into metadata if not present
    if 'VoxelSize' not in metadata:
        metadata['VoxelSize'] = voxel_size
    if 'NumDirections' not in metadata:
        metadata['NumDirections'] = len(gtab.bvals)
    
    return {
        'nifti_path': nii,
        'data_shape': data.shape,
        'n_gradients': len(gtab.bvals),
        'metadata': metadata
    }
