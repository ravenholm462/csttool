
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
        print("  ⚠️ Ingest module not available, using legacy import")
    
    if USE_INGEST and args.dicom:
        # Use new ingest pipeline
        args.out.mkdir(parents=True, exist_ok=True)
        
        # Scan-only mode
        if getattr(args, 'scan_only', False):
            print(f"  → Scanning DICOM directory: {args.dicom}")
            series_list = scan_study(args.dicom)

            if not series_list:
                print("  ⚠️ No valid DICOM series found")
                return None

            print(f"  ✓ Found {len(series_list)} series")
            return {'series': series_list, 'scan_only': True}
        
        # Full conversion
        result = run_ingest_pipeline(
            study_dir=args.dicom,
            output_dir=args.out,
            series_index=getattr(args, 'series', None),
            series_uid=getattr(args, 'series_uid', None),
            subject_id=args.subject_id,
            verbose=getattr(args, 'verbose', False),
            field_strength=getattr(args, 'field_strength', None),
            echo_time=getattr(args, 'echo_time', None)
        )
        
        if result and result.get('nifti_path'):
            print(f"\n✓ Import complete")
            print(f"  {result['nifti_path']}")
            return result
        else:
            print("  ✗ Import failed")
            return None
    
    else:
        # Legacy import behavior
        return cmd_import_legacy(args)


def cmd_import_legacy(args: argparse.Namespace) -> dict | None:
    """Legacy import using preproc functions."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return None

    data, _affine, hdr, gtab, bids_json = load_with_preproc(nii)

    print(f"\n✓ Dataset loaded")
    print(f"  File:       {nii}")
    print(f"  Shape:      {data.shape}")
    print(f"  Directions: {len(gtab.bvals)}")
    voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
    print(f"  Voxel size: {voxel_size[0]:.2f} x {voxel_size[1]:.2f} x {voxel_size[2]:.2f} mm")
    print(f"  B-values:   {sorted(set(gtab.bvals.astype(int)))}")
    
    # Build CLI overrides for acquisition metadata
    overrides = {}
    if getattr(args, 'field_strength', None):
        overrides['field_strength_T'] = args.field_strength
    if getattr(args, 'echo_time', None):
        overrides['echo_time_ms'] = args.echo_time
    
    # Extract full acquisition metadata
    from csttool.ingest import extract_acquisition_metadata
    acquisition = extract_acquisition_metadata(
        bvecs=gtab.bvecs,
        bvals=gtab.bvals,
        voxel_size=voxel_size,
        bids_json=bids_json,
        overrides=overrides
    )
    
    return {
        'nifti_path': nii,
        'data_shape': data.shape,
        'n_gradients': len(gtab.bvals),
        'metadata': acquisition
    }
