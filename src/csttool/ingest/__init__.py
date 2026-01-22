"""
Ingest Module

DICOM import and conversion for csttool's diffusion tractography pipeline.

This module provides:
- DICOM study scanning and series discovery
- Series analysis and suitability assessment
- DICOM to NIfTI conversion with gradient file handling
- Organized output structure


Typical usage:
    from csttool.ingest.modules import (
        scan_study,
        analyze_all_series,
        recommend_series,
        convert_dicom_to_nifti,
        save_ingest_outputs
    )
    
    # Scan for series
    series_list = scan_study("/path/to/study")
    
    # Analyze and recommend
    analyses = analyze_all_series(series_list)
    best = recommend_series(analyses)
    
    # Convert
    result = convert_dicom_to_nifti(best.path, output_dir)
    
    # Save with structure
    outputs = save_ingest_outputs(result, best, output_dir)
"""

# Import key functions for convenient access
from .modules.scan_study import (
    is_dicom_directory,
    scan_study,
    print_series_summary
)

from .modules.analyze_series import (
    SeriesType,
    SeriesAnalysis,
    analyze_series,
    analyze_all_series,
    recommend_series
)

from .modules.convert_series import (
    convert_dicom_to_nifti,
    validate_conversion
)

from .modules.save_ingest_outputs import (
    save_ingest_outputs,
    print_import_summary,
    get_nifti_stem
)

from .modules.assess_quality import assess_acquisition_quality, extract_acquisition_metadata


def run_ingest_pipeline(
    study_dir,
    output_dir,
    subject_id=None,
    series_index=None,
    series_uid=None,
    auto_select=True,
    verbose=True,
    field_strength=None,
    echo_time=None
):
    """
    Run the complete ingest pipeline.
    
    Convenience function that runs all ingest steps:
    1. Scan study directory for DICOM series
    2. Analyze all series for tractography suitability
    3. Select best series (or use specified index)
    4. Convert DICOM to NIfTI
    5. Validate conversion
    6. Save organized outputs
    
    Parameters
    ----------
    study_dir : str or Path
        Path to study directory containing DICOM series
    output_dir : str or Path
        Directory for output files
    subject_id : str, optional
        Subject identifier. If None, extracted from series name.
    series_index : int, optional
        Index of series to convert (1-based, as shown in scan output).
        If None and auto_select=True, best series is automatically selected.
    auto_select : bool, optional
        Automatically select the best series. Default is True.
        If False and series_index is None, will prompt for selection.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    outputs : dict
        Dictionary containing:
        - nifti_path: Path to converted NIfTI
        - bval_path: Path to b-value file
        - bvec_path: Path to b-vector file
        - report_path: Path to import report
        - subject_id: Subject identifier used
        - series_analysis: SeriesAnalysis object
        
    Raises
    ------
    FileNotFoundError
        If study_dir does not exist
    ValueError
        If no suitable series found and no series_index specified
    RuntimeError
        If conversion fails
        
    Examples
    --------
    >>> # Auto-select best series
    >>> outputs = run_ingest_pipeline("~/anom", "~/output")
    
    >>> # Specify series by index
    >>> outputs = run_ingest_pipeline("~/anom", "~/output", series_index=1)
    
    >>> # With custom subject ID
    >>> outputs = run_ingest_pipeline("~/anom", "~/output", subject_id="sub-01")
    """
    from pathlib import Path
    
    study_dir = Path(study_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    
    if verbose:
        print("=" * 60)
        print("CSTTOOL INGEST PIPELINE")
        print("=" * 60)
    
    # Step 1: Scan for series
    if verbose:
        print("\n[Step 1/5] Scanning for DICOM series...")
    
    series_list = scan_study(study_dir, verbose=verbose)
    
    if not series_list:
        raise FileNotFoundError(f"No DICOM series found in {study_dir}")
    
    # Step 2: Analyze all series
    if verbose:
        print("\n[Step 2/5] Analyzing series...")
    
    analyses = analyze_all_series(series_list, verbose=verbose)
    
    # Step 3: Select series
    if verbose:
        print("\n[Step 3/5] Selecting series...")
    
    selected = None
    
    if series_uid:
        # Use specified UID
        selected = next((s for s in analyses if s.uid == series_uid), None)
        if selected:
            if verbose:
                print(f"  Using series with UID {series_uid}: {selected.name}")
        else:
             raise ValueError(f"Series with UID {series_uid} not found in {study_dir}")
            
    elif series_index is not None:
        # Use specified index
        if 1 <= series_index <= len(analyses):
            selected = analyses[series_index - 1]
            if verbose:
                print(f"  Using series {series_index}: {selected.name}")
        else:
            raise ValueError(
                f"Invalid series_index {series_index}. "
                f"Valid range: 1-{len(analyses)}"
            )
    elif auto_select:
        # Auto-select best
        selected = recommend_series(analyses, verbose=verbose)
        if selected is None:
            raise ValueError(
                "No suitable series found for tractography. "
                "Use series_index or series_uid to manually select a series."
            )
    else:
        # Interactive selection would go here
        # For now, fall back to auto-select
        selected = recommend_series(analyses, verbose=verbose)
        if selected is None:
            raise ValueError("No suitable series found")
    
    # Step 4: Convert
    if verbose:
        print("\n[Step 4/5] Converting DICOM to NIfTI...")
    
    # Create temp conversion directory
    convert_dir = output_dir / "temp_convert"
    convert_dir.mkdir(parents=True, exist_ok=True)
    
    conversion_result = convert_dicom_to_nifti(
        selected.path,
        convert_dir,
        verbose=verbose
    )
    
    if not conversion_result['success']:
        raise RuntimeError("DICOM conversion failed")
    
    # Validate
    validation = validate_conversion(
        conversion_result['nifti_path'],
        conversion_result['bval_path'],
        conversion_result['bvec_path'],
        verbose=verbose
    )
    
    if not validation['valid']:
        if verbose:
            print("\n⚠️  Conversion validation warnings:")
            for issue in validation['issues']:
                print(f"    - {issue}")
    
    # Step 5: Save organized outputs
    if verbose:
        print("\n[Step 5/5] Organizing outputs...")
    
    outputs = save_ingest_outputs(
        conversion_result,
        selected,
        output_dir,
        subject_id=subject_id,
        verbose=verbose
    )
    
    # Add analysis to outputs
    outputs['series_analysis'] = selected
    outputs['validation'] = validation
    
    # Extract acquisition metadata for pipeline
    try:
        from dipy.io.gradients import read_bvals_bvecs
        from nibabel import load as nib_load
        from .modules.assess_quality import extract_acquisition_metadata
        
        # Load bvals/bvecs
        bvals, bvecs = read_bvals_bvecs(
            str(outputs['bval_path']),
            str(outputs['bvec_path'])
        )
        
        # Load NIfTI header for voxel size
        nii_img = nib_load(str(outputs['nifti_path']))
        voxel_size = tuple(float(v) for v in nii_img.header.get_zooms()[:3])
        
        # Check for BIDS JSON sidecar
        bids_json = None
        json_path = outputs['nifti_path'].with_suffix('.json')
        if json_path.exists():
            import json
            with open(json_path) as f:
                bids_json = json.load(f)
        
        # Build CLI overrides
        overrides = {}
        if field_strength is not None:
            overrides['field_strength_T'] = field_strength
        if echo_time is not None:
            overrides['echo_time_ms'] = echo_time
        
        # Extract metadata
        acquisition = extract_acquisition_metadata(
            bvecs=bvecs,
            bvals=bvals,
            voxel_size=voxel_size,
            bids_json=bids_json,
            overrides=overrides
        )
        
        outputs['metadata'] = acquisition
        
        if verbose:
            print(f"  ✓ Extracted acquisition metadata")
            
    except Exception as e:
        if verbose:
            print(f"  Note: Could not extract acquisition metadata: {e}")
        outputs['metadata'] = {}
    
    # Print summary
    if verbose:
        print_import_summary(outputs, selected)
    
    # Cleanup temp directory
    import shutil
    if convert_dir.exists():
        shutil.rmtree(convert_dir)
    
    return outputs


__all__ = [
    # Main pipeline
    'run_ingest_pipeline',
    
    # Scanning
    'is_dicom_directory',
    'scan_study',
    'print_series_summary',
    
    # Analysis
    'SeriesType',
    'SeriesAnalysis',
    'analyze_series',
    'analyze_all_series',
    'recommend_series',
    
    # Conversion
    'convert_dicom_to_nifti',
    'validate_conversion',
    
    # Output
    'save_ingest_outputs',
    'print_import_summary',
    'get_nifti_stem',
    
    # Quality Assessment
    'assess_acquisition_quality',
    'extract_acquisition_metadata',
]