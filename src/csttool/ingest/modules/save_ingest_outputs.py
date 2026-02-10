"""
save_ingest_outputs.py

Save and organize import pipeline outputs.

This module provides:
- Structured output directory organization
- Import report generation
- Metadata preservation
- Consistent file naming
"""

from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import json
import shutil


def save_ingest_outputs(
    conversion_result: Dict,
    series_analysis,  # SeriesAnalysis from analyze_series
    output_dir: Path,
    subject_id: Optional[str] = None,
    copy_to_structured: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Save and organize import outputs with consistent structure.
    
    Creates an organized output structure:
    ```
    output_dir/
    ├── nifti/
    │   ├── {subject_id}.nii.gz
    │   ├── {subject_id}.bval
    │   └── {subject_id}.bvec
    ├── logs/
    │   └── {subject_id}_import_report.json
    └── dicom_info/
        └── {subject_id}_series_info.json
    ```
    
    Parameters
    ----------
    conversion_result : Dict
        Output from convert_dicom_to_nifti()
    series_analysis : SeriesAnalysis
        Analysis from analyze_series()
    output_dir : Path
        Base output directory
    subject_id : str, optional
        Subject identifier. If None, extracted from series name.
    copy_to_structured : bool, optional
        Copy files to structured directory. If False, files remain in
        original locations. Default is True.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    outputs : Dict
        Dictionary containing:
        - nifti_path: Path to NIfTI file
        - bval_path: Path to bval file
        - bvec_path: Path to bvec file
        - report_path: Path to import report
        - output_dir: Base output directory
    """
    output_dir = Path(output_dir)
    
    # Determine subject ID
    if subject_id is None:
        subject_id = _extract_subject_id(series_analysis.name)
    
    if verbose:
        print(f"\nOrganizing outputs for: {subject_id}")
    
    # Create directory structure
    nifti_dir = output_dir / "nifti"
    info_dir = output_dir / "dicom_info"

    nifti_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {
        'subject_id': subject_id,
        'nifti_path': None,
        'bval_path': None,
        'bvec_path': None,
        'report_path': None,
        'output_dir': output_dir
    }
    
    # Copy/move files to structured location
    if copy_to_structured and conversion_result['nifti_path']:
        # NIfTI file
        src_nii = conversion_result['nifti_path']
        dst_nii = nifti_dir / f"{subject_id}.nii.gz"
        shutil.copy2(src_nii, dst_nii)
        outputs['nifti_path'] = dst_nii
        
        if verbose:
            print(f"  ✓ NIfTI: {dst_nii}")
        
        # bval file
        if conversion_result['bval_path']:
            src_bval = conversion_result['bval_path']
            dst_bval = nifti_dir / f"{subject_id}.bval"
            shutil.copy2(src_bval, dst_bval)
            outputs['bval_path'] = dst_bval
            
            if verbose:
                print(f"  ✓ bval:  {dst_bval}")
        
        # bvec file
        if conversion_result['bvec_path']:
            src_bvec = conversion_result['bvec_path']
            dst_bvec = nifti_dir / f"{subject_id}.bvec"
            shutil.copy2(src_bvec, dst_bvec)
            outputs['bvec_path'] = dst_bvec
            
            if verbose:
                print(f"  ✓ bvec:  {dst_bvec}")
    else:
        # Use original locations
        outputs['nifti_path'] = conversion_result['nifti_path']
        outputs['bval_path'] = conversion_result['bval_path']
        outputs['bvec_path'] = conversion_result['bvec_path']
    
    # Save series info
    series_info_path = info_dir / f"{subject_id}_series_info.json"
    _save_series_info(series_analysis, series_info_path)
    
    if verbose:
        print(f"  ✓ Series info: {series_info_path}")
    
    # Save import report
    report_path = nifti_dir / f"{subject_id}_import_report.json"
    _save_import_report(
        conversion_result, series_analysis, outputs, subject_id, report_path
    )
    outputs['report_path'] = report_path
    
    if verbose:
        print(f"  ✓ Report: {report_path}")
    
    return outputs


def _extract_subject_id(series_name: str) -> str:
    """
    Extract a clean subject ID from series name.
    
    Handles common naming patterns like:
    - "cmrr_mbep2d_diff_AP_TDI_Series0017" → "cmrr_diff_AP_0017"
    - "sub-01_dwi" → "sub-01"
    """
    import re
    
    name = series_name
    
    # Remove common verbose parts
    name = re.sub(r'_?Series0*(\d+)', r'_\1', name)  # Series0017 → _17
    name = re.sub(r'mbep2d_?', '', name, flags=re.IGNORECASE)  # Remove mbep2d
    name = re.sub(r'TDI_?', '', name, flags=re.IGNORECASE)  # Remove TDI
    name = re.sub(r'__+', '_', name)  # Multiple underscores → single
    name = name.strip('_')
    
    # Truncate if too long
    if len(name) > 50:
        name = name[:50]
    
    # Ensure valid filename
    name = re.sub(r'[^\w\-]', '_', name)
    
    return name if name else "subject"


def _save_series_info(series_analysis, output_path: Path) -> None:
    """Save series analysis as JSON."""
    
    info = {
        'series_name': series_analysis.name,
        'series_path': str(series_analysis.path),
        'series_description': series_analysis.series_description,
        'series_number': series_analysis.series_number,
        'n_files': series_analysis.n_files,
        'image_type': series_analysis.image_type,
        'modality': series_analysis.modality,
        'matrix_size': [series_analysis.rows, series_analysis.columns],
        'n_slices': series_analysis.n_slices,
        'n_volumes_estimated': series_analysis.n_volumes_estimated,
        'b_values': series_analysis.b_values,
        'n_directions': series_analysis.n_directions,
        'phase_encoding_direction': series_analysis.phase_encoding_direction,
        'series_type': series_analysis.series_type.value,
        'is_derived': series_analysis.is_derived,
        'is_original': series_analysis.is_original,
        'suitability_score': series_analysis.suitability_score,
        'suitable_for_tractography': series_analysis.suitable_for_tractography,
        'warnings': series_analysis.warnings,
        'recommendation': series_analysis.recommendation
    }
    
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)


def _save_import_report(
    conversion_result: Dict,
    series_analysis,
    outputs: Dict,
    subject_id: str,
    output_path: Path
) -> None:
    """Save comprehensive import report."""
    
    report = {
        'processing_info': {
            'date': datetime.now().isoformat(),
            'csttool_module': 'ingest',
            'subject_id': subject_id
        },
        'input': {
            'dicom_directory': str(series_analysis.path),
            'series_name': series_analysis.name,
            'series_description': series_analysis.series_description,
            'n_dicom_files': series_analysis.n_files
        },
        'conversion': {
            'success': conversion_result['success'],
            'warnings': conversion_result['warnings']
        },
        'output_files': {
            'nifti': str(outputs['nifti_path']) if outputs['nifti_path'] else None,
            'bval': str(outputs['bval_path']) if outputs['bval_path'] else None,
            'bvec': str(outputs['bvec_path']) if outputs['bvec_path'] else None
        },
        'data_properties': {
            'b_values': series_analysis.b_values,
            'n_volumes_estimated': series_analysis.n_volumes_estimated,
            'matrix_size': [series_analysis.rows, series_analysis.columns],
            'phase_encoding': series_analysis.phase_encoding_direction
        },
        'quality_assessment': {
            'series_type': series_analysis.series_type.value,
            'suitable_for_tractography': series_analysis.suitable_for_tractography,
            'suitability_score': series_analysis.suitability_score,
            'warnings': series_analysis.warnings
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def print_import_summary(outputs: Dict, series_analysis) -> None:
    """
    Print a formatted summary of the import.
    
    Parameters
    ----------
    outputs : Dict
        Output from save_ingest_outputs()
    series_analysis : SeriesAnalysis
        Analysis from analyze_series()
    """
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    
    print(f"\nSubject ID: {outputs['subject_id']}")
    print(f"Source:     {series_analysis.name}")
    
    print(f"\nData Properties:")
    if series_analysis.b_values:
        print(f"  B-values:    {series_analysis.b_values}")
    if series_analysis.n_volumes_estimated:
        print(f"  Volumes:     ~{series_analysis.n_volumes_estimated}")
    if series_analysis.phase_encoding_direction:
        print(f"  Phase Enc:   {series_analysis.phase_encoding_direction}")
    
    print(f"\nOutput Files:")
    print(f"  NIfTI: {outputs['nifti_path']}")
    if outputs['bval_path']:
        print(f"  bval:  {outputs['bval_path']}")
    if outputs['bvec_path']:
        print(f"  bvec:  {outputs['bvec_path']}")
    print(f"  Report: {outputs['report_path']}")
    
    print(f"\n{series_analysis.recommendation}")
    
    if series_analysis.warnings:
        print(f"\n  Warnings:")
        for w in series_analysis.warnings:
            print(f"  ⚠️ {w}")
    
    print("=" * 60)


def get_nifti_stem(outputs: Dict) -> str:
    """
    Get the stem (filename without extension) for the NIfTI file.
    
    Useful for downstream processing that needs to find associated files.
    
    Parameters
    ----------
    outputs : Dict
        Output from save_ingest_outputs()
        
    Returns
    -------
    stem : str
        Filename without .nii.gz extension
    """
    if outputs['nifti_path']:
        name = outputs['nifti_path'].name
        # Remove .nii.gz or .nii
        if name.endswith('.nii.gz'):
            return name[:-7]
        elif name.endswith('.nii'):
            return name[:-4]
        return name
    return outputs['subject_id']