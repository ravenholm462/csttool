"""
Ingest pipeline modules.

Exports the main functions for DICOM import and conversion.
"""

from .scan_study import (
    is_dicom_file,
    is_dicom_directory,
    count_dicom_files,
    scan_study,
    filter_series_by_file_count,
    print_series_summary
)

from .analyze_series import (
    SeriesType,
    SeriesAnalysis,
    analyze_series,
    analyze_all_series,
    recommend_series
)

from .convert_series import (
    convert_dicom_to_nifti,
    validate_conversion,
    convert_with_fallback
)

from .save_ingest_outputs import (
    save_ingest_outputs,
    print_import_summary,
    get_nifti_stem
)


__all__ = [
    # scan_study
    'is_dicom_file',
    'is_dicom_directory',
    'count_dicom_files',
    'scan_study',
    'filter_series_by_file_count',
    'print_series_summary',
    
    # analyze_series
    'SeriesType',
    'SeriesAnalysis',
    'analyze_series',
    'analyze_all_series',
    'recommend_series',
    
    # convert_series
    'convert_dicom_to_nifti',
    'validate_conversion',
    'convert_with_fallback',
    
    # save_ingest_outputs
    'save_ingest_outputs',
    'print_import_summary',
    'get_nifti_stem',
]