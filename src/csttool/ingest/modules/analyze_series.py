"""
analyze_series.py

Analyze DICOM series to determine suitability for diffusion tractography.

This module provides:
- DICOM metadata extraction
- Series classification (raw DWI, derived, structural, etc.)
- Suitability scoring for tractography
- Recommendations for series selection
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings


class SeriesType(Enum):
    """Classification of DICOM series types."""
    DIFFUSION_RAW = "diffusion_raw"          # Raw multi-direction DWI
    DIFFUSION_TRACE = "diffusion_trace"       # Trace-weighted (derived)
    DIFFUSION_ADC = "diffusion_adc"           # ADC map (derived)
    DIFFUSION_FA = "diffusion_fa"             # FA map (derived)
    DIFFUSION_OTHER = "diffusion_other"       # Other diffusion-derived
    STRUCTURAL_T1 = "structural_t1"           # T1-weighted
    STRUCTURAL_T2 = "structural_t2"           # T2-weighted
    LOCALIZER = "localizer"                   # Scout/localizer
    UNKNOWN = "unknown"                       # Cannot determine


@dataclass
class SeriesAnalysis:
    """Results of DICOM series analysis."""
    path: Path
    name: str
    n_files: int
    
    # DICOM metadata
    series_description: str = ""
    uid: str = ""
    series_number: int = 0
    image_type: List[str] = field(default_factory=list)
    modality: str = ""
    
    # Image properties
    rows: int = 0
    columns: int = 0
    n_slices: int = 0
    n_volumes_estimated: int = 0
    
    # Diffusion-specific
    b_values: List[float] = field(default_factory=list)
    n_directions: int = 0
    phase_encoding_direction: str = ""
    
    # Classification
    series_type: SeriesType = SeriesType.UNKNOWN
    is_derived: bool = False
    is_original: bool = False
    
    # Suitability assessment
    suitable_for_tractography: bool = False
    suitability_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""


def analyze_series(
    series_path: Path,
    verbose: bool = True
) -> SeriesAnalysis:
    """
    Analyze a DICOM series for tractography suitability.
    
    Extracts metadata from DICOM headers and classifies the series
    based on image type, acquisition parameters, and derived status.
    
    Parameters
    ----------
    series_path : Path
        Path to directory containing DICOM files for one series
    verbose : bool, optional
        Print analysis results. Default is True.
        
    Returns
    -------
    analysis : SeriesAnalysis
        Comprehensive analysis results including suitability assessment
        
    Notes
    -----
    Requires pydicom for full functionality. Falls back to basic
    analysis if pydicom is not installed.
    """
    series_path = Path(series_path)
    
    # Initialize analysis
    analysis = SeriesAnalysis(
        path=series_path,
        name=series_path.name,
        n_files=_count_dicom_files(series_path)
    )
    
    # Try to load pydicom
    try:
        import pydicom
        _analyze_with_pydicom(analysis, series_path, pydicom)
    except ImportError:
        analysis.warnings.append(
            "pydicom not installed - limited analysis available. "
            "Install with: pip install pydicom"
        )
        _analyze_without_pydicom(analysis)
    
    # Classify series type
    _classify_series(analysis)
    
    # Assess tractography suitability
    _assess_suitability(analysis)
    
    if verbose:
        _print_analysis(analysis)
    
    return analysis


def _count_dicom_files(directory: Path) -> int:
    """Count DICOM files in directory."""
    dcm_files = list(directory.glob("*.dcm")) + list(directory.glob("*.DCM"))
    if dcm_files:
        return len(dcm_files)
    
    # Count files that might be DICOM (no extension)
    return len([f for f in directory.iterdir() 
                if f.is_file() and not f.name.startswith('.')])


def _analyze_with_pydicom(analysis: SeriesAnalysis, series_path: Path, pydicom) -> None:
    """Extract metadata using pydicom."""
    
    # Find DICOM files
    dcm_files = list(series_path.glob("*.dcm")) + list(series_path.glob("*.DCM"))
    if not dcm_files:
        dcm_files = [f for f in series_path.iterdir() 
                     if f.is_file() and not f.name.startswith('.')]
    
    if not dcm_files:
        analysis.warnings.append("No DICOM files found in directory")
        return
    
    # Read first file for series-level metadata
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    except Exception as e:
        analysis.warnings.append(f"Error reading DICOM: {e}")
        return
    
    # Extract basic metadata
    analysis.series_description = str(getattr(ds, 'SeriesDescription', ''))
    analysis.uid = str(getattr(ds, 'SeriesInstanceUID', ''))
    analysis.series_number = int(getattr(ds, 'SeriesNumber', 0))
    analysis.image_type = list(getattr(ds, 'ImageType', []))
    analysis.modality = str(getattr(ds, 'Modality', ''))
    analysis.rows = int(getattr(ds, 'Rows', 0))
    analysis.columns = int(getattr(ds, 'Columns', 0))
    
    # Check derived status
    analysis.is_derived = ('DERIVED' in analysis.image_type or 
                           'SECONDARY' in analysis.image_type)
    analysis.is_original = ('ORIGINAL' in analysis.image_type or 
                            'PRIMARY' in analysis.image_type)
    
    # Extract diffusion-specific info
    if hasattr(ds, 'InPlanePhaseEncodingDirection'):
        analysis.phase_encoding_direction = str(ds.InPlanePhaseEncodingDirection)
    
    # Scan multiple files for b-values and slice count
    b_values = set()
    slice_locations = set()
    instance_numbers = set()
    
    # Sample files for efficiency
    sample_files = dcm_files[:min(200, len(dcm_files))]
    
    for dcm_file in sample_files:
        try:
            ds_sample = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            
            # Collect b-values
            if hasattr(ds_sample, 'DiffusionBValue'):
                b_values.add(float(ds_sample.DiffusionBValue))
            
            # Collect slice locations
            if hasattr(ds_sample, 'SliceLocation'):
                slice_locations.add(float(ds_sample.SliceLocation))
            
            # Collect instance numbers
            if hasattr(ds_sample, 'InstanceNumber'):
                instance_numbers.add(int(ds_sample.InstanceNumber))
                
        except Exception:
            pass
    
    analysis.b_values = sorted(list(b_values))
    analysis.n_slices = len(slice_locations) if slice_locations else 0
    
    # Estimate number of volumes
    if analysis.n_slices > 0:
        analysis.n_volumes_estimated = analysis.n_files // analysis.n_slices
    
    # Estimate gradient directions from b-values
    if len(analysis.b_values) > 1:
        # Number of non-zero b-value acquisitions suggests directions
        n_dwi = analysis.n_volumes_estimated - analysis.b_values.count(0)
        analysis.n_directions = max(0, n_dwi)


def _analyze_without_pydicom(analysis: SeriesAnalysis) -> None:
    """Basic analysis using only filename patterns."""
    name_upper = analysis.name.upper()
    
    # Infer from directory name
    if 'TRACE' in name_upper:
        analysis.is_derived = True
    if 'ADC' in name_upper:
        analysis.is_derived = True
    if 'FA' in name_upper and 'DIFF' not in name_upper:
        analysis.is_derived = True
    
    if '_AP_' in name_upper or '_AP' in name_upper:
        analysis.phase_encoding_direction = 'AP'
    elif '_PA_' in name_upper or '_PA' in name_upper:
        analysis.phase_encoding_direction = 'PA'
    
    if 'DIFF' in name_upper or 'DWI' in name_upper:
        analysis.series_description = "Diffusion (inferred from name)"


def _classify_series(analysis: SeriesAnalysis) -> None:
    """Classify series type based on metadata."""
    
    desc_upper = analysis.series_description.upper()
    name_upper = analysis.name.upper()
    combined = f"{desc_upper} {name_upper}"
    
    # Check for derived diffusion images first
    if 'TRACEW' in combined or 'TRACE' in combined:
        analysis.series_type = SeriesType.DIFFUSION_TRACE
        return
    
    if 'ADC' in combined:
        analysis.series_type = SeriesType.DIFFUSION_ADC
        return
    
    if '_FA' in combined or 'FA_' in combined or combined.endswith('FA'):
        if 'DIFF' not in combined:  # Avoid matching "DIFF_FA"
            analysis.series_type = SeriesType.DIFFUSION_FA
            return
    
    # Check for raw diffusion
    is_diffusion = ('DIFF' in combined or 'DWI' in combined or 
                    'DTI' in combined or 'HARDI' in combined or
                    len(analysis.b_values) > 1)
    
    if is_diffusion and not analysis.is_derived:
        analysis.series_type = SeriesType.DIFFUSION_RAW
        return
    
    if is_diffusion:
        analysis.series_type = SeriesType.DIFFUSION_OTHER
        return
    
    # Check for structural
    if 'T1' in combined or 'MPRAGE' in combined or 'SPGR' in combined:
        analysis.series_type = SeriesType.STRUCTURAL_T1
        return
    
    if 'T2' in combined or 'FLAIR' in combined:
        analysis.series_type = SeriesType.STRUCTURAL_T2
        return
    
    # Check for localizer
    if 'LOC' in combined or 'SCOUT' in combined or 'SURVEY' in combined:
        analysis.series_type = SeriesType.LOCALIZER
        return
    
    analysis.series_type = SeriesType.UNKNOWN


def _assess_suitability(analysis: SeriesAnalysis) -> None:
    """Assess suitability for diffusion tractography."""
    
    score = 0.0
    
    # Series type scoring
    if analysis.series_type == SeriesType.DIFFUSION_RAW:
        score += 50.0
    elif analysis.series_type == SeriesType.DIFFUSION_TRACE:
        score -= 100.0
        analysis.warnings.append(
            "TRACE-weighted image: Pre-computed scalar, lacks directional "
            "information needed for tractography"
        )
    elif analysis.series_type == SeriesType.DIFFUSION_ADC:
        score -= 100.0
        analysis.warnings.append(
            "ADC map: Derived scalar image, not raw diffusion data"
        )
    elif analysis.series_type == SeriesType.DIFFUSION_FA:
        score -= 100.0
        analysis.warnings.append(
            "FA map: Derived scalar image, not raw diffusion data"
        )
    elif analysis.series_type in [SeriesType.STRUCTURAL_T1, 
                                   SeriesType.STRUCTURAL_T2,
                                   SeriesType.LOCALIZER]:
        score -= 100.0
        analysis.warnings.append(
            f"Not a diffusion sequence: {analysis.series_type.value}"
        )
    
    # Derived image penalty
    if analysis.is_derived:
        score -= 30.0
        if "DERIVED" not in str(analysis.warnings):
            analysis.warnings.append(
                "DERIVED image type - may be pre-processed or computed"
            )
    
    # Original image bonus
    if analysis.is_original:
        score += 10.0
    
    # File count scoring
    if analysis.n_files >= 100:
        score += 20.0
    elif analysis.n_files >= 50:
        score += 10.0
    elif analysis.n_files < 20:
        score -= 10.0
        analysis.warnings.append(
            f"Low file count ({analysis.n_files}) - may be incomplete or derived"
        )
    
    # B-value scoring
    if len(analysis.b_values) >= 2:
        score += 15.0
    if len(analysis.b_values) >= 3:
        score += 5.0
    if 0 in analysis.b_values:  # Has b0
        score += 5.0
    
    # Phase encoding assessment
    pe = analysis.phase_encoding_direction.upper()
    if 'PA' in pe or 'P' == pe:
        analysis.warnings.append(
            "PA phase encoding: Typically a short acquisition for distortion "
            "correction. Consider using the AP series for tractography."
        )
        score -= 10.0
    elif 'AP' in pe or 'A' == pe:
        score += 10.0  # AP is standard primary direction
    
    # Set final assessment
    analysis.suitability_score = score
    analysis.suitable_for_tractography = score > 0
    
    # Generate recommendation
    if score >= 70:
        analysis.recommendation = "✓ Highly recommended for tractography"
    elif score >= 40:
        analysis.recommendation = "✓ Suitable for tractography"
    elif score > 0:
        analysis.recommendation = "⚠️ May be suitable - review warnings"
    else:
        analysis.recommendation = "✗ Not recommended for tractography"


def _print_analysis(analysis: SeriesAnalysis) -> None:
    """Print formatted analysis results."""
    
    print(f"\n{'=' * 60}")
    print(f"Series: {analysis.name}")
    print(f"{'=' * 60}")
    
    print(f"  Description:     {analysis.series_description or 'N/A'}")
    print(f"  Series Number:   {analysis.series_number}")
    print(f"  Files:           {analysis.n_files}")
    print(f"  Matrix:          {analysis.rows} × {analysis.columns}")
    print(f"  Image Type:      {analysis.image_type}")
    print(f"  Classification:  {analysis.series_type.value}")
    
    if analysis.b_values:
        print(f"  B-values:        {analysis.b_values}")
    if analysis.n_volumes_estimated:
        print(f"  Est. Volumes:    {analysis.n_volumes_estimated}")
    if analysis.phase_encoding_direction:
        print(f"  Phase Encoding:  {analysis.phase_encoding_direction}")
    
    print(f"\n  {analysis.recommendation}")
    print(f"  Score: {analysis.suitability_score:.0f}")
    
    if analysis.warnings:
        print(f"\n  Warnings:")
        for w in analysis.warnings:
            print(f"    ⚠️ {w}")


def analyze_all_series(
    series_list: List[Dict],
    verbose: bool = True
) -> List[SeriesAnalysis]:
    """
    Analyze all series in a list.
    
    Parameters
    ----------
    series_list : List[Dict]
        List of series from scan_study()
    verbose : bool
        Print analysis results
        
    Returns
    -------
    analyses : List[SeriesAnalysis]
        Analysis results for each series
    """
    analyses = []
    
    for series_info in series_list:
        analysis = analyze_series(series_info['path'], verbose=verbose)
        analyses.append(analysis)
    
    return analyses


def recommend_series(
    analyses: List[SeriesAnalysis],
    verbose: bool = True
) -> Optional[SeriesAnalysis]:
    """
    Recommend the best series for tractography.
    
    Parameters
    ----------
    analyses : List[SeriesAnalysis]
        Analysis results from analyze_all_series()
    verbose : bool
        Print recommendation
        
    Returns
    -------
    best : SeriesAnalysis or None
        Recommended series, or None if no suitable series found
    """
    # Filter to suitable series
    suitable = [a for a in analyses if a.suitable_for_tractography]
    
    if not suitable:
        if verbose:
            print("\n  ✗ No suitable series found for tractography")
            print("  All series appear to be derived images or non-diffusion data")
        return None
    
    # Sort by score (descending)
    suitable.sort(key=lambda a: a.suitability_score, reverse=True)
    best = suitable[0]
    
    if verbose:
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print(f"\n  Best series: {best.name}")
        print(f"  Score:       {best.suitability_score:.0f}")
        print(f"  Files:       {best.n_files}")
        
        if len(suitable) > 1:
            print(f"\n  Alternative suitable series:")
            for alt in suitable[1:]:
                print(f"    - {alt.name} (score: {alt.suitability_score:.0f})")
        
        # Check for AP/PA pair
        ap_series = [a for a in suitable if 'AP' in a.name.upper()]
        pa_series = [a for a in analyses if 'PA' in a.name.upper()]
        
        if ap_series and pa_series:
            print(f"\n  → AP/PA pair detected")
            print(f"     For distortion correction, use both with FSL topup/eddy")
            print(f"     (not yet implemented in csttool)")
    
    return best