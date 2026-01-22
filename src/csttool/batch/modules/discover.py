import re
from pathlib import Path
from typing import List, Dict, Optional, Literal, Tuple
import logging

logger = logging.getLogger(__name__)

def sanitize_subject_id(subject_id: str) -> str:
    """
    Sanitize subject ID to prevent path traversal attacks.
    Only allows alphanumeric, hyphen, and underscore.
    """
    if not subject_id:
        raise ValueError("Subject ID cannot be empty")
    
    if not re.match(r'^[a-zA-Z0-9\-_]+$', subject_id):
        raise ValueError(
            f"Invalid subject ID: '{subject_id}'. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )
    return subject_id

def discover_subjects(
    bids_dir: Path,
    subject_pattern: str = "sub-*",
    session_pattern: str = "ses-*",
    include_subjects: Optional[List[str]] = None,
    exclude_subjects: Optional[List[str]] = None
) -> List[Dict]:
    """
    Auto-discover subjects and sessions in a BIDS directory.
    
    Returns:
        List of dicts with 'id', 'session', 'dir'
    """
    subjects = []
    bids_path = Path(bids_dir)
    
    if not bids_path.is_dir():
        raise ValueError(f"BIDS directory not found: {bids_dir}")

    # Iterate over subject directories
    for sub_dir in sorted(bids_path.glob(subject_pattern)):
        if not sub_dir.is_dir():
            continue
            
        sub_id = sub_dir.name
        
        # Apply include/exclude filters
        if include_subjects:
            if not any(re.match(p, sub_id) for p in include_subjects):
                continue
        if exclude_subjects:
            if any(re.match(p, sub_id) for p in exclude_subjects):
                continue
                
        # Check for sessions
        ses_dirs = sorted(sub_dir.glob(session_pattern))
        
        if ses_dirs:
            for ses_dir in ses_dirs:
                if not ses_dir.is_dir():
                    continue
                subjects.append({
                    "id": sub_id,
                    "session": ses_dir.name,
                    "dir": ses_dir
                })
        else:
            # No sessions - check for dwi directory directly under subject
            # or just assume the subject directory is the container
            subjects.append({
                "id": sub_id,
                "session": None,
                "dir": sub_dir
            })
            
    return subjects

def detect_input_type(dwi_dir: Path) -> Tuple[Literal["nifti", "dicom"], Path]:
    """
    Detect whether a directory contains NIfTI or DICOM data.
    Priority: NIfTI > DICOM
    """
    # Look for dwi subfolder first (BIDS convention)
    search_dirs = [dwi_dir / "dwi", dwi_dir]
    
    for s_dir in search_dirs:
        if not s_dir.is_dir():
            continue
            
        # 1. NIfTI check
        nifti_files = sorted(s_dir.glob("*_dwi.nii.gz"))
        if not nifti_files:
            nifti_files = sorted(s_dir.glob("*.nii.gz"))
            
        if nifti_files:
            return "nifti", nifti_files[0]
            
        # 2. DICOM check (look for .dcm files)
        if any(s_dir.glob("*.dcm")) or any(s_dir.glob("*.DCM")):
            return "dicom", s_dir
            
        # Check for extensionless DICOMs
        for f in s_dir.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                # Basic magic number check without pydicom dependency if possible
                try:
                    with open(f, 'rb') as fd:
                        fd.seek(128)
                        if fd.read(4) == b'DICM':
                            return "dicom", s_dir
                except Exception:
                    pass
                break
                
    raise FileNotFoundError(f"No NIfTI or DICOM data found in {dwi_dir}")

def find_bval_bvec(nifti_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find matching .bval and .bvec files for a NIfTI file."""
    # Try typical BIDS naming: sub-X_ses-Y_dwi.nii.gz -> sub-X_ses-Y_dwi.bval
    base = str(nifti_path).replace('.nii.gz', '').replace('.nii', '')
    bval = Path(f"{base}.bval")
    bvec = Path(f"{base}.bvec")
    
    return (bval if bval.exists() else None, 
            bvec if bvec.exists() else None)

def find_json_sidecar(nifti_path: Path) -> Optional[Path]:
    """Find matching .json sidecar for a NIfTI file."""
    base = str(nifti_path).replace('.nii.gz', '').replace('.nii', '')
    json_path = Path(f"{base}.json")
    return json_path if json_path.exists() else None

def detect_dicom_series(dicom_dir: Path) -> List[Dict]:
    """
    Scan DICOM directory and return unique series information.
    Requires pydicom.
    """
    try:
        import pydicom
    except ImportError:
        logger.warning("pydicom not installed; cannot detect DICOM series.")
        return []
        
    series = {}
    
    # Scan files to group by series
    for f in dicom_dir.iterdir():
        if not f.is_file() or f.name.startswith('.'):
            continue
            
        try:
            # stop_before_pixels=True makes this much faster
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            uid = getattr(ds, 'SeriesInstanceUID', None)
            if not uid:
                continue
                
            if uid not in series:
                series[uid] = {
                    "uid": uid,
                    "description": str(getattr(ds, 'SeriesDescription', 'Unknown')),
                    "number": int(getattr(ds, 'SeriesNumber', 0)),
                    "file_count": 0
                }
            series[uid]["file_count"] += 1
        except Exception:
            continue
            
    return sorted(list(series.values()), key=lambda x: x['number'])

def validate_single_series(dicom_dir: Path, series_uid: Optional[str] = None) -> str:
    """
    Ensure unambiguous DICOM series selection.
    If multiple series exist and no UID is provided, raises ValueError.
    """
    series_list = detect_dicom_series(dicom_dir)
    
    if not series_list:
        raise ValueError(f"No valid DICOM series found in {dicom_dir}")
        
    if series_uid:
        if any(s['uid'] == series_uid for s in series_list):
            return series_uid
        raise ValueError(f"Specified Series UID '{series_uid}' not found in {dicom_dir}")
        
    if len(series_list) > 1:
        msg = f"Multiple DICOM series found in {dicom_dir}:\n"
        for s in series_list:
            msg += f"  - {s['uid']} ({s['description']}, {s['file_count']} files)\n"
        msg += "Specify 'series_uid' in manifest to select one."
        raise ValueError(msg)
        
    return series_list[0]['uid']
