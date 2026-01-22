import os
import shutil
import logging
from pathlib import Path
from typing import List, Literal, Optional, Any
from dataclasses import dataclass

from .locking import acquire_batch_lock, release_lock, LockError
from .discover import sanitize_subject_id

logger = logging.getLogger(__name__)

@dataclass
class PreflightError:
    """Represents an error found during batch preflight validation."""
    category: Literal["SYSTEM", "INPUT", "CONFIG"]
    message: str
    subject_id: Optional[str] = None

def validate_batch_preflight(subjects: List[Any], config: Any) -> List[PreflightError]:
    """
    Run comprehensive preflight checks before starting batch processing.
    
    Checks:
    - Environment (dependencies)
    - Concurrency (batch lock)
    - Output directory (writable, disk space)
    - Inputs (existence, duplicates, BIDS naming)
    
    Returns:
        List of PreflightError objects. Empty if all checks pass.
    """
    errors = []
    
    # 1. Environment Check
    # We reuse the existing cmd_check logic
    try:
        from csttool.cli.commands.check import cmd_check
        import argparse
        import sys
        from io import StringIO
        
        # Capture stdout to avoid cluttering during internal validation
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            check_ok = cmd_check(argparse.Namespace())
        finally:
            sys.stdout = old_stdout
            
        if not check_ok:
            errors.append(PreflightError(
                "SYSTEM", 
                "Required dependencies are missing. Run 'csttool check' for details."
            ))
    except Exception as e:
        errors.append(PreflightError("SYSTEM", f"Failed to run environment check: {e}"))
    
    # 2. Concurrent Batch Lock
    try:
        lock = acquire_batch_lock(config.out)
        release_lock(lock)
    except LockError as e:
        errors.append(PreflightError("SYSTEM", str(e)))
    except Exception as e:
        # If output dir doesn't exist yet, locking might fail
        pass
        
    # 3. Output Directory
    out_dir = Path(config.out)
    if out_dir.exists():
        if not os.access(out_dir, os.W_OK):
            errors.append(PreflightError("SYSTEM", f"Output directory is not writable: {out_dir}"))
    else:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(PreflightError("SYSTEM", f"Cannot create output directory: {e}"))
            
    # Heuristic disk space check
    if out_dir.exists():
        try:
            _, _, free = shutil.disk_usage(out_dir)
            # Estimate 2GB per subject as a safe heuristic for CST pipeline
            estimated_required = len(subjects) * 2 * 1024 * 1024 * 1024
            if free < estimated_required:
                errors.append(PreflightError(
                    "SYSTEM", 
                    f"Insufficient disk space. Estimated {estimated_required/1e9:.1f} GB required, "
                    f"{free/1e9:.1f} GB available."
                ))
        except Exception:
            pass # Could not check disk usage - not fatal
            
    # 4. Input Validation
    seen_combinations = set()
    for s in subjects:
        # BIDS naming check
        try:
            sanitize_subject_id(s.subject_id)
        except ValueError as e:
            errors.append(PreflightError("INPUT", str(e), s.subject_id))
            
        # Duplicate subject/session check
        key = (s.subject_id, getattr(s, 'session_id', None))
        if key in seen_combinations:
            errors.append(PreflightError(
                "CONFIG", 
                f"Duplicate subject/session combination: {s.subject_id}/{key[1]}", 
                s.subject_id
            ))
        seen_combinations.add(key)
        
        # File/Directory existence
        if s.input_type == "nifti":
            if not s.input_path.exists():
                errors.append(PreflightError("INPUT", f"NIfTI file not found: {s.input_path}", s.subject_id))
            if s.bval_path and not s.bval_path.exists():
                errors.append(PreflightError("INPUT", f"bval file not found: {s.bval_path}", s.subject_id))
            if s.bvec_path and not s.bvec_path.exists():
                errors.append(PreflightError("INPUT", f"bvec file not found: {s.bvec_path}", s.subject_id))
        elif s.input_type == "dicom":
            if not s.input_path.is_dir():
                errors.append(PreflightError("INPUT", f"DICOM directory not found: {s.input_path}", s.subject_id))
            # Optional: deep check of DICOM files count? 
            # already partially handled in discover.detect_input_type
            
    return errors
