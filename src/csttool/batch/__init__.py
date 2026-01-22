"""
Batch processing package for csttool.

Provides multi-subject batch processing capabilities with BIDS-compliant
input discovery and output organization.
"""

from .batch import run_batch, BatchConfig, SubjectSpec, SubjectResult
from .modules.discover import discover_subjects, detect_input_type
from .modules.manifest import load_manifest, save_manifest_template
from .modules.report import generate_batch_reports

__all__ = [
    "run_batch",
    "BatchConfig",
    "SubjectSpec",
    "SubjectResult",
    "discover_subjects",
    "detect_input_type",
    "load_manifest",
    "save_manifest_template",
    "generate_batch_reports",
]
