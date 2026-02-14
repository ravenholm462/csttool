"""
Data loading module for csttool atlas and template files.

This module provides a two-tier data access system:
- Tier 1: Bundled data (MNI152) shipped with the package (permissive license)
- Tier 2: User-fetched data (FMRIB58_FA, Harvard-Oxford) downloaded on demand (FSL non-commercial)
"""

from .loader import (
    load_mni152_template,
    get_fmrib58_fa_path,
    get_fmrib58_fa_skeleton_path,
    get_harvard_oxford_path,
    get_user_data_dir,
    is_data_installed,
    DataNotInstalledError,
)

__all__ = [
    "load_mni152_template",
    "get_fmrib58_fa_path",
    "get_fmrib58_fa_skeleton_path",
    "get_harvard_oxford_path",
    "get_user_data_dir",
    "is_data_installed",
    "DataNotInstalledError",
]
