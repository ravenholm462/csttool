"""
Preprocessing modules.

Exports visualization functions for preprocessing QC.
"""

from .visualizations import (
    plot_denoising_comparison,
    plot_brain_mask_overlay,
    plot_motion_correction_summary,
    create_preprocessing_summary,
    save_all_preprocessing_visualizations,
)

__all__ = [
    'plot_denoising_comparison',
    'plot_brain_mask_overlay',
    'plot_motion_correction_summary',
    'create_preprocessing_summary',
    'save_all_preprocessing_visualizations',
]