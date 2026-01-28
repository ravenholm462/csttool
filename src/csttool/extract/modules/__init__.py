"""
Extract pipeline modules.

Exports the main functions for extraction of the left and right CST.
"""

from .registration import (
    load_mni_template,
    compute_affine_registration,
    compute_syn_registration,
    register_mni_to_subject,
    save_registration_report,
    plot_registration_comparison
)

from .visualizations import (
    plot_registration_comparison,
    plot_cst_extraction,
    plot_hemisphere_separation,
    create_extraction_summary,
    save_all_extraction_visualizations,
)

__all__ = [
    'load_mni_template',
    'register_mni_to_subject',
    'save_registration_report',
    'compute_affine_registration',
    'compute_syn_registration',
    'plot_registration_comparison',
    'plot_cst_extraction',
    'plot_hemisphere_separation',
    'create_extraction_summary',
    'save_all_extraction_visualizations',
]
