"""
Tracking pipeline modules.

Exports the main pipeline functions for tractography and visualization.
"""

from .load_and_mask import load_and_mask
from .fit_tensors import fit_tensors
from .estimate_directions import estimate_directions
from .seed_and_stop import seed_and_stop
from .run_tractography import run_tractography
from .save_tracking_outputs import save_tracking_outputs

from .visualizations import (
    plot_tensor_maps,
    plot_white_matter_mask,
    plot_streamlines_2d,
    plot_streamline_statistics,
    create_tracking_summary,
    save_all_tracking_visualizations,
)

__all__ = [
    # Pipeline functions
    'load_and_mask',
    'fit_tensors',
    'estimate_directions',
    'seed_and_stop',
    'run_tractography',
    'save_tracking_outputs',
    # Visualization functions
    'plot_tensor_maps',
    'plot_white_matter_mask',
    'plot_streamlines_2d',
    'plot_streamline_statistics',
    'create_tracking_summary',
    'save_all_tracking_visualizations',
]