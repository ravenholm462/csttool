"""
funcs.py

Utility functions for csttool's metric analysis pipeline.

"""

# General imports
import numpy as np

# Analysis imports
from dipy.tracking.streamline import length, Streamlines
from dipy.tracking.utils import density_map

def analyze_cst_bundle(
        streamlines,
        fa_map=None,
        md_map=None,
        affine=None
):
    """Analysis of a single bundle.

    Args:
        streamlines (_type_): _description_
        fa_map (_type_, optional): _description_. Defaults to None.
        md_map (_type_, optional): _description_. Defaults to None.
        affine (_type_, optional): _description_. Defaults to None.
    """

    metrics = {}

    # Morphology of the CST
    metrics["morphology"] = {
        "n_streamlines": len(streamlines),
        "mean_length": compute_streamline_length(streamlines)["mean_length"],
        "tract_volume": compute_tract_volume(streamlines)  # TO-DO
    }

    if fa_map is not None:
        metrics["fa"] = {
            "mean": mean_fa_along_tract(streamlines, fa_map),  # TO-DO
            "std": std_fa_along_tract(streamlines, fa_map),  # TO-DO
            "profile": compute_tract_profile(streamlines, fa_map, md_map=None)  # TO-DO
        }

    if md_map is not None:
        metrics["md"] = {
            "mean": mean_md_along_tract(streamlines, md_map),  # TO-DO
            "profile": compute_tract_profile(streamlines, md_map, fa_map=None)  # TO-DO, same as above
        }

    return metrics


def compute_streamline_length(streamlines):
    """Computes streamline length values

    Args:
        streamlines (Streamlines): input streamlines

    Returns:
        dict: Dictionary of length statistics and streamlines.
    """

    vals = {}

    if len(streamlines) == 0:
        vals

    length_array = np.array([len(s) for s in streamlines])

    vals["mean_length"] = float(np.mean(length_array))
    vals["std_length"] = float(np.std(length_array))
    vals["min_length"] = float(np.min(length_array))
    vals["max_length"] = float(np.max(length_array))
    vals["n_streamlines"] = len(length_array)
    vals["lengths"] = length_array

    return vals


def compute_tract_volume(streamlines, affine, voxel_size=None):

    # If no voxel size is given, compute
    if voxel_size is None:
        voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))

    density = density_map(
        streamlines=streamlines,
        affine=affine,
        shape=shape
    )

    voxel_volume = np.prod(voxel_size)
    volume = np.sum(density>0)*voxel_volume

    return density, volume


def mean_fa_along_tract(streamline, fa_map):

    pass


def std_fa_along_tract(streamlines, fa_map):

    pass


def mean_md_along_tract(streamlines, md_map):

    pass


def compute_tract_profile(streamlines, fa_map, md_map):

    pass