"""
funcs.py

Utility functions for csttool's metric analysis pipeline.

"""

# Imports

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
        "mean_length": mean_streamline_length(streamlines),  # TO-DO
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


def mean_streamline_length(streamlines):

    pass


def compute_tract_volume(streamlines):

    pass


def mean_fa_along_tract(streamline, fa_map):

    pass


def std_fa_along_tract(streamlines, fa_map):

    pass


def mean_md_along_tract(streamlines, md_map):

    pass


def compute_tract_profile(streamlines, fa_map, md_map):

    pass