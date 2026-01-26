import argparse
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from unittest.mock import MagicMock, patch

from csttool.cli.commands.validate import cmd_validate

# --- Helper ---
def create_bundle(path, streamlines, affine, ref_nii=None):
    if ref_nii:
        ref_input = str(ref_nii)
    else:
        header = nib.Nifti1Header()
        header['dim'][1:4] = [10, 10, 10]
        header.set_sform(affine)
        header.set_qform(affine)
        ref_input = header
        
    sft = StatefulTractogram(streamlines, ref_input, Space.RASMM)
    save_trk(sft, str(path), bbox_valid_check=False)

def create_ref_img(path, affine):
    data = np.zeros((10, 10, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)
    return path

@pytest.fixture
def validation_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    
    affine = np.eye(4)
    ref_img = create_ref_img(d / "ref_space.nii.gz", affine)
    
    sl = [np.array([[1,1,1], [2,2,2]])]
    
    cand_l = d / "cand_left.trk"
    cand_r = d / "cand_right.trk"
    ref_l = d / "ref_left.trk"
    ref_r = d / "ref_right.trk"
    
    create_bundle(cand_l, sl, affine, ref_img)
    create_bundle(cand_r, sl, affine, ref_img)
    create_bundle(ref_l, sl, affine, ref_img)
    create_bundle(ref_r, sl, affine, ref_img)
    
    return {
        "root": d,
        "ref_space": ref_img,
        "cand_l": cand_l,
        "cand_r": cand_r,
        "ref_l": ref_l,
        "ref_r": ref_r,
        "output": tmp_path / "output"
    }

def test_cmd_validate_success(validation_data):
    d = validation_data
    args = argparse.Namespace(
        cand_left=d["cand_l"],
        cand_right=d["cand_r"],
        ref_left=d["ref_l"],
        ref_right=d["ref_r"],
        ref_space=d["ref_space"],
        output_dir=d["output"],
        visualize=False,
        disable_hemisphere_check=False
    )
    
    ret = cmd_validate(args)
    assert ret == 0
    assert (d["output"] / "validation_report.json").exists()

def test_cmd_validate_missing_files(validation_data):
    d = validation_data
    args = argparse.Namespace(
        cand_left=d["root"] / "missing.trk",
        cand_right=d["cand_r"],
        ref_left=d["ref_l"],
        ref_right=d["ref_r"],
        ref_space=d["ref_space"],
        output_dir=d["output"],
        visualize=False,
        disable_hemisphere_check=False
    )
    
    ret = cmd_validate(args)
    assert ret == 1

def test_cmd_validate_spatial_mismatch(validation_data):
    d = validation_data
    
    # Create bad candidate
    bad_path = d["root"] / "bad_affine.trk"
    bad_aff = np.eye(4)
    bad_aff[0, 3] = 100.0 # Huge shift
    
    # Trick create_bundle to save with bad affine by passing header with bad affine
    # StatefulTractogram requires consistency with reference. 
    # Create fake ref for the BAD bundle
    bad_ref_path = d["root"] / "bad_ref_space.nii.gz"
    create_ref_img(bad_ref_path, bad_aff)
    
    create_bundle(bad_path, [np.array([[0,0,0]])], bad_aff, bad_ref_path)
    
    args = argparse.Namespace(
        cand_left=bad_path, # Mismatched against ref_space (identity)
        cand_right=d["cand_r"],
        ref_left=d["ref_l"],
        ref_right=d["ref_r"],
        ref_space=d["ref_space"],
        output_dir=d["output"],
        visualize=False,
        disable_hemisphere_check=True
    )
    
    ret = cmd_validate(args)
    assert ret == 1


@patch("csttool.cli.commands.validate.save_overlap_maps")
@patch("csttool.cli.commands.validate.save_validation_snapshots")
def test_visualization_call(mock_snaps, mock_maps, validation_data):
    # Only test if logic calls visualization
    # We mock visualization functions because they require nilearn which is heavy
    
    d = validation_data
    args = argparse.Namespace(
        cand_left=d["cand_l"],
        cand_right=d["cand_r"],
        ref_left=d["ref_l"],
        ref_right=d["ref_r"],
        ref_space=d["ref_space"],
        output_dir=d["output"],
        visualize=True, # Active
        disable_hemisphere_check=True
    )
    
    # We need to make sure VISUALIZATION_AVAILABLE is True inside validate module
    # or ensure our environment supports it. 
    # We can patch the visualizer imports?
    # Actually, validate.py does generic import. 
    # If nilearn is installed in enviroment, it works. 
    # csttool likely has it.
    
    with patch("csttool.cli.commands.validate.VISUALIZATION_AVAILABLE", True):
        ret = cmd_validate(args)
    
    assert ret == 0
    assert mock_maps.call_count == 2 # Called for Left and Right
    assert mock_snaps.call_count == 2
