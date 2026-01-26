import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from csttool.validation.bundle_comparison import (
    check_spatial_compatibility,
    SpatialMismatchError,
    compute_bundle_overlap,
    compute_overreach,
    compute_coverage,
    mean_closest_distance,
    streamline_count_ratio,
)

@pytest.fixture
def temp_data_dir(tmp_path):
    return tmp_path

@pytest.fixture
def ref_anatomy(temp_data_dir):
    # Create valid ref anatomy
    affine = np.eye(4)
    data = np.zeros((10, 10, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, affine)
    path = temp_data_dir / "ref.nii.gz"
    nib.save(img, path)
    return path, affine, (10, 10, 10)

def create_bundle(path, streamlines, affine, ref_nii=None):
    if ref_nii:
        ref_input = str(ref_nii)
    else:
        # Create a dummy header if no ref provided (should not happen in these tests)
        header = nib.Nifti1Header()
        header['dim'][1:4] = [10, 10, 10]
        header.set_sform(affine)
        header.set_qform(affine)
        ref_input = header
        
    sft = StatefulTractogram(streamlines, ref_input, Space.RASMM)
    save_trk(sft, str(path), bbox_valid_check=False)

def test_spatial_compatibility_success(temp_data_dir, ref_anatomy):
    ref_path, affine, _ = ref_anatomy
    cand_trk = temp_data_dir / "cand.trk"
    ref_trk = temp_data_dir / "ref.trk"
    
    streamlines = [np.array([[0,0,0], [1,1,1]])]
    create_bundle(cand_trk, streamlines, affine, ref_path)
    create_bundle(ref_trk, streamlines, affine, ref_path)
    
    # Should pass
    check_spatial_compatibility(cand_trk, ref_trk, ref_path)

def test_spatial_compatibility_fail_translation(temp_data_dir, ref_anatomy):
    ref_path, base_aff, _ = ref_anatomy
    cand_trk = temp_data_dir / "cand_bad.trk"
    ref_trk = temp_data_dir / "ref.trk"
    
    # Modified affine
    bad_aff = base_aff.copy()
    bad_aff[0, 3] = 2.0  # Shift 2mm > 1.0mm tol
    
    # Needs to hack save_trk to force different affine 
    # StatefulTractogram enforces consistency. 
    # We can create a header with different affine manually.
    # Actually, simpler to create a fake NIfTI with bad affine and use it as ref for the TRK
    bad_ref_img = temp_data_dir / "bad_ref.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10,10,10)), bad_aff), bad_ref_img)
    
    streamlines = [np.array([[0,0,0], [1,1,1]])]
    create_bundle(cand_trk, streamlines, bad_aff, bad_ref_img)
    # Ref trk is fine
    create_bundle(ref_trk, streamlines, base_aff, ref_path)
    
    with pytest.raises(SpatialMismatchError, match="translation mismatch"):
        check_spatial_compatibility(cand_trk, ref_trk, ref_path)

def test_compute_bundle_overlap_identical(temp_data_dir, ref_anatomy):
    ref_path, _, _ = ref_anatomy
    path = temp_data_dir / "bundle.trk"
    # Streamline crossing voxel (1,1,1)
    streamlines = [np.array([[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])]
    create_bundle(path, streamlines, np.eye(4), ref_path)
    
    metrics = compute_bundle_overlap(path, path, ref_path)
    assert metrics['dice'] == 1.0
    assert metrics['intersection_volume'] > 0

def test_compute_bundle_overlap_disjoint(temp_data_dir, ref_anatomy):
    ref_path, _, _ = ref_anatomy
    p1 = temp_data_dir / "b1.trk"
    p2 = temp_data_dir / "b2.trk"
    
    s1 = [np.array([[1,1,1], [1.1, 1.1, 1.1]])] # Voxel 1,1,1
    s2 = [np.array([[5,5,5], [5.1, 5.1, 5.1]])] # Voxel 5,5,5
    
    create_bundle(p1, s1, np.eye(4), ref_path)
    create_bundle(p2, s2, np.eye(4), ref_path)
    
    metrics = compute_bundle_overlap(p1, p2, ref_path)
    assert metrics['dice'] == 0.0

def test_empty_candidates(temp_data_dir, ref_anatomy):
    ref_path, _, _ = ref_anatomy
    empty = temp_data_dir / "empty.trk"
    full_path = temp_data_dir / "full.trk"
    
    create_bundle(empty, [], np.eye(4), ref_path)
    create_bundle(full_path, [np.array([[1,1,1],[2,2,2]])], np.eye(4), ref_path)
    
    # Cand empty, Ref full -> Dice 0
    m1 = compute_bundle_overlap(empty, full_path, ref_path)
    assert m1['dice'] == 0.0
    
    # Ref empty -> Dice NaN
    m2 = compute_bundle_overlap(full_path, empty, ref_path)
    assert np.isnan(m2['dice'])

def test_coverage_and_overreach(temp_data_dir, ref_anatomy):
    ref_path, _, _ = ref_anatomy
    # Candidate is subset of Reference
    cand_s = [np.array([[1.5,1.5,1.5]])]
    ref_s = [np.array([[1.5,1.5,1.5]]), np.array([[3.5,3.5,3.5]])] 
    
    cand_p = temp_data_dir / "cand.trk"
    ref_p = temp_data_dir / "ref.trk"
    create_bundle(cand_p, cand_s, np.eye(4), ref_path)
    create_bundle(ref_p, ref_s, np.eye(4), ref_path)
    
    # Coverage: Cand covers 1 of 2 ref voxels -> 0.5
    cov = compute_coverage(cand_p, ref_p, ref_path)
    assert cov['coverage'] == 0.5
    
    # Overreach: Cand has 1 voxel, which is in Ref -> 0.0
    over = compute_overreach(cand_p, ref_p, ref_path)
    assert over['overreach'] == 0.0

def test_mdf_symmetric(temp_data_dir, ref_anatomy):
    ref_path, _, _ = ref_anatomy
    p1 = temp_data_dir / "p1.trk"
    p2 = temp_data_dir / "p2.trk"
    
    # Two parallel lines distance 2.0mm
    s1 = [np.array([[0,0,0], [10,0,0]])]
    s2 = [np.array([[0,2,0], [10,2,0]])]
    
    create_bundle(p1, s1, np.eye(4), ref_path)
    create_bundle(p2, s2, np.eye(4), ref_path)
    
    mdf = mean_closest_distance(p1, p2, step_size_mm=1.0)
    # Distance should be exactly 2.0 everywhere
    assert np.isclose(mdf['mdf_symmetric'], 2.0, atol=0.1)
