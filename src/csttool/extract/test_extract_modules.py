"""
test_registration.py

Complete test script for csttool's registration module.

Usage:
    python test_registration.py
"""

from time import time
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input: Subject FA map from tracking pipeline
SUBJECT_FA_PATH = "/home/alem/Documents/thesis/out/trk/scalar_maps/csttool_dwi.nii_fa.nii.gz"

# Output directory for registration results
OUTPUT_DIR = "/home/alem/Documents/thesis/out/extract/registration"

# Use reduced iterations for faster testing (set to False for full registration)
FAST_TEST = True

# =============================================================================
# IMPORTS
# =============================================================================

from csttool.extract.modules.registration import (
    load_mni_template,
    compute_affine_registration,
    compute_syn_registration,
    register_mni_to_subject,
    save_registration_report,
    plot_registration_comparison
)

# =============================================================================
# SETUP
# =============================================================================

print("=" * 70)
print("CSTTOOL REGISTRATION MODULE - TEST SCRIPT")
print("=" * 70)

# Verify input exists
subject_fa_path = Path(SUBJECT_FA_PATH)
if not subject_fa_path.exists():
    print(f"✗ ERROR: Subject FA not found: {subject_fa_path}")
    print("  Please update SUBJECT_FA_PATH in this script")
    exit(1)

# Create output directory
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nInput:  {subject_fa_path}")
print(f"Output: {output_dir}")

# Set iteration parameters based on test mode
if FAST_TEST:
    print("\n⚡ FAST TEST MODE: Using reduced iterations")
    LEVEL_ITERS_AFFINE = [1000, 100, 10]
    LEVEL_ITERS_SYN = [5, 5, 3]
else:
    print("\n🐢 FULL TEST MODE: Using production iterations (slower)")
    LEVEL_ITERS_AFFINE = [10000, 1000, 100]
    LEVEL_ITERS_SYN = [10, 10, 5]

# =============================================================================
# TEST 1: Load MNI Template
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: load_mni_template()")
print("=" * 70)

t0 = time()
mni_img, mni_data, mni_affine = load_mni_template(contrast="T1")

print(f"\n    MNI shape: {mni_data.shape}")
print(f"    MNI dtype: {mni_data.dtype}")
print(f"    MNI range: [{mni_data.min():.2f}, {mni_data.max():.2f}]")
print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 1 PASSED")

# =============================================================================
# TEST 2: Load Subject FA
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: Load Subject FA Map")
print("=" * 70)

t0 = time()
subject_img = nib.load(subject_fa_path)
subject_data = subject_img.get_fdata()
subject_affine = subject_img.affine

voxel_size = np.sqrt(np.sum(subject_affine[:3, :3]**2, axis=0))

print(f"\n    Subject shape: {subject_data.shape}")
print(f"    Subject dtype: {subject_data.dtype}")
print(f"    Subject voxel size: {voxel_size.round(3)} mm")
print(f"    FA range: [{subject_data.min():.3f}, {subject_data.max():.3f}]")
print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 2 PASSED")

# =============================================================================
# TEST 3: Affine Registration
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: compute_affine_registration()")
print("=" * 70)

t0 = time()
affine_map = compute_affine_registration(
    static_image=subject_data,
    static_affine=subject_affine,
    moving_image=mni_data,
    moving_affine=mni_affine,
    level_iters=LEVEL_ITERS_AFFINE,
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1],
    verbose=True
)
affine_time = time() - t0

print(f"\n    Affine matrix:")
print(f"    {np.array2string(affine_map.affine, precision=3, suppress_small=True)}")
print(f"    Time: {affine_time:.2f}s")

# Test transform
print("\n    Testing affine transform...")
warped_affine = affine_map.transform(mni_data)
print(f"    Warped shape: {warped_affine.shape}")
print(f"    Warped range: [{warped_affine.min():.2f}, {warped_affine.max():.2f}]")
print("✓ Test 3 PASSED")

# =============================================================================
# TEST 4: Visualization - Before vs After Affine
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: plot_registration_comparison() - Affine Stage")
print("=" * 70)

viz_dir = output_dir / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

t0 = time()

# Before registration: resample MNI to subject grid with identity transform
from dipy.align.imaffine import AffineMap

identity_map = AffineMap(
    np.eye(4),
    subject_data.shape, subject_affine,
    mni_data.shape, mni_affine
)
mni_resampled = identity_map.transform(mni_data)

# Plot before registration
print("\n    Plotting before registration...")
figs_before = plot_registration_comparison(
    static_data=subject_data,
    moving_data=mni_resampled,
    ltitle="Subject FA",
    rtitle="MNI (identity)",
    output_dir=viz_dir,
    fname_prefix="01_before_registration"
)

# Plot after affine
print("\n    Plotting after affine registration...")
figs_affine = plot_registration_comparison(
    static_data=subject_data,
    moving_data=warped_affine,
    ltitle="Subject FA",
    rtitle="MNI (affine)",
    output_dir=viz_dir,
    fname_prefix="02_after_affine"
)

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 4 PASSED")

# =============================================================================
# TEST 5: SyN Registration
# =============================================================================

print("\n" + "=" * 70)
print("TEST 5: compute_syn_registration()")
print("=" * 70)

t0 = time()
syn_mapping = compute_syn_registration(
    static_image=subject_data,
    static_affine=subject_affine,
    moving_image=mni_data,
    moving_affine=mni_affine,
    prealign=affine_map.affine,
    level_iters=LEVEL_ITERS_SYN,
    metric_radius=4,
    verbose=True
)
syn_time = time() - t0

print(f"\n    Mapping type: {type(syn_mapping).__name__}")
print(f"    Time: {syn_time:.2f}s")

# Test transform
print("\n    Testing SyN transform...")
warped_syn = syn_mapping.transform(mni_data)
print(f"    Warped shape: {warped_syn.shape}")
print(f"    Warped range: [{warped_syn.min():.2f}, {warped_syn.max():.2f}]")
print("✓ Test 5 PASSED")

# =============================================================================
# TEST 6: Visualization - After SyN
# =============================================================================

print("\n" + "=" * 70)
print("TEST 6: plot_registration_comparison() - After SyN")
print("=" * 70)

t0 = time()
print("\n    Plotting after SyN registration...")
figs_syn = plot_registration_comparison(
    static_data=subject_data,
    moving_data=warped_syn,
    ltitle="Subject FA",
    rtitle="MNI (SyN)",
    output_dir=viz_dir,
    fname_prefix="03_after_syn"
)

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 6 PASSED")

# =============================================================================
# TEST 7: Save Registration Report (Standalone)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 7: save_registration_report()")
print("=" * 70)

t0 = time()

# Construct result dict matching register_mni_to_subject output
test_result = {
    'mapping': syn_mapping,
    'affine_map': affine_map,
    'subject_affine': subject_affine,
    'subject_shape': subject_data.shape,
    'mni_affine': mni_affine,
    'mni_shape': mni_data.shape,
    'warped_template_path': None
}

report_path = save_registration_report(
    result=test_result,
    output_dir=output_dir,
    subject_id="test_manual"
)

print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 7 PASSED")

# =============================================================================
# TEST 8: Full Pipeline - register_mni_to_subject()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 8: register_mni_to_subject() - Full Pipeline")
print("=" * 70)

pipeline_dir = output_dir / "full_pipeline"

t0 = time()
result = register_mni_to_subject(
    subject_fa_path=SUBJECT_FA_PATH,
    output_dir=pipeline_dir,
    level_iters_affine=LEVEL_ITERS_AFFINE,
    level_iters_syn=LEVEL_ITERS_SYN,
    save_warped=True,
    generate_qc=False,  # We already generated QC above
    verbose=True
)
pipeline_time = time() - t0

print(f"\n    Result keys: {list(result.keys())}")
print(f"    Mapping type: {type(result['mapping']).__name__}")
print(f"    Warped template: {result['warped_template_path']}")
print(f"    Time: {pipeline_time:.2f}s")
print("✓ Test 8 PASSED")

# =============================================================================
# TEST 9: Verify Mapping Can Transform Atlas (Label Image)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 9: Atlas Transformation Test (Nearest Neighbor)")
print("=" * 70)

t0 = time()

# Create a dummy parcellation with discrete labels in MNI space
dummy_parcellation = np.zeros(mni_data.shape, dtype=np.int32)

# Simulate some ROIs
center = np.array(mni_data.shape) // 2
dummy_parcellation[center[0]-10:center[0]+10, 
                   center[1]-10:center[1]+10, 
                   center[2]-20:center[2]] = 1  # "Brainstem"
dummy_parcellation[center[0]-20:center[0], 
                   center[1]+10:center[1]+30, 
                   center[2]:center[2]+20] = 2  # "Motor_L"
dummy_parcellation[center[0]:center[0]+20, 
                   center[1]+10:center[1]+30, 
                   center[2]:center[2]+20] = 3  # "Motor_R"

print(f"\n    Dummy parcellation shape: {dummy_parcellation.shape}")
print(f"    Unique labels before: {np.unique(dummy_parcellation)}")
print(f"    Label counts before: {[np.sum(dummy_parcellation == i) for i in range(4)]}")

# Transform with nearest neighbor interpolation (preserves discrete labels)
warped_parcellation = result['mapping'].transform(
    dummy_parcellation.astype(np.float64),
    interpolation='nearest'
)

print(f"\n    Warped parcellation shape: {warped_parcellation.shape}")
print(f"    Unique labels after: {np.unique(warped_parcellation.astype(int))}")
print(f"    Label counts after: {[np.sum(warped_parcellation.astype(int) == i) for i in range(4)]}")

# Save the test parcellation for inspection
parcel_path = output_dir / "test_parcellation_warped.nii.gz"
nib.save(
    nib.Nifti1Image(warped_parcellation.astype(np.int32), subject_affine),
    parcel_path
)
print(f"\n    Saved test parcellation: {parcel_path}")

print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 9 PASSED")

# =============================================================================
# TEST 10: Verify Inverse Transform
# =============================================================================

print("\n" + "=" * 70)
print("TEST 10: Inverse Transform Test")
print("=" * 70)

t0 = time()

# Transform subject FA to MNI space (inverse direction)
subject_to_mni = result['mapping'].transform_inverse(subject_data)

print(f"\n    Subject FA shape: {subject_data.shape}")
print(f"    Transformed to MNI space: {subject_to_mni.shape}")
print(f"    Range: [{subject_to_mni.min():.3f}, {subject_to_mni.max():.3f}]")

# Save for inspection
inverse_path = output_dir / "test_subject_fa_in_mni_space.nii.gz"
nib.save(
    nib.Nifti1Image(subject_to_mni.astype(np.float32), mni_affine),
    inverse_path
)
print(f"    Saved: {inverse_path}")

print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 10 PASSED")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

total_time = affine_time + syn_time + pipeline_time

print(f"""
All 10 tests PASSED! ✓

Timing Breakdown:
    Affine registration:     {affine_time:6.2f}s
    SyN registration:        {syn_time:6.2f}s
    Full pipeline:           {pipeline_time:6.2f}s
    ────────────────────────────────
    Total registration time: {total_time:6.2f}s

Output Directory: {output_dir}

Generated Files:
""")

# List all generated files
for f in sorted(output_dir.rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"    {f.relative_to(output_dir)} ({size_kb:.1f} KB)")

print(f"""
QC Images to Review:
    1. {viz_dir}/01_before_registration_*.png  (initial misalignment)
    2. {viz_dir}/02_after_affine_*.png         (affine correction)
    3. {viz_dir}/03_after_syn_*.png            (non-linear refinement)

Next Steps:
    - Review QC images to verify registration quality
    - Use result['mapping'] to warp MNI parcellation atlas
    - Proceed to warp_atlas_to_subject module
""")

print("=" * 70)
print("REGISTRATION MODULE TESTING COMPLETE")
print("=" * 70)

# Close all matplotlib figures
plt.close('all')