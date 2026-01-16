from time import time
from csttool.tracking.modules import (
    load_and_mask,
    fit_tensors,
    estimate_directions,
    seed_and_stop,
    run_tractography,
    save_tracking_outputs
)

nii_dir = "/home/alemnalo/anom/nifti"
nii_fname = "17_cmrr_mbep2d_diff_ap_tdi"
out_dir = "/home/alemnalo/anom/tracking_test_output"

tracking_params = {
    'step_size': 0.5,
    'fa_thresh': 0.2,
    'seed_density': 1,
    'sh_order': 6,
    'sphere': 'symmetric362',
    'stopping_criterion': 'fa_threshold',
    'relative_peak_threshold': 0.8,
    'min_separation_angle': 45,
}

print("=" * 70)
print("CSTTOOL TRACKING PIPELINE TEST")
print("=" * 70)

total_start = time()

# Step 1: Load and mask
print("\n[1/6] LOAD & MASK")
t0 = time()
data, affine, img, gtab, masked_data, brain_mask = load_and_mask(
    nii_dir, nii_fname, visualize=False, verbose=True
)
print(f"Time: {time() - t0:.1f}s")

# Step 2: Fit tensors
print("\n[2/6] TENSOR FITTING")
t0 = time()
tenfit, fa, md, rd, ad, white_matter = fit_tensors(
    masked_data, gtab, brain_mask,
    fa_thresh=tracking_params['fa_thresh'],
    verbose=True
)
print(f"Time: {time() - t0:.1f}s")

# Step 3: Estimate directions
print("\n[3/6] DIRECTION FIELD")
t0 = time()
csapeaks = estimate_directions(
    masked_data, gtab, white_matter,
    sh_order=tracking_params['sh_order'],
    verbose=True
)
print(f"Time: {time() - t0:.1f}s")

# Step 4: Seeds and stopping
print("\n[4/6] STOPPING & SEEDING")
t0 = time()
seeds, stopping = seed_and_stop(
    fa, affine,
    fa_thresh=tracking_params['fa_thresh'],
    density=tracking_params['seed_density'],
    verbose=True
)
print(f"Time: {time() - t0:.1f}s")

# Step 5: Tractography
print("\n[5/6] TRACTOGRAPHY")
t0 = time()
streamlines = run_tractography(
    csapeaks, stopping, seeds, affine,
    step_size=tracking_params['step_size'],
    verbose=True, visualize=False
)
print(f"Time: {time() - t0:.1f}s")

# Step 6: Save outputs
print("\n[6/6] SAVE OUTPUTS")
t0 = time()
outputs = save_tracking_outputs(
    streamlines, img, fa, md, affine,
    out_dir=out_dir,
    stem=nii_fname,
    rd=rd,
    ad=ad,
    tracking_params=tracking_params,
    verbose=True
)
print(f"Time: {time() - t0:.1f}s")

# Summary
total_time = time() - total_start
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"Streamlines: {len(streamlines):,}")
print(f"\nOutputs:")
for key, path in outputs.items():
    print(f"  {key}: {path}")
print("=" * 70)