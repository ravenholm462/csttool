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

data, affine, img, gtab, masked_data, brain_mask = load_and_mask(nii_dir, nii_fname, visualize=False, verbose=True)
print("")
tenfit, fa, md, white_matter = fit_tensors(masked_data, gtab, brain_mask, fa_thresh=tracking_params['fa_thresh'], verbose=True)
print("")
csapeaks = estimate_directions(masked_data, gtab, white_matter, sh_order=tracking_params['sh_order'], verbose=True)
print("")
seeds, stopping = seed_and_stop(fa, affine, fa_thresh=tracking_params['fa_thresh'], density=tracking_params['seed_density'], verbose=True)
print("")
streamlines = run_tractography(csapeaks, stopping, seeds, affine, step_size=tracking_params['step_size'], verbose=True, visualize=False)
print("")
# Save with full parameter record
outputs = save_tracking_outputs(
    streamlines, img, fa, md, affine,
    out_dir=out_dir,
    stem=nii_fname,
    tracking_params=tracking_params,
    verbose=True
)