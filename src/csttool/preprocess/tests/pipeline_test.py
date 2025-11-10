from csttool.preprocess.funcs import (
    load_dataset,
    denoise_nlmeans,
    background_segmentation,
    perform_motion_correction,
    save_output,
)

# Define paths
# dicom_path = "/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017/"
# dicom_path = "/home/alem/Documents/thesis/data/anom/"
# nifti_path = "/home/alemnalo/anom/nifti"
nifti_path = "/home/alemnalo/anom/nifti"
out_path = "/home/alemnalo/anom/outtest"

fname = "17_cmrr_mbep2d_diff_ap_tdi"

data, affine, img, gtab = load_dataset(nifti_path, fname)
den, brain_mask_piesno = denoise_nlmeans(data, visualize=False)
b0masked_data, brain_mask_median = background_segmentation(den, gtab, visualize=False)
data_corrected, reg_affines = perform_motion_correction(b0masked_data, gtab, affine)
save_output(data_corrected, affine, out_path, fname)
