# Imports
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from os.path import join
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import deterministic_tracking
from dipy.tracking.utils import seeds_from_mask
from dipy.viz import actor, colormap, has_fury, window

# Enables/disables interactive visualization
interactive = False

# Define paths
#dicom_path = "/home/alemnalo/anom/cmrr_mbep2d_diff_AP_TDI_Series0017/"
dicom_path = "/home/alem/Documents/thesis/data/anom/"
#nifti_path = "/home/alemnalo/anom/nifti"
nifti_path = "/home/alem/Documents/thesis/data/nifti"

# out_path = "/home/alem/Documents/thesis/data/nifti/out"

fname = "17_cmrr_mbep2d_diff_ap_tdi"
fdwi = join(nifti_path, fname + "_preproc.nii.gz")
print(fdwi)

fbval = join(nifti_path, fname + ".bval")
print(fbval)

fbvec = join(nifti_path, fname + ".bvec")
print(fbvec)

# Load dataset, show shape and voxel size
data, affine, img = load_nifti(fdwi, return_img=True)
print("\n" + "="*70)
print(f"Loaded study {fname}")
print(f"Data shape: {data.shape}")
print(f"Voxel size [mm]: {img.header.get_zooms()[:3]}")
print("\n" + "="*70)

labels = load_nifti_data(label_fname)  #  This needs to be created first
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs=bvecs)

seed_mask = labels == 2
seeds = seeds_from_mask(seed_mask, affine, density=2)

white_matter = (labels == 1) | (labels == 2)
sc = BinaryStoppingCriterion(white_matter)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)