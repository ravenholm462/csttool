"""
Diagnostic script to inspect Harvard-Oxford atlas labels.
"""

import nibabel as nib
import numpy as np
from ...data.loader import get_harvard_oxford_path

# Load subcortical atlas (2mm resolution)
print("Loading subcortical atlas (2mm)...")
subcort_path = get_harvard_oxford_path('subcortical', '2mm')
subcort_img = nib.load(subcort_path)
subcort_data = subcort_img.get_fdata()

# Get unique label values
subcort_labels = np.unique(subcort_data).astype(int)
print(f"\nSubcortical atlas: {subcort_path}")
print(f"Shape: {subcort_data.shape}")
print(f"Unique labels ({len(subcort_labels)}):")
for label in subcort_labels:
    if label > 0:  # Skip background (0)
        print(f"  Label {label}")

# Load cortical atlas (2mm resolution)
print("\nLoading cortical atlas (2mm)...")
cort_path = get_harvard_oxford_path('cortical', '2mm')
cort_img = nib.load(cort_path)
cort_data = cort_img.get_fdata()

# Get unique label values
cort_labels = np.unique(cort_data).astype(int)
print(f"\nCortical atlas: {cort_path}")
print(f"Shape: {cort_data.shape}")
print(f"Unique labels ({len(cort_labels)}):")
for label in cort_labels:
    if label > 0:  # Skip background (0)
        print(f"  Label {label}")
