from nilearn import datasets

subcort = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
print("Subcortical labels:")
for i, label in enumerate(subcort.labels):
    print(f"  {i}: {label}")

cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
print("Cortical labels:")
for i, label in enumerate(cort.labels):
    print(f"    {i}: {label}")
