# Recommended Datasets for CST Extraction Testing

**Last Updated**: 2026-01-11  
**Purpose**: Public diffusion MRI datasets suitable for corticospinal tract (CST) tractography

---

## ✅ Highly Recommended Datasets

### 1. Human Connectome Project (HCP)
**Best for**: Production-quality CST analysis, method validation

- **URL**: https://www.humanconnectome.org/
- **Access**: Free registration required, open to scientific community
- **Data Quality**: 
  - High angular resolution (90+ directions)
  - 1.25mm isotropic resolution
  - Multi-shell acquisition (b=1000, 2000, 3000 s/mm²)
  - Full brain coverage including superior motor cortex
- **Sample Size**: 1,200+ young adults (HCP-YA), plus lifespan and disease cohorts
- **Preprocessing**: Minimal preprocessing pipelines available
- **CST Validation**: Extensively used for CST delineation in published research
- **Download Size**: ~3-5 GB per subject (diffusion + structural)

**Why it works**: HCP data is specifically designed for tractography with high SNR, full cortical coverage, and validated preprocessing. Multiple studies have successfully used HCP for CST analysis.

**Getting Started**:
1. Register at https://db.humanconnectome.org/
2. Accept data use terms
3. Download "Diffusion" and "Structural" preprocessed data
4. Use `*_dwi.nii.gz`, `*.bval`, `*.bvec` files directly with csttool

---

### 2. Brainlife.io - Open Diffusion Data Derivatives (O3D)
**Best for**: Quick testing, pre-computed tractograms available

- **URL**: https://brainlife.io/
- **Access**: Free, no registration required for public datasets
- **Data Quality**:
  - Curated high-quality datasets
  - Includes whole-brain tractograms (pre-computed)
  - Segmented white matter tracts available
- **Datasets**: Multiple collections including collegiate athletes, clinical populations
- **Preprocessing**: Fully preprocessed derivatives available
- **Unique Feature**: Can download pre-computed tractograms to test extraction only

**Why it works**: Brainlife provides both raw and processed data, allowing you to validate your pipeline at different stages.

**Getting Started**:
1. Browse https://brainlife.io/datasets
2. Filter for "diffusion" modality
3. Download preprocessed DWI + tractograms
4. Test extraction step directly with existing tractograms

---

### 3. Fiber Data Hub (Brain MRI Collections)
**Best for**: Recent high-resolution datasets, benchmarking

- **URL**: https://labsolver.org/fiber-data-hub.html
- **Access**: Free, curated external datasets
- **Data Quality**:
  - Focus on post-2020 high-resolution acquisitions
  - Suitable for tractography benchmarking
  - Multiple acquisition protocols
- **Coverage**: Various clinical and research populations
- **Use Case**: Method development and validation

**Why it works**: Curated specifically for tractography research with quality control.

---

## 🔬 Specialized/Research Datasets

### 4. Kennedy Krieger Institute DTI Database
**Best for**: Normal population reference data

- **URL**: https://www.kennedykrieger.org/research/centers-labs-cores/pediatric-neuroimaging-research-consortium
- **Access**: Registration required
- **Data Quality**:
  - 1.0-2.5mm³ resolution
  - Normal population focus
  - Online fiber tract database included
- **Preprocessing**: Raw and processed DTI available

**Note**: May have more restricted access than HCP.

---

### 5. Digital Brain Bank - High-Resolution Post-Mortem
**Best for**: Ultra-high resolution validation, anatomical ground truth

- **URL**: https://digitalbrain.org/ (or via NIH repositories)
- **Access**: Open access
- **Data Quality**:
  - 500 μm isotropic resolution (highest available)
  - Post-mortem imaging
  - Includes polarized light imaging (PLI) for validation
- **Use Case**: Understanding resolution effects on tractography

**Note**: Post-mortem data may have different characteristics than in-vivo.

---

## 📋 Dataset Selection Criteria

When choosing a dataset for CST extraction, verify:

### ✅ Essential Requirements
- [ ] **Full brain coverage** - Must include superior motor cortex (check Z-range)
- [ ] **Adequate resolution** - ≤2.5mm isotropic preferred
- [ ] **Multi-shell or HARDI** - At least 30 directions, preferably 60+
- [ ] **b-value ≥1000** - Higher b-values (1000-3000) better for tractography
- [ ] **Gradient files included** - `.bval` and `.bvec` must be available

### ⚠️ Warning Signs (Like Stanford HARDI)
- [ ] Limited field of view (FOV < 220mm superior-inferior)
- [ ] Low resolution (>3mm voxels)
- [ ] Few directions (<30)
- [ ] Low b-value (<1000 s/mm²)
- [ ] Partial brain coverage

---

## 🚀 Quick Start Recommendation

**For immediate testing**: Start with **HCP Young Adult** dataset
1. Download one subject's preprocessed diffusion data
2. Use the `*_dwi.nii.gz` file directly with csttool
3. Expected result: 5-15% CST extraction rate with default settings

**For comparison**: Download **Brainlife.io** pre-computed tractogram
1. Test your extraction step independently
2. Validate ROI creation and filtering logic
3. Compare with their segmented tract results

---

## 📊 Expected Results by Dataset Type

| Dataset Type | Typical CST Extraction Rate | Notes |
|-------------|---------------------------|-------|
| HCP (high quality) | 5-15% | Gold standard |
| Clinical (1.5T, standard) | 2-8% | Variable quality |
| Research (3T, optimized) | 8-20% | Protocol-dependent |
| Stanford HARDI | <0.001% | **Negative control** |

---

## 📚 Additional Resources

### Data Repositories
- **OpenNeuro**: https://openneuro.org/ (filter for "dwi" modality)
- **NITRC**: https://www.nitrc.org/search/?type_of_search=group&q=diffusion
- **INDI**: http://fcon_1000.projects.nitrc.org/ (various initiatives)

### Documentation
- HCP Data Dictionary: https://wiki.humanconnectome.org/
- BIDS Format Guide: https://bids-specification.readthedocs.io/ (for data organization)
- Brainlife Tutorials: https://brainlife.io/docs/

---

## 🎯 Recommendation Summary

**Primary Choice**: **Human Connectome Project (HCP)**
- Proven track record for CST analysis
- High quality, full coverage
- Large sample size for validation

**Alternative**: **Brainlife.io**
- Faster download
- Pre-computed derivatives for testing
- Good for pipeline validation

**Avoid**: Datasets similar to Stanford HARDI (low resolution, limited FOV, <30 directions)
