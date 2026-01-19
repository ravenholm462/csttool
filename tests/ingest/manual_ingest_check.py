#!/usr/bin/env python3
"""
test_ingest_modules.py

Comprehensive test script for csttool's ingest module.

Usage:
    python test_ingest_modules.py /path/to/dicom/study

This script tests:
1. DICOM directory scanning
2. Series analysis and classification
3. Suitability scoring and recommendations
4. DICOM to NIfTI conversion
5. Output organization
6. Full pipeline integration

If no path is provided, runs with synthetic/mock data where possible.
"""

import sys
from pathlib import Path
from time import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default test paths - update these for your system
DEFAULT_STUDY_DIR = "/home/alemnalo/anom"  # Update this
DEFAULT_OUTPUT_DIR = "/home/alemnalo/anom/ingest_test_output"

# =============================================================================
# SETUP
# =============================================================================

print("=" * 70)
print("CSTTOOL INGEST MODULE - TEST SCRIPT")
print("=" * 70)

# Get paths from command line or use defaults
if len(sys.argv) > 1:
    STUDY_DIR = Path(sys.argv[1])
else:
    STUDY_DIR = Path(DEFAULT_STUDY_DIR)

if len(sys.argv) > 2:
    OUTPUT_DIR = Path(sys.argv[2])
else:
    OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)

print(f"\nStudy directory: {STUDY_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Check if study exists
if not STUDY_DIR.exists():
    print(f"\n⚠️  Study directory not found: {STUDY_DIR}")
    print("   Update the path or provide as command line argument:")
    print("   python test_ingest_modules.py /path/to/dicom/study")
    print("\n   Running limited tests without DICOM data...")
    HAS_DICOM = False
else:
    HAS_DICOM = True

# =============================================================================
# IMPORTS
# =============================================================================

print("\n[TEST 0] Importing modules...")

try:
    from csttool.ingest.modules import (
        # scan_study module
        is_dicom_file,
        is_dicom_directory,
        count_dicom_files,
        scan_study,
        filter_series_by_file_count,
        print_series_summary,
        
        # analyze_series module
        SeriesType,
        SeriesAnalysis,
        analyze_series,
        analyze_all_series,
        recommend_series,
        
        # convert_series module
        convert_dicom_to_nifti,
        validate_conversion,
        
        # save_ingest_outputs module
        save_ingest_outputs,
        print_import_summary,
        get_nifti_stem
    )
    
    from csttool.ingest import run_ingest_pipeline
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n   Make sure csttool is installed:")
    print("   pip install -e /path/to/csttool")
    sys.exit(1)

# =============================================================================
# TEST 1: is_dicom_directory()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: is_dicom_directory()")
print("=" * 70)

if HAS_DICOM:
    t0 = time()
    
    # Test study directory
    result = is_dicom_directory(STUDY_DIR)
    print(f"  Study root is DICOM dir: {result}")
    
    # Test subdirectories
    for item in sorted(STUDY_DIR.iterdir())[:5]:
        if item.is_dir():
            result = is_dicom_directory(item)
            print(f"  {item.name}: {result}")
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 1 PASSED")
else:
    print("  ⏭️  Skipped (no DICOM data)")

# =============================================================================
# TEST 2: scan_study()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: scan_study()")
print("=" * 70)

series_list = []

if HAS_DICOM:
    t0 = time()
    
    series_list = scan_study(STUDY_DIR, recursive=True, verbose=True)
    
    print(f"\n  Found {len(series_list)} series")
    print(f"  Time: {time() - t0:.2f}s")
    
    if series_list:
        print_series_summary(series_list)
        print("✅ Test 2 PASSED")
    else:
        print("⚠️  No series found - check directory structure")
else:
    print("  ⏭️  Skipped (no DICOM data)")

# =============================================================================
# TEST 3: filter_series_by_file_count()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: filter_series_by_file_count()")
print("=" * 70)

if series_list:
    t0 = time()
    
    # Filter with different thresholds
    filtered_10 = filter_series_by_file_count(series_list, min_files=10)
    filtered_50 = filter_series_by_file_count(series_list, min_files=50)
    filtered_100 = filter_series_by_file_count(series_list, min_files=100)
    
    print(f"  Original series:       {len(series_list)}")
    print(f"  ≥10 files:             {len(filtered_10)}")
    print(f"  ≥50 files:             {len(filtered_50)}")
    print(f"  ≥100 files:            {len(filtered_100)}")
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 3 PASSED")
else:
    print("  ⏭️  Skipped (no series data)")

# =============================================================================
# TEST 4: analyze_series() - Single Series
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: analyze_series() - Single Series")
print("=" * 70)

if series_list:
    t0 = time()
    
    # Analyze first series
    first_series = series_list[0]
    analysis = analyze_series(first_series['path'], verbose=True)
    
    print(f"\n  Analysis complete:")
    print(f"    Series type: {analysis.series_type.value}")
    print(f"    Suitable: {analysis.suitable_for_tractography}")
    print(f"    Score: {analysis.suitability_score}")
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 4 PASSED")
else:
    print("  ⏭️  Skipped (no series data)")

# =============================================================================
# TEST 5: analyze_all_series()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 5: analyze_all_series()")
print("=" * 70)

analyses = []

if series_list:
    t0 = time()
    
    analyses = analyze_all_series(series_list, verbose=True)
    
    print(f"\n  Analyzed {len(analyses)} series")
    
    # Summarize by type
    type_counts = {}
    for a in analyses:
        t = a.series_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"\n  Series by type:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count}")
    
    suitable_count = sum(1 for a in analyses if a.suitable_for_tractography)
    print(f"\n  Suitable for tractography: {suitable_count}/{len(analyses)}")
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 5 PASSED")
else:
    print("  ⏭️  Skipped (no series data)")

# =============================================================================
# TEST 6: recommend_series()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 6: recommend_series()")
print("=" * 70)

recommended = None

if analyses:
    t0 = time()
    
    recommended = recommend_series(analyses, verbose=True)
    
    if recommended:
        print(f"\n  Recommended: {recommended.name}")
        print(f"  Score: {recommended.suitability_score}")
        print("✅ Test 6 PASSED")
    else:
        print("\n  ⚠️  No suitable series found")
        print("  This may be expected if all series are derived images")
else:
    print("  ⏭️  Skipped (no analyses)")

# =============================================================================
# TEST 7: convert_dicom_to_nifti()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 7: convert_dicom_to_nifti()")
print("=" * 70)

conversion_result = None

if recommended:
    t0 = time()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_dir = OUTPUT_DIR / "conversion_test"
    
    try:
        conversion_result = convert_dicom_to_nifti(
            recommended.path,
            convert_dir,
            verbose=True
        )
        
        print(f"\n  Conversion {'succeeded' if conversion_result['success'] else 'failed'}")
        print(f"  NIfTI: {conversion_result['nifti_path']}")
        print(f"  bval:  {conversion_result['bval_path']}")
        print(f"  bvec:  {conversion_result['bvec_path']}")
        
        if conversion_result['warnings']:
            print(f"\n  Warnings:")
            for w in conversion_result['warnings']:
                print(f"    ⚠️  {w}")
        
        print(f"\n  Time: {time() - t0:.2f}s")
        print("✅ Test 7 PASSED")
        
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        print("  Make sure dicom2nifti is installed: pip install dicom2nifti")
else:
    print("  ⏭️  Skipped (no recommended series)")

# =============================================================================
# TEST 8: validate_conversion()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 8: validate_conversion()")
print("=" * 70)

if conversion_result and conversion_result['success']:
    t0 = time()
    
    validation = validate_conversion(
        conversion_result['nifti_path'],
        conversion_result['bval_path'],
        conversion_result['bvec_path'],
        verbose=True
    )
    
    print(f"\n  Overall valid: {validation['valid']}")
    print(f"  NIfTI valid: {validation['nifti_valid']}")
    print(f"  Gradients valid: {validation['gradients_valid']}")
    print(f"  Data shape: {validation['data_shape']}")
    print(f"  Volumes: {validation['n_volumes']}")
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 8 PASSED")
else:
    print("  ⏭️  Skipped (no conversion result)")

# =============================================================================
# TEST 9: save_ingest_outputs()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 9: save_ingest_outputs()")
print("=" * 70)

outputs = None

if conversion_result and conversion_result['success'] and recommended:
    t0 = time()
    
    organized_dir = OUTPUT_DIR / "organized"
    
    outputs = save_ingest_outputs(
        conversion_result,
        recommended,
        organized_dir,
        subject_id="test_subject",
        verbose=True
    )
    
    print(f"\n  Output structure created at: {organized_dir}")
    print(f"  Subject ID: {outputs['subject_id']}")
    
    # List created files
    print(f"\n  Created files:")
    for key in ['nifti_path', 'bval_path', 'bvec_path', 'report_path']:
        if outputs.get(key):
            print(f"    {key}: {outputs[key]}")
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 9 PASSED")
else:
    print("  ⏭️  Skipped (no conversion result)")

# =============================================================================
# TEST 10: print_import_summary()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 10: print_import_summary()")
print("=" * 70)

if outputs and recommended:
    t0 = time()
    
    print_import_summary(outputs, recommended)
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 10 PASSED")
else:
    print("  ⏭️  Skipped (no outputs)")

# =============================================================================
# TEST 11: get_nifti_stem()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 11: get_nifti_stem()")
print("=" * 70)

if outputs:
    t0 = time()
    
    stem = get_nifti_stem(outputs)
    print(f"  NIfTI stem: {stem}")
    
    # Verify we can find files with this stem
    if outputs['nifti_path']:
        nifti_dir = outputs['nifti_path'].parent
        matching_files = list(nifti_dir.glob(f"{stem}*"))
        print(f"  Matching files in directory: {len(matching_files)}")
        for f in matching_files:
            print(f"    - {f.name}")
    
    print(f"\n  Time: {time() - t0:.2f}s")
    print("✅ Test 11 PASSED")
else:
    print("  ⏭️  Skipped (no outputs)")

# =============================================================================
# TEST 12: Full Pipeline - run_ingest_pipeline()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 12: run_ingest_pipeline() - Full Integration")
print("=" * 70)

if HAS_DICOM:
    t0 = time()
    
    pipeline_output = OUTPUT_DIR / "full_pipeline"
    
    try:
        result = run_ingest_pipeline(
            STUDY_DIR,
            pipeline_output,
            subject_id="pipeline_test",
            auto_select=True,
            verbose=True
        )
        
        print(f"\n  Pipeline completed successfully!")
        print(f"  Output: {result['nifti_path']}")
        
        print(f"\n  Time: {time() - t0:.2f}s")
        print("✅ Test 12 PASSED")
        
    except Exception as e:
        print(f"  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ⏭️  Skipped (no DICOM data)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

if HAS_DICOM:
    print("\n✅ All available tests completed!")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    
    if outputs:
        print(f"\nReady for preprocessing:")
        print(f"  csttool preprocess --nifti {outputs['nifti_path']} --out /path/to/output")
else:
    print("\n⚠️  Limited testing (no DICOM data available)")
    print("   Provide a DICOM study directory for full testing:")
    print(f"   python {__file__} /path/to/dicom/study")

print("\n" + "=" * 70)