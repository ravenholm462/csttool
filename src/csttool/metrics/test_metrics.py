#!/usr/bin/env python3
"""
test_metrics_module.py

Comprehensive test script for csttool's metrics module.

This script tests:
1. Module imports
2. Unilateral analysis functions
3. Bilateral comparison functions
4. Visualization generation
5. Report generation
6. Full pipeline integration

Usage:
    python test_metrics_module.py
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path if needed
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Also need DIPY
try:
    import dipy
except ImportError:
    print("⚠️  WARNING: DIPY not found. Installing required packages...")
    print("   Run: pip install numpy scipy dipy matplotlib")
    sys.exit(1)

print("="*70)
print("CSTTOOL METRICS MODULE - COMPREHENSIVE TEST")
print("="*70)

# ============================================================================
# TEST 1: MODULE IMPORTS
# ============================================================================
print("\n[TEST 1] Testing module imports...")

try:
    from csttool.metrics.modules import (
        analyze_cst_hemisphere,
        compare_bilateral_cst,
        plot_tract_profiles,
        plot_bilateral_comparison,
        create_summary_figure
    )
    from csttool.metrics.modules.reports import (
        save_json_report,
        save_csv_summary,
        generate_complete_report
    )
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: CREATE SYNTHETIC DATA
# ============================================================================
print("\n[TEST 2] Creating synthetic test data...")

def create_synthetic_streamlines(n_streamlines=100, length=50):
    """Create synthetic streamlines for testing."""
    from dipy.tracking.streamline import Streamlines
    
    streamlines = []
    for i in range(n_streamlines):
        # Create a curved streamline
        t = np.linspace(0, 1, length)
        x = 50 + 20 * np.sin(2 * np.pi * t) + np.random.randn(length) * 0.5
        y = 50 + 30 * t + np.random.randn(length) * 0.5
        z = 40 + 10 * np.cos(2 * np.pi * t) + np.random.randn(length) * 0.5
        
        streamline = np.column_stack([x, y, z])
        streamlines.append(streamline)
    
    return Streamlines(streamlines)

def create_synthetic_scalar_map(shape=(100, 100, 100), mean_value=0.5):
    """Create synthetic FA or MD map."""
    # Create a gradient with some noise
    scalar_map = np.ones(shape) * mean_value
    
    # Add spatial variation
    x, y, z = np.meshgrid(
        np.linspace(0, 1, shape[0]),
        np.linspace(0, 1, shape[1]),
        np.linspace(0, 1, shape[2]),
        indexing='ij'
    )
    
    scalar_map += 0.2 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    scalar_map += np.random.randn(*shape) * 0.05
    
    # Clip to valid range
    if mean_value == 0.5:  # FA-like
        scalar_map = np.clip(scalar_map, 0, 1)
    else:  # MD-like
        scalar_map = np.clip(scalar_map, 0, 0.003)
    
    return scalar_map

# Create test data
print("  Creating synthetic streamlines...")
streamlines_left = create_synthetic_streamlines(n_streamlines=120, length=50)
streamlines_right = create_synthetic_streamlines(n_streamlines=110, length=48)

print("  Creating synthetic scalar maps...")
fa_map = create_synthetic_scalar_map(shape=(100, 100, 100), mean_value=0.5)
md_map = create_synthetic_scalar_map(shape=(100, 100, 100), mean_value=0.0008)

# Create affine matrix
affine = np.eye(4)
affine[:3, :3] = np.diag([2.0, 2.0, 2.0])  # 2mm isotropic

print(f"✅ Synthetic data created:")
print(f"   Left CST: {len(streamlines_left)} streamlines")
print(f"   Right CST: {len(streamlines_right)} streamlines")
print(f"   FA map: {fa_map.shape}, mean = {fa_map.mean():.3f}")
print(f"   MD map: {md_map.shape}, mean = {md_map.mean():.3e}")

# ============================================================================
# TEST 3: UNILATERAL ANALYSIS
# ============================================================================
print("\n[TEST 3] Testing unilateral analysis...")

try:
    print("  Analyzing LEFT hemisphere...")
    left_metrics = analyze_cst_hemisphere(
        streamlines=streamlines_left,
        fa_map=fa_map,
        md_map=md_map,
        affine=affine,
        hemisphere='left'
    )
    
    print("  Analyzing RIGHT hemisphere...")
    right_metrics = analyze_cst_hemisphere(
        streamlines=streamlines_right,
        fa_map=fa_map,
        md_map=md_map,
        affine=affine,
        hemisphere='right'
    )
    
    # Verify structure
    assert 'hemisphere' in left_metrics
    assert 'morphology' in left_metrics
    assert 'fa' in left_metrics
    assert 'md' in left_metrics
    
    assert left_metrics['morphology']['n_streamlines'] == 120
    assert right_metrics['morphology']['n_streamlines'] == 110
    
    print("✅ Unilateral analysis successful!")
    print(f"   Left: {left_metrics['morphology']['n_streamlines']} streamlines, "
          f"FA={left_metrics['fa']['mean']:.3f}")
    print(f"   Right: {right_metrics['morphology']['n_streamlines']} streamlines, "
          f"FA={right_metrics['fa']['mean']:.3f}")
    
except Exception as e:
    print(f"❌ Unilateral analysis failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: BILATERAL COMPARISON
# ============================================================================
print("\n[TEST 4] Testing bilateral comparison...")

try:
    comparison = compare_bilateral_cst(
        left_metrics=left_metrics,
        right_metrics=right_metrics
    )
    
    # Verify structure
    assert 'left' in comparison
    assert 'right' in comparison
    assert 'asymmetry' in comparison
    
    assert 'volume' in comparison['asymmetry']
    assert 'fa' in comparison['asymmetry']
    assert 'laterality_index' in comparison['asymmetry']['volume']
    
    volume_li = comparison['asymmetry']['volume']['laterality_index']
    fa_li = comparison['asymmetry']['fa']['laterality_index']
    
    print("✅ Bilateral comparison successful!")
    print(f"   Volume LI: {volume_li:+.3f}")
    print(f"   FA LI: {fa_li:+.3f}")
    print(f"   Interpretation: {comparison['asymmetry']['volume']['interpretation']}")
    
except Exception as e:
    print(f"❌ Bilateral comparison failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: VISUALIZATIONS
# ============================================================================
print("\n[TEST 5] Testing visualization generation...")

output_dir = Path(__file__).parent / "test_output" / "metrics_test"
output_dir.mkdir(parents=True, exist_ok=True)
subject_id = "TEST-001"

try:
    print("  Generating tract profiles...")
    profile_path = plot_tract_profiles(
        left_metrics=left_metrics,
        right_metrics=right_metrics,
        output_dir=output_dir / "visualizations",
        subject_id=subject_id,
        scalar='fa'
    )
    
    print("  Generating bilateral comparison...")
    comparison_path = plot_bilateral_comparison(
        comparison=comparison,
        output_dir=output_dir / "visualizations",
        subject_id=subject_id
    )
    
    print("  Generating summary figure...")
    summary_path = create_summary_figure(
        comparison=comparison,
        streamlines_left=streamlines_left,
        streamlines_right=streamlines_right,
        fa_map=fa_map,
        affine=affine,
        output_dir=output_dir / "visualizations",
        subject_id=subject_id
    )
    
    # Verify files exist
    assert profile_path.exists(), f"Profile plot not created: {profile_path}"
    assert comparison_path.exists(), f"Comparison plot not created: {comparison_path}"
    assert summary_path.exists(), f"Summary figure not created: {summary_path}"
    
    print("✅ Visualization generation successful!")
    print(f"   Tract profiles: {profile_path}")
    print(f"   Comparison: {comparison_path}")
    print(f"   Summary: {summary_path}")
    
except Exception as e:
    print(f"❌ Visualization generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: REPORT GENERATION
# ============================================================================
print("\n[TEST 6] Testing report generation...")

try:
    print("  Generating JSON report...")
    json_path = save_json_report(
        comparison=comparison,
        output_dir=output_dir,
        subject_id=subject_id
    )
    
    print("  Generating CSV summary...")
    csv_path = save_csv_summary(
        comparison=comparison,
        output_dir=output_dir,
        subject_id=subject_id
    )
    
    # Verify files exist
    assert json_path.exists(), f"JSON report not created: {json_path}"
    assert csv_path.exists(), f"CSV summary not created: {csv_path}"
    
    # Read JSON to verify format
    import json
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    assert 'subject_id' in json_data
    assert 'metrics' in json_data
    assert json_data['subject_id'] == subject_id
    
    # Read CSV to verify format
    import csv
    with open(csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        rows = list(csv_reader)
    
    assert len(rows) == 1
    assert rows[0]['subject_id'] == subject_id
    
    print("✅ Report generation successful!")
    print(f"   JSON: {json_path}")
    print(f"   CSV: {csv_path}")
    
    # Try PDF generation (may fail if reportlab not installed)
    try:
        from csttool.metrics.modules.reports import save_pdf_report
        
        viz_paths = {
            'tract_profiles': profile_path,
            'bilateral_comparison': comparison_path,
            'summary': summary_path
        }
        
        print("  Generating PDF report...")
        pdf_path = save_pdf_report(
            comparison=comparison,
            visualization_paths=viz_paths,
            output_dir=output_dir,
            subject_id=subject_id
        )
        
        if pdf_path and pdf_path.exists():
            print(f"✅ PDF report generated: {pdf_path}")
        else:
            print("⚠️  PDF generation skipped (reportlab not installed)")
    except Exception as e:
        print(f"⚠️  PDF generation failed (this is optional): {e}")
    
except Exception as e:
    print(f"❌ Report generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 7: FULL PIPELINE INTEGRATION
# ============================================================================
print("\n[TEST 7] Testing full pipeline integration...")

try:
    print("  Running complete report generation...")
    
    report_paths = generate_complete_report(
        comparison=comparison,
        streamlines_left=streamlines_left,
        streamlines_right=streamlines_right,
        fa_map=fa_map,
        affine=affine,
        output_dir=output_dir / "full_pipeline",
        subject_id=subject_id
    )
    
    # Verify all reports generated
    assert 'json' in report_paths
    assert 'csv' in report_paths
    assert 'visualizations' in report_paths
    
    print("✅ Full pipeline integration successful!")
    print(f"   Generated {len(report_paths)} report types")
    
except Exception as e:
    print(f"❌ Full pipeline integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 8: EDGE CASES
# ============================================================================
print("\n[TEST 8] Testing edge cases...")

try:
    # Test with empty streamlines
    from dipy.tracking.streamline import Streamlines
    empty_streamlines = Streamlines([])
    
    print("  Testing empty streamlines...")
    empty_metrics = analyze_cst_hemisphere(
        streamlines=empty_streamlines,
        fa_map=fa_map,
        md_map=md_map,
        affine=affine,
        hemisphere='empty'
    )
    
    assert empty_metrics['morphology']['n_streamlines'] == 0
    assert empty_metrics['morphology']['tract_volume'] == 0.0
    print("  ✓ Empty streamlines handled correctly")
    
    # Test without scalar maps
    print("  Testing without scalar maps...")
    no_scalar_metrics = analyze_cst_hemisphere(
        streamlines=streamlines_left,
        fa_map=None,
        md_map=None,
        affine=affine,
        hemisphere='test'
    )
    
    assert 'fa' not in no_scalar_metrics
    assert 'md' not in no_scalar_metrics
    assert 'morphology' in no_scalar_metrics
    print("  ✓ Missing scalar maps handled correctly")
    
    print("✅ Edge cases handled correctly!")
    
except Exception as e:
    print(f"❌ Edge case testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("✅ All tests passed successfully!")
print(f"\nTest outputs saved to: {output_dir}")
print("\nGenerated files:")
print(f"  - Visualizations: {len(list((output_dir / 'visualizations').glob('*.png')))} PNG files")
print(f"  - JSON report: {json_path.name}")
print(f"  - CSV summary: {csv_path.name}")
if 'pdf_path' in locals() and pdf_path and pdf_path.exists():
    print(f"  - PDF report: {pdf_path.name}")

print("\n" + "="*70)
print("METRICS MODULE IS READY FOR USE!")
print("="*70)
print("\nNext steps:")
print("1. Integrate into CLI (see METRICS_USAGE_GUIDE.py)")
print("2. Test with real CST data")
print("3. Validate metrics against known values")
print("="*70)

sys.exit(0)