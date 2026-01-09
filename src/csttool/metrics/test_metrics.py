"""
test.py

Test script for csttool's metrics analysis pipeline.

This script tests:
1. Loading bilateral CST tractograms (left, right, bilateral)
2. Loading FA and MD scalar maps
3. Analyzing individual CST bundles
4. Comparing bilateral CST metrics
5. Generating comprehensive reports
6. Visualizing tract profiles
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti

# Import from modular structure
from csttool.metrics.modules.unilateral_analysis import analyze_cst_hemisphere
from csttool.metrics.modules.bilateral_analysis import compare_bilateral_cst
from csttool.metrics.modules.reports import print_metrics_summary


def main():
    """Main test function for metrics module."""
    
    # =================================================================
    # PATHS - Update these to match your system
    # =================================================================
    base_dir = Path("/home/alem/Documents/thesis/data/out")
    
    # Input paths
    cst_dir = base_dir / "extraction_test_output" / "cst_tractograms"
    trk_left = cst_dir / "17_cmrr_cst_left.trk"
    trk_right = cst_dir / "17_cmrr_cst_right.trk"
    trk_bilateral = cst_dir / "17_cmrr_cst_bilateral.trk"
    
    scalar_dir = base_dir / "trk" / "scalar_maps"
    fa_map_path = scalar_dir / "17_cmrr_mbep2d_diff_ap_tdi.nii_fa.nii.gz"
    md_map_path = scalar_dir / "17_cmrr_mbep2d_diff_ap_tdi.nii_md.nii.gz"
    
    # Output path
    output_dir = base_dir / "metrics_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CSTTOOL METRICS MODULE TEST")
    print("="*70)
    
    # =================================================================
    # STEP 1: Load tractograms
    # =================================================================
    print("\nSTEP 1: Loading CST tractograms...")
    print("-" * 70)
    
    sft_left = load_tractogram(str(trk_left), 'same')
    left_streamlines = sft_left.streamlines
    print(f"✓ Left CST loaded: {len(left_streamlines)} streamlines")
    
    sft_right = load_tractogram(str(trk_right), 'same')
    right_streamlines = sft_right.streamlines
    print(f"✓ Right CST loaded: {len(right_streamlines)} streamlines")
    
    sft_bilateral = load_tractogram(str(trk_bilateral), 'same')
    bilateral_streamlines = sft_bilateral.streamlines
    print(f"✓ Bilateral CST loaded: {len(bilateral_streamlines)} streamlines")
    
    # Get affine from tractogram
    affine = sft_left.affine
    
    # =================================================================
    # STEP 2: Load scalar maps
    # =================================================================
    print("\nSTEP 2: Loading scalar maps...")
    print("-" * 70)
    
    fa_map, fa_affine = load_nifti(str(fa_map_path))
    print(f"✓ FA map loaded: shape {fa_map.shape}")
    print(f"  FA range: [{fa_map.min():.3f}, {fa_map.max():.3f}]")
    
    md_map, md_affine = load_nifti(str(md_map_path))
    print(f"✓ MD map loaded: shape {md_map.shape}")
    print(f"  MD range: [{md_map.min():.3e}, {md_map.max():.3e}]")
    
    # =================================================================
    # STEP 3: Analyze individual bundles
    # =================================================================
    print("\nSTEP 3: Analyzing individual CST bundles...")
    print("-" * 70)
    
    print("\n--- LEFT CST ---")
    left_metrics = analyze_cst_hemisphere(
        left_streamlines, 
        fa_map=fa_map, 
        md_map=md_map, 
        affine=affine
    )
    print_metrics_summary(left_metrics)
    
    print("\n--- RIGHT CST ---")
    right_metrics = analyze_cst_hemisphere(
        right_streamlines,
        fa_map=fa_map,
        md_map=md_map,
        affine=affine
    )
    print_metrics_summary(right_metrics)
    
    print("\n--- BILATERAL CST ---")
    bilateral_metrics = analyze_cst_hemisphere(
        bilateral_streamlines,
        fa_map=fa_map,
        md_map=md_map,
        affine=affine
    )
    print_metrics_summary(bilateral_metrics)
    
    # =================================================================
    # STEP 4: Bilateral comparison
    # =================================================================
    print("\nSTEP 4: Comparing bilateral CST...")
    print("-" * 70)
    
    comparison = compare_bilateral_cst(
        left_streamlines,
        right_streamlines,
        fa_map=fa_map,
        md_map=md_map,
        affine=affine
    )
    
    print("\nAsymmetry Metrics:")
    print(f"  Volume laterality index: {comparison['asymmetry']['volume_laterality']:.3f}")
    if 'fa_laterality' in comparison['asymmetry']:
        print(f"  FA laterality index: {comparison['asymmetry']['fa_laterality']:.3f}")
    
    # =================================================================
    # STEP 5: Save comprehensive report
    # =================================================================
    print("\nSTEP 5: Saving comprehensive report...")
    print("-" * 70)
    
    full_report = {
        'subject_id': '17_cmrr',
        'individual_metrics': {
            'left': left_metrics,
            'right': right_metrics,
            'bilateral': bilateral_metrics
        },
        'bilateral_comparison': comparison,
        'input_files': {
            'left_tractogram': str(trk_left),
            'right_tractogram': str(trk_right),
            'bilateral_tractogram': str(trk_bilateral),
            'fa_map': str(fa_map_path),
            'md_map': str(md_map_path)
        }
    }
    
    report_path = output_dir / "cst_metrics_full_report.json"
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"✓ Full report saved to: {report_path}")
    
    # Save individual CSV files for easier analysis
    save_metrics_csv(left_metrics, output_dir / "left_cst_metrics.csv")
    save_metrics_csv(right_metrics, output_dir / "right_cst_metrics.csv")
    save_metrics_csv(bilateral_metrics, output_dir / "bilateral_cst_metrics.csv")
    
    # =================================================================
    # STEP 6: Visualize tract profiles
    # =================================================================
    print("\nSTEP 6: Creating visualizations...")
    print("-" * 70)
    
    # Plot tract profiles
    visualize_tract_profiles(
        left_metrics, 
        right_metrics,
        output_dir / "cst_tract_profiles.png"
    )
    
    # Plot asymmetry
    visualize_asymmetry(
        comparison,
        output_dir / "cst_asymmetry.png"
    )
    
    print("\n" + "="*70)
    print("METRICS TEST COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")


def save_metrics_csv(metrics_dict, output_path):
    """Save metrics to CSV format for easy analysis."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        
        # Morphology
        writer.writerow(['n_streamlines', metrics_dict['morphology']['n_streamlines']])
        writer.writerow(['mean_length_mm', metrics_dict['morphology']['mean_length']])
        writer.writerow(['tract_volume_mm3', metrics_dict['morphology']['tract_volume']])
        
        # FA metrics
        if 'fa' in metrics_dict:
            writer.writerow(['fa_mean', metrics_dict['fa']['mean']])
            writer.writerow(['fa_std', metrics_dict['fa']['std']])
            writer.writerow(['fa_median', metrics_dict['fa']['median']])
        
        # MD metrics
        if 'md' in metrics_dict:
            writer.writerow(['md_mean', metrics_dict['md']['mean']])
            writer.writerow(['md_std', metrics_dict['md']['std']])
            writer.writerow(['md_median', metrics_dict['md']['median']])
    
    print(f"✓ CSV metrics saved to: {output_path}")


def visualize_tract_profiles(left_metrics, right_metrics, output_path):
    """Create visualization of FA and MD tract profiles."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # FA profile
    if 'fa' in left_metrics and 'fa' in right_metrics:
        left_fa_profile = left_metrics['fa']['profile']
        right_fa_profile = right_metrics['fa']['profile']
        
        x = np.linspace(0, 100, len(left_fa_profile))
        
        axes[0].plot(x, left_fa_profile, 'b-', linewidth=2, label='Left CST')
        axes[0].plot(x, right_fa_profile, 'r-', linewidth=2, label='Right CST')
        axes[0].fill_between(x, left_fa_profile, alpha=0.3, color='blue')
        axes[0].fill_between(x, right_fa_profile, alpha=0.3, color='red')
        axes[0].set_xlabel('Normalized Position Along Tract (%)', fontsize=12)
        axes[0].set_ylabel('Fractional Anisotropy (FA)', fontsize=12)
        axes[0].set_title('FA Tract Profile', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
    
    # MD profile
    if 'md' in left_metrics and 'md' in right_metrics:
        left_md_profile = left_metrics['md']['profile']
        right_md_profile = right_metrics['md']['profile']
        
        x = np.linspace(0, 100, len(left_md_profile))
        
        axes[1].plot(x, left_md_profile, 'b-', linewidth=2, label='Left CST')
        axes[1].plot(x, right_md_profile, 'r-', linewidth=2, label='Right CST')
        axes[1].fill_between(x, left_md_profile, alpha=0.3, color='blue')
        axes[1].fill_between(x, right_md_profile, alpha=0.3, color='red')
        axes[1].set_xlabel('Normalized Position Along Tract (%)', fontsize=12)
        axes[1].set_ylabel('Mean Diffusivity (MD) [mm²/s]', fontsize=12)
        axes[1].set_title('MD Tract Profile', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Tract profiles visualization saved to: {output_path}")


def visualize_asymmetry(comparison, output_path):
    """Create bar plot comparing left vs right CST metrics."""
    
    left = comparison['left']
    right = comparison['right']
    
    # Prepare data
    metrics_names = ['Volume\n(mm³)', 'Mean\nLength\n(mm)', 'Mean\nFA', 'Mean\nMD\n(×10⁻³)']
    left_values = [
        left['morphology']['tract_volume'],
        left['morphology']['mean_length'],
        left['fa']['mean'] if 'fa' in left else 0,
        left['md']['mean'] * 1000 if 'md' in left else 0  # Convert to 10^-3
    ]
    right_values = [
        right['morphology']['tract_volume'],
        right['morphology']['mean_length'],
        right['fa']['mean'] if 'fa' in right else 0,
        right['md']['mean'] * 1000 if 'md' in right else 0
    ]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, left_values, width, label='Left CST', color='steelblue')
    bars2 = ax.bar(x + width/2, right_values, width, label='Right CST', color='coral')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Bilateral CST Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Asymmetry visualization saved to: {output_path}")


if __name__ == "__main__":
    main()