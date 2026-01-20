
import argparse
from pathlib import Path
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti

from csttool.metrics import (
    analyze_cst_hemisphere,
    compare_bilateral_cst,
    print_hemisphere_summary,
    plot_tract_profiles,
    plot_bilateral_comparison,
    plot_stacked_profiles,
    plot_tractogram_qc_preview,
)
from csttool.metrics.modules.reports import (
    save_json_report,
    save_csv_summary,
    save_html_report,
    save_pdf_report,
)

def cmd_metrics(args: argparse.Namespace) -> dict | None:
    """
    Compute bilateral CST metrics and generate reports.
    """
    verbose = getattr(args, 'verbose', True)
    
    # Validate inputs
    if not args.cst_left.exists():
        print(f"Error: Left CST tractogram not found: {args.cst_left}")
        return None
    
    if not args.cst_right.exists():
        print(f"Error: Right CST tractogram not found: {args.cst_right}")
        return None
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Load tractograms
    print(f"Loading left CST: {args.cst_left}")
    try:
        sft_left = load_tractogram(str(args.cst_left), 'same')
        streamlines_left = sft_left.streamlines
        print(f"  Loaded {len(streamlines_left):,} streamlines")
    except Exception as e:
        print(f"Error loading left CST: {e}")
        return None
    
    print(f"Loading right CST: {args.cst_right}")
    try:
        sft_right = load_tractogram(str(args.cst_right), 'same')
        streamlines_right = sft_right.streamlines
        print(f"  Loaded {len(streamlines_right):,} streamlines")
    except Exception as e:
        print(f"Error loading right CST: {e}")
        return None
    
    # Load scalar maps
    fa_map, fa_affine = None, None
    md_map = None
    
    if args.fa:
        if args.fa.exists():
            print(f"Loading FA map: {args.fa}")
            fa_map, fa_affine = load_nifti(str(args.fa))
        else:
            print(f"Warning: FA map not found: {args.fa}")
    
    if args.md:
        if args.md.exists():
            print(f"Loading MD map: {args.md}")
            md_map, _ = load_nifti(str(args.md))
        else:
            print(f"Warning: MD map not found: {args.md}")
    
    # Load RD and AD maps if provided
    rd_map = None
    ad_map = None
    
    if getattr(args, 'rd', None):
        if args.rd.exists():
            print(f"Loading RD map: {args.rd}")
            rd_map, _ = load_nifti(str(args.rd))
        else:
            print(f"Warning: RD map not found: {args.rd}")
    
    if getattr(args, 'ad', None):
        if args.ad.exists():
            print(f"Loading AD map: {args.ad}")
            ad_map, _ = load_nifti(str(args.ad))
        else:
            print(f"Warning: AD map not found: {args.ad}")
    
    affine = fa_affine if fa_affine is not None else sft_left.affine
    
    # Analyze hemispheres
    print("\n" + "="*60)
    print("Analyzing LEFT CST")
    print("="*60)
    try:
        left_metrics = analyze_cst_hemisphere(
            streamlines=streamlines_left,
            fa_map=fa_map,
            md_map=md_map,
            rd_map=rd_map,
            ad_map=ad_map,
            affine=affine,
            hemisphere='left'
        )
        if verbose:
            print_hemisphere_summary(left_metrics)
    except Exception as e:
        print(f"Error analyzing left CST: {e}")
        return None
    
    print("\n" + "="*60)
    print("Analyzing RIGHT CST")
    print("="*60)
    try:
        right_metrics = analyze_cst_hemisphere(
            streamlines=streamlines_right,
            fa_map=fa_map,
            md_map=md_map,
            rd_map=rd_map,
            ad_map=ad_map,
            affine=affine,
            hemisphere='right'
        )
        if verbose:
            print_hemisphere_summary(right_metrics)
    except Exception as e:
        print(f"Error analyzing right CST: {e}")
        return None
    
    # Bilateral comparison
    print("\n" + "="*60)
    print("Computing bilateral comparison")
    print("="*60)
    try:
        comparison = compare_bilateral_cst(left_metrics, right_metrics)
    except Exception as e:
        print(f"Error during bilateral comparison: {e}")
        return None
    
    # Save reports
    print("\n" + "="*60)
    print("Generating reports")
    print("="*60)
    
    # Build metadata for reports
    metadata = {
        'acquisition': {},  # Populated by cmd_run when full pipeline is used
        'processing': {
            'tracking_method': 'Deterministic (DTI)',
            'roi_approach': 'Atlas-to-Subject (HO)',
        },
        'qc_thresholds': {
            'fa_threshold': getattr(args, 'fa_threshold', None),
            'min_length': getattr(args, 'min_length', None),
            'max_length': getattr(args, 'max_length', None),
        }
    }
    
    # Remove None values from qc_thresholds
    metadata['qc_thresholds'] = {k: v for k, v in metadata['qc_thresholds'].items() if v is not None}
    
    pipeline_metadata = getattr(args, 'pipeline_metadata', {})
    if pipeline_metadata:
        # Deep merge or specific updates
        if 'preprocessing' in pipeline_metadata:
            metadata['processing']['preprocessing_step'] = pipeline_metadata['preprocessing']
        if 'acquisition' in pipeline_metadata:
            metadata['acquisition'] = pipeline_metadata['acquisition']
        if 'tracking' in pipeline_metadata:
            metadata['processing']['tracking_params'] = pipeline_metadata['tracking']
    
    json_path = None
    csv_path = None
    
    try:
        json_path = save_json_report(comparison, args.out, args.subject_id, metadata=metadata)
        print(f"✓ JSON report: {json_path}")
    except Exception as e:
        print(f"Error saving JSON report: {e}")
    
    try:
        csv_path = save_csv_summary(comparison, args.out, args.subject_id)
        print(f"✓ CSV summary: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV summary: {e}")
    
    # Generate visualizations
    viz_dir = args.out / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    viz_paths = {}
    
    try:
        if fa_map is not None and 'fa' in left_metrics:
            viz_paths['tract_profiles'] = plot_tract_profiles(
                left_metrics, right_metrics, viz_dir, args.subject_id, scalar='fa'
            )
            print(f"✓ Tract profiles: {viz_paths['tract_profiles']}")
    except Exception as e:
        print(f"Warning: Could not generate tract profiles: {e}")
    
    try:
        viz_paths['bilateral_comparison'] = plot_bilateral_comparison(
            comparison, viz_dir, args.subject_id
        )
        print(f"✓ Bilateral comparison: {viz_paths['bilateral_comparison']}")
    except Exception as e:
        print(f"Warning: Could not generate bilateral comparison: {e}")

    # Generate additional visualizations for PDF if requested
    if getattr(args, 'generate_pdf', False) or getattr(args, 'save_visualizations', False):
        try:
            viz_paths['stacked_profiles'] = plot_stacked_profiles(
                left_metrics, right_metrics, viz_dir, args.subject_id
            )
            print(f"✓ Stacked profiles: {viz_paths['stacked_profiles']}")
        except Exception as e:
            print(f"Warning: Could not generate stacked profiles: {e}")
            
        try:
            # For QC preview, we need background image (FA) and affine
            # If FA is not available, we can try to use a dummy or skip
            if fa_map is not None:
                bg_img = fa_map
                bg_affine = affine
            else:
                 # TODO: Handle case where FA is missing more gracefully if possible
                 # For now, skip if no background
                 bg_img = None
                 bg_affine = None
            
            if bg_img is not None:
                for view in ['axial', 'sagittal', 'coronal']:
                    viz_paths[f'tractogram_qc_{view}'] = plot_tractogram_qc_preview(
                        streamlines_left,
                        streamlines_right,
                        bg_img,
                        bg_affine,
                        viz_dir,
                        args.subject_id,
                        slice_type=view,
                        set_title=False  # HTML template adds titles
                    )
                    print(f"✓ Tractogram QC ({view}): {viz_paths[f'tractogram_qc_{view}']}")
        except Exception as e:
            print(f"Warning: Could not generate tractogram QC: {e}")
    
    # Generate PDF if requested
    pdf_path = None
    html_path = None
    if getattr(args, 'generate_pdf', False):
        try:
            # Use new HTML→PDF pipeline with metadata
            space = getattr(args, 'space', "Native Space")
            
            # First generate HTML report
            html_path = save_html_report(
                comparison, 
                viz_paths, 
                args.out, 
                args.subject_id,
                space=space,
                metadata=metadata
            )
            print(f"✓ HTML report: {html_path}")
            
            # Then convert to PDF
            pdf_path = save_pdf_report(
                comparison, 
                viz_paths, 
                args.out, 
                args.subject_id,
                space=space,
                html_path=html_path
            )
            if pdf_path:
                print(f"✓ PDF report: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not generate report: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("METRICS COMPLETE")
    print(f"{'='*60}")
    print(f"Subject: {args.subject_id}")
    print(f"\nMorphology:")
    print(f"  Left CST:  {left_metrics['morphology']['n_streamlines']:,} streamlines, "
          f"volume = {left_metrics['morphology']['tract_volume']/1000:.2f} cm³")
    print(f"  Right CST: {right_metrics['morphology']['n_streamlines']:,} streamlines, "
          f"volume = {right_metrics['morphology']['tract_volume']/1000:.2f} cm³")
    
    if 'fa' in left_metrics:
        print(f"\nFA:")
        print(f"  Left:  {left_metrics['fa']['mean']:.3f} ± {left_metrics['fa']['std']:.3f}")
        print(f"  Right: {right_metrics['fa']['mean']:.3f} ± {right_metrics['fa']['std']:.3f}")
        if 'fa' in comparison['asymmetry']:
            print(f"  LI:    {comparison['asymmetry']['fa']['laterality_index']:.3f}")
    
    print(f"{'='*60}")
    
    return {
        'json_path': json_path,
        'csv_path': csv_path,
        'pdf_path': pdf_path,
        'comparison': comparison
    }
