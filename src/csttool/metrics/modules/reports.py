"""
report.py

Report generation functions for CST metrics.

This module generates:
- JSON reports (complete machine-readable data)
- CSV summaries (metrics table for Excel/R analysis)
- PDF clinical reports (human-readable summary with visualizations)
"""

import json
import csv
import base64
from pathlib import Path
from datetime import datetime
import numpy as np

from jinja2 import Environment, FileSystemLoader

# Import version from package
from csttool import __version__

# Module-level template environment
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))


def _embed_image(path):
    """Embed image as base64 data URI.
    
    Parameters
    ----------
    path : str or Path or None
        Path to image file
        
    Returns
    -------
    str or None
        Base64 data URI or None if path invalid
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = path.suffix.lower()
        mime = 'image/png' if ext == '.png' else 'image/jpeg'
        return f'data:{mime};base64,{data}'
    except Exception:
        return None


def save_json_report(comparison, output_dir, subject_id):
    """
    Save comprehensive metrics report as JSON.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    json_path : Path
        Path to saved JSON file
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    report = {
        'subject_id': subject_id,
        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': comparison
    }
    
    # Save JSON
    json_path = output_dir / f"{subject_id}_bilateral_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ JSON report saved: {json_path}")
    return json_path


def save_csv_summary(comparison, output_dir, subject_id):
    """
    Save metrics summary as CSV table.
    
    Creates a flat CSV file with key metrics suitable for group analysis.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    csv_path : Path
        Path to saved CSV file
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    # Prepare data row
    data = {
        'subject_id': subject_id,
        'processing_date': datetime.now().strftime("%Y-%m-%d"),
        
        # Left morphology
        'left_n_streamlines': left['morphology']['n_streamlines'],
        'left_mean_length_mm': left['morphology']['mean_length'],
        'left_tract_volume_mm3': left['morphology']['tract_volume'],
        
        # Right morphology
        'right_n_streamlines': right['morphology']['n_streamlines'],
        'right_mean_length_mm': right['morphology']['mean_length'],
        'right_tract_volume_mm3': right['morphology']['tract_volume'],
        
        # Asymmetry
        'volume_laterality_index': asym['volume']['laterality_index'],
        'streamline_count_laterality_index': asym['streamline_count']['laterality_index'],
    }
    
    # Add FA if available
    if 'fa' in left:
        data.update({
            'left_fa_mean': left['fa']['mean'],
            'left_fa_std': left['fa']['std'],
            'right_fa_mean': right['fa']['mean'],
            'right_fa_std': right['fa']['std'],
            'fa_laterality_index': asym['fa']['laterality_index'],
        })
    
    # Add MD if available
    if 'md' in left:
        data.update({
            'left_md_mean': left['md']['mean'],
            'left_md_std': left['md']['std'],
            'right_md_mean': right['md']['mean'],
            'right_md_std': right['md']['std'],
            'md_laterality_index': asym['md']['laterality_index'],
        })
    
    # Add RD if available
    if 'rd' in left:
        data.update({
            'left_rd_mean': left['rd']['mean'],
            'left_rd_std': left['rd']['std'],
            'right_rd_mean': right['rd']['mean'],
            'right_rd_std': right['rd']['std'],
            'rd_laterality_index': asym['rd']['laterality_index'],
        })
    
    # Add AD if available
    if 'ad' in left:
        data.update({
            'left_ad_mean': left['ad']['mean'],
            'left_ad_std': left['ad']['std'],
            'right_ad_mean': right['ad']['mean'],
            'right_ad_std': right['ad']['std'],
            'ad_laterality_index': asym['ad']['laterality_index'],
        })
    
    # Save CSV
    csv_path = output_dir / f"{subject_id}_metrics_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    
    print(f"✓ CSV summary saved: {csv_path}")
    return csv_path


def save_html_report(
    comparison,
    visualization_paths,
    output_dir,
    subject_id,
    version=None,
    space="Native Space",
    metadata=None
):
    """
    Generate HTML report using Jinja2 template.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    visualization_paths : dict
        Paths to generated visualizations
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
    version : str, optional
        csttool version (defaults to package version)
    space : str
        Space declaration (e.g., "Native Space")
    metadata : dict, optional
        Acquisition and processing metadata
        
    Returns
    -------
    html_path : Path
        Path to saved HTML file
    """
    if version is None:
        version = __version__
    
    if metadata is None:
        metadata = {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract comparison data
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    # Build metrics list for template
    metrics = []
    
    # Streamlines
    metrics.append({
        "label": "Streamlines",
        "left": left['morphology']['n_streamlines'],
        "right": right['morphology']['n_streamlines'],
        "li": asym['streamline_count']['laterality_index']
    })
    
    # Volume (convert mm³ to cm³)
    metrics.append({
        "label": "Volume (cm³)",
        "left": f"{left['morphology']['tract_volume'] / 1000.0:.2f}",
        "right": f"{right['morphology']['tract_volume'] / 1000.0:.2f}",
        "li": asym['volume']['laterality_index']
    })
    
    # Length
    metrics.append({
        "label": "Length (mm)",
        "left": f"{left['morphology']['mean_length']:.1f}",
        "right": f"{right['morphology']['mean_length']:.1f}",
        "li": asym['mean_length']['laterality_index']
    })
    
    # FA
    if 'fa' in left:
        metrics.append({
            "label": "FA",
            "left": f"{left['fa']['mean']:.3f}",
            "right": f"{right['fa']['mean']:.3f}",
            "li": asym['fa']['laterality_index']
        })
    
    # MD (×10⁻³ mm²/s)
    if 'md' in left:
        metrics.append({
            "label": "MD (×10⁻³ mm²/s)",
            "left": f"{left['md']['mean']*1000:.2f}",
            "right": f"{right['md']['mean']*1000:.2f}",
            "li": asym['md']['laterality_index']
        })
    
    # RD (×10⁻³ mm²/s)
    if 'rd' in left:
        metrics.append({
            "label": "RD (×10⁻³ mm²/s)",
            "left": f"{left['rd']['mean']*1000:.2f}",
            "right": f"{right['rd']['mean']*1000:.2f}",
            "li": asym['rd']['laterality_index']
        })
    
    # AD (×10⁻³ mm²/s)
    if 'ad' in left:
        metrics.append({
            "label": "AD (×10⁻³ mm²/s)",
            "left": f"{left['ad']['mean']*1000:.2f}",
            "right": f"{right['ad']['mean']*1000:.2f}",
            "li": asym['ad']['laterality_index']
        })
    
    # Build visualization data for template
    viz = {
        "stacked_profiles": _embed_image(visualization_paths.get('stacked_profiles')),
        "profiles": [
            {"title": "Fractional Anisotropy", "data": _embed_image(visualization_paths.get('profile_fa'))},
            {"title": "Mean Diffusivity", "data": _embed_image(visualization_paths.get('profile_md'))},
            {"title": "Radial Diffusivity", "data": _embed_image(visualization_paths.get('profile_rd'))},
            {"title": "Axial Diffusivity", "data": _embed_image(visualization_paths.get('profile_ad'))},
        ],
        "tractograms": [
            {"title": "Axial", "data": _embed_image(visualization_paths.get('tractogram_qc_axial'))},
            {"title": "Sagittal", "data": _embed_image(visualization_paths.get('tractogram_qc_sagittal'))},
            {"title": "Coronal", "data": _embed_image(visualization_paths.get('tractogram_qc_coronal'))},
        ]
    }
    
    # Get acquisition/processing metadata with defaults
    acquisition = metadata.get('acquisition', {})
    processing = metadata.get('processing', {})
    
    # Format whole brain streamline count
    whole_brain = processing.get('whole_brain_streamlines', 'N/A')
    if whole_brain != 'N/A' and isinstance(whole_brain, (int, float)):
        processing = {**processing, 'whole_brain': f'{int(whole_brain):,} streamlines'}
    else:
        processing = {**processing, 'whole_brain': whole_brain}
    
    # Load template and CSS
    template = _jinja_env.get_template("report.html.j2")
    css = (_TEMPLATE_DIR / "report.css").read_text()
    
    # Render template with context
    html_content = template.render(
        subject_id=subject_id,
        date=datetime.now().strftime("%Y-%m-%d"),
        version=version,
        space=space,
        css=css,
        metrics=metrics,
        viz=viz,
        acquisition=acquisition,
        processing=processing,
    )
    
    html_path = output_dir / f"{subject_id}_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    
    print(f"✓ HTML report saved: {html_path}")
    return html_path

def html_to_pdf(html_file, pdf_file):
    """Convert HTML to single-page A4 PDF.
    
    Parameters
    ----------
    html_file : Path
        Path to the HTML file to convert.
    pdf_file : Path
        Path to the PDF file to save.
        
    Returns
    -------
    Path or None
        Path to generated PDF, or None if WeasyPrint not installed.
    """
    try:
        from weasyprint import HTML
        from weasyprint.text.fonts import FontConfiguration
        
        font_config = FontConfiguration()
        html = HTML(filename=str(html_file))
        
        # Render PDF using the embedded CSS from the template
        html.write_pdf(
            str(pdf_file),
            font_config=font_config,
            presentational_hints=True
        )
        
        print(f"✓ PDF generated: {pdf_file}")
        return pdf_file
        
    except ImportError:
        print("⚠️ Install weasyprint: pip install weasyprint")
        return None

def save_pdf_report(
    comparison,
    visualization_paths,
    output_dir,
    subject_id,
    version=None,
    space="Native Space",
    html_path=None
):
    """
    Generate PDF report by converting HTML report.
    """
    output_dir = Path(output_dir)
    pdf_path = output_dir / f"{subject_id}_report.pdf"
    
    if html_path is None:
        # Fallback: try to find the standard HTML report
        possible_html = output_dir / f"{subject_id}_report.html"
        if possible_html.exists():
            html_path = possible_html
        else:
            print("⚠️ HTML report not found for PDF conversion. Generating temporary HTML...")
            # We would need to call save_html_report here, but let's rely on caller
            return None

    return html_to_pdf(html_path, pdf_path)

def generate_complete_report(
    comparison,
    streamlines_left,
    streamlines_right,
    fa_map,
    affine,
    output_dir,
    subject_id,
    background_image=None,
    version=None,
    space="Native Space"
):
    """
    Generate all report formats: JSON, CSV, and PDF with visualizations.
    
    Parameters
    ----------
    comparison : dict
        Bilateral comparison metrics
    streamlines_left : Streamlines
        Left CST streamlines
    streamlines_right : Streamlines
        Right CST streamlines
    fa_map : ndarray
        3D FA map
    affine : ndarray
        4x4 affine transformation
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
    background_image : ndarray, optional
        3D T1 or FA image for tractogram QC background (defaults to fa_map)
    version : str
        csttool version string
        
    Returns
    -------
    report_paths : dict
        Dictionary of all generated report file paths
    """
    
    from .visualizations import (
        plot_tract_profiles,
        plot_bilateral_comparison,
        create_summary_figure,
        plot_stacked_profiles,
        plot_tractogram_qc_preview
    )
    
    # Use package version if not specified
    if version is None:
        version = __version__
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating complete report package...")
    
    # Use FA map as background if no background image provided
    if background_image is None:
        background_image = fa_map
    
    # Generate visualizations for PDF (new single-page layout)
    pdf_viz_paths = {}
    
    # Stacked FA/MD profiles for PDF
    pdf_viz_paths['stacked_profiles'] = plot_stacked_profiles(
        comparison['left'],
        comparison['right'],
        viz_dir,
        subject_id
    )
    
    # Tractogram QC preview for PDF (Axial, Sagittal, Coronal)
    for view in ['axial', 'sagittal', 'coronal']:
        pdf_viz_paths[f'tractogram_qc_{view}'] = plot_tractogram_qc_preview(
            streamlines_left,
            streamlines_right,
            background_image,
            affine,
            viz_dir,
            subject_id,
            slice_type=view,
            set_title=False
        )
    
    # Also generate individual plots for detailed analysis
    viz_paths = {}
    
    if 'fa' in comparison['left']:
        viz_paths['tract_profiles_fa'] = plot_tract_profiles(
            comparison['left'],
            comparison['right'],
            viz_dir,
            subject_id,
            scalar='fa'
        )
    
    if 'md' in comparison['left']:
        viz_paths['tract_profiles_md'] = plot_tract_profiles(
            comparison['left'],
            comparison['right'],
            viz_dir,
            subject_id,
            scalar='md'
        )
    
    viz_paths['bilateral_comparison'] = plot_bilateral_comparison(
        comparison,
        viz_dir,
        subject_id
    )
    
    viz_paths['summary'] = create_summary_figure(
        comparison,
        streamlines_left,
        streamlines_right,
        fa_map,
        affine,
        viz_dir,
        subject_id
    )
    
    # Merge all visualization paths
    viz_paths.update(pdf_viz_paths)
    
    # Generate reports
    # 1. HTML Report (now critical as PDF is derived from it)
    html_path = save_html_report(comparison, pdf_viz_paths, output_dir, subject_id, version, space)
    
    # 2. PDF Report (from HTML)
    pdf_path = save_pdf_report(comparison, pdf_viz_paths, output_dir, subject_id, version, space, html_path)
    
    report_paths = {
        'json': save_json_report(comparison, output_dir, subject_id),
        'csv': save_csv_summary(comparison, output_dir, subject_id),
        'html': html_path,
        'pdf': pdf_path,
        'visualizations': viz_paths
    }
    
    print("\n✅ Complete report package generated!")
    return report_paths