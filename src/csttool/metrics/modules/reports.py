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
from pathlib import Path
from datetime import datetime
import numpy as np


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
    
    # Save CSV
    csv_path = output_dir / f"{subject_id}_metrics_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    
    print(f"✓ CSV summary saved: {csv_path}")
    return csv_path


def save_pdf_report(
    comparison,
    visualization_paths,
    output_dir,
    subject_id
):
    """
    Generate clinical PDF report with metrics and visualizations.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    visualization_paths : dict
        Paths to generated visualization figures
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    pdf_path : Path
        Path to saved PDF file
    """
    
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        print("⚠️  reportlab not installed. Skipping PDF generation.")
        print("   Install with: pip install reportlab")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f"{subject_id}_clinical_report.pdf"
    
    # Create document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2196F3'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1976D2'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph(f"CST Analysis Report", title_style))
    story.append(Paragraph(f"Subject: {subject_id}", styles['Normal']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary Table
    story.append(Paragraph("Summary Metrics", heading_style))
    
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    table_data = [
        ['Metric', 'Left CST', 'Right CST', 'Laterality Index', 'Interpretation'],
        [
            'Streamline Count',
            str(left['morphology']['n_streamlines']),
            str(right['morphology']['n_streamlines']),
            f"{asym['streamline_count']['laterality_index']:+.3f}",
            asym['streamline_count']['interpretation']
        ],
        [
            'Tract Volume (mm³)',
            f"{left['morphology']['tract_volume']:.0f}",
            f"{right['morphology']['tract_volume']:.0f}",
            f"{asym['volume']['laterality_index']:+.3f}",
            asym['volume']['interpretation']
        ],
        [
            'Mean Length (mm)',
            f"{left['morphology']['mean_length']:.1f}",
            f"{right['morphology']['mean_length']:.1f}",
            f"{asym['mean_length']['laterality_index']:+.3f}",
            asym['mean_length']['interpretation']
        ]
    ]
    
    if 'fa' in left:
        table_data.append([
            'Mean FA',
            f"{left['fa']['mean']:.3f}",
            f"{right['fa']['mean']:.3f}",
            f"{asym['fa']['laterality_index']:+.3f}",
            asym['fa']['interpretation']
        ])
    
    if 'md' in left:
        table_data.append([
            'Mean MD (×10⁻³)',
            f"{left['md']['mean']*1000:.2f}",
            f"{right['md']['mean']*1000:.2f}",
            f"{asym['md']['laterality_index']:+.3f}",
            asym['md']['interpretation']
        ])
    
    table = Table(table_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Interpretation
    story.append(Paragraph("Clinical Interpretation", heading_style))
    interpretation_text = generate_interpretation_text(asym)
    story.append(Paragraph(interpretation_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Add visualizations if available
    if visualization_paths:
        story.append(PageBreak())
        story.append(Paragraph("Visualizations", heading_style))
        
        for viz_name, viz_path in visualization_paths.items():
            if viz_path and Path(viz_path).exists():
                try:
                    img = Image(str(viz_path), width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    print(f"⚠️  Could not add image {viz_path}: {e}")
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF report saved: {pdf_path}")
    return pdf_path


def generate_interpretation_text(asymmetry):
    """
    Generate clinical interpretation text based on asymmetry metrics.
    
    Parameters
    ----------
    asymmetry : dict
        Asymmetry metrics from bilateral comparison
        
    Returns
    -------
    text : str
        Clinical interpretation text
    """
    
    significant_asymmetries = []
    
    # Check each metric
    for metric_name, metric_data in asymmetry.items():
        if isinstance(metric_data, dict) and 'laterality_index' in metric_data:
            li = abs(metric_data['laterality_index'])
            if li > 0.10:  # Threshold for significance
                direction = "left" if metric_data['laterality_index'] > 0 else "right"
                significant_asymmetries.append(f"{metric_name} ({direction}-sided, LI={metric_data['laterality_index']:+.3f})")
    
    if not significant_asymmetries:
        text = ("Bilateral CST analysis reveals symmetric white matter integrity between hemispheres. "
                "All laterality indices are within normal ranges (|LI| < 0.10), suggesting "
                "balanced corticospinal tract development and preservation.")
    else:
        text = ("Bilateral CST analysis reveals the following asymmetries: " +
                ", ".join(significant_asymmetries) + ". "
                "These findings may warrant clinical correlation and follow-up assessment.")
    
    return text


def generate_complete_report(
    comparison,
    streamlines_left,
    streamlines_right,
    fa_map,
    affine,
    output_dir,
    subject_id
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
        
    Returns
    -------
    report_paths : dict
        Dictionary of all generated report file paths
    """
    
    from .visualizations import (
        plot_tract_profiles,
        plot_bilateral_comparison,
        create_summary_figure
    )
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating complete report package...")
    
    # Generate visualizations
    viz_paths = {}
    
    if 'fa' in comparison['left']:
        viz_paths['tract_profiles'] = plot_tract_profiles(
            comparison['left'],
            comparison['right'],
            viz_dir,
            subject_id,
            scalar='fa'
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
    
    # Generate reports
    report_paths = {
        'json': save_json_report(comparison, output_dir, subject_id),
        'csv': save_csv_summary(comparison, output_dir, subject_id),
        'pdf': save_pdf_report(comparison, viz_paths, output_dir, subject_id),
        'visualizations': viz_paths
    }
    
    print("\n✅ Complete report package generated!")
    return report_paths