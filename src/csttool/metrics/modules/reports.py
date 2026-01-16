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

# Import version from package
from csttool import __version__


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
    Generate HTML report based on mockup design.
    
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
    import base64
    
    if version is None:
        version = __version__
    
    if metadata is None:
        metadata = {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    # Get metadata with defaults
    acquisition = metadata.get('acquisition', {})
    processing = metadata.get('processing', {})
    
    # Calculate metrics
    left_count = left['morphology']['n_streamlines']
    right_count = right['morphology']['n_streamlines']
    
    def format_li(li_value):
        """Format LI with appropriate CSS class."""
        if li_value > 0:
            return f'<td class="li-positive">+{li_value:.3f}</td>'
        else:
            return f'<td class="li-negative">{li_value:.3f}</td>'
    
    def embed_image(path):
        """Embed image as base64 data URI."""
        if path is None or not Path(path).exists():
            return None
        try:
            with open(path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            ext = Path(path).suffix.lower()
            mime = 'image/png' if ext == '.png' else 'image/jpeg'
            return f'data:{mime};base64,{data}'
        except Exception:
            return None
    
    # Embed visualizations - individual profile plots
    fa_profile = embed_image(visualization_paths.get('profile_fa'))
    md_profile = embed_image(visualization_paths.get('profile_md'))
    rd_profile = embed_image(visualization_paths.get('profile_rd'))
    ad_profile = embed_image(visualization_paths.get('profile_ad'))
    # Fallback to stacked profiles if individual not available
    if not any([fa_profile, md_profile, rd_profile, ad_profile]):
        stacked_profile = embed_image(visualization_paths.get('stacked_profiles'))
    else:
        stacked_profile = None
    
    qc_axial = embed_image(visualization_paths.get('tractogram_qc_axial'))
    qc_sagittal = embed_image(visualization_paths.get('tractogram_qc_sagittal'))
    qc_coronal = embed_image(visualization_paths.get('tractogram_qc_coronal'))
    
    # Build metrics table rows
    metrics_rows = []
    
    # Streamlines
    metrics_rows.append(f'''<tr>
                    <td>Streamlines</td>
                    <td>{left_count}</td>
                    <td>{right_count}</td>
                    {format_li(asym['streamline_count']['laterality_index'])}
                </tr>''')
    
    # Volume (convert mm³ to cm³)
    left_vol = left['morphology']['tract_volume'] / 1000.0
    right_vol = right['morphology']['tract_volume'] / 1000.0
    metrics_rows.append(f'''<tr>
                    <td>Volume (cm³)</td>
                    <td>{left_vol:.2f}</td>
                    <td>{right_vol:.2f}</td>
                    {format_li(asym['volume']['laterality_index'])}
                </tr>''')
    
    # Length
    metrics_rows.append(f'''<tr>
                    <td>Length (mm)</td>
                    <td>{left['morphology']['mean_length']:.1f}</td>
                    <td>{right['morphology']['mean_length']:.1f}</td>
                    {format_li(asym['mean_length']['laterality_index'])}
                </tr>''')
    
    # FA
    if 'fa' in left:
        metrics_rows.append(f'''<tr>
                    <td>FA</td>
                    <td>{left['fa']['mean']:.3f}</td>
                    <td>{right['fa']['mean']:.3f}</td>
                    {format_li(asym['fa']['laterality_index'])}
                </tr>''')
    
    # MD (×10⁻³ mm²/s)
    if 'md' in left:
        metrics_rows.append(f'''<tr>
                    <td>MD (×10⁻³ mm²/s)</td>
                    <td>{left['md']['mean']*1000:.2f}</td>
                    <td>{right['md']['mean']*1000:.2f}</td>
                    {format_li(asym['md']['laterality_index'])}
                </tr>''')
    
    # RD (×10⁻³ mm²/s)
    if 'rd' in left:
        metrics_rows.append(f'''<tr>
                    <td>RD (×10⁻³ mm²/s)</td>
                    <td>{left['rd']['mean']*1000:.2f}</td>
                    <td>{right['rd']['mean']*1000:.2f}</td>
                    {format_li(asym['rd']['laterality_index'])}
                </tr>''')
    
    # AD (×10⁻³ mm²/s)
    if 'ad' in left:
        metrics_rows.append(f'''<tr>
                    <td>AD (×10⁻³ mm²/s)</td>
                    <td>{left['ad']['mean']*1000:.2f}</td>
                    <td>{right['ad']['mean']*1000:.2f}</td>
                    {format_li(asym['ad']['laterality_index'])}
                </tr>''')
    
    # Build profile visualization HTML
    def make_profile_html(img_data, title, placeholder_text):
        if img_data:
            return f'''<div class="profile-plot">
                    <h4>{title}</h4>
                    <img src="{img_data}" alt="{title}" style="width:100%;height:auto">
                </div>'''
        else:
            return f'''<div class="profile-plot">
                    <h4>{title}</h4>
                    <div class="profile-placeholder">{placeholder_text}</div>
                </div>'''
    
    # Build profiles section
    if stacked_profile:
        # Use single stacked image
        profiles_html = f'''<div class="profile-plot">
                    <h4>Along-Tract Profiles</h4>
                    <img src="{stacked_profile}" alt="Profiles" style="width:100%;height:auto">
                </div>'''
    else:
        # Use individual profile plots
        profiles_html = ''
        profiles_html += make_profile_html(fa_profile, 'Fractional Anisotropy', 
                                           '[FA Profile Plot]<br>0% Pontine → 50% PLIC → 100% Precentral')
        profiles_html += make_profile_html(md_profile, 'Mean Diffusivity',
                                           '[MD Profile Plot]<br>Left CST — Right CST')
        profiles_html += make_profile_html(rd_profile, 'Radial Diffusivity', '[RD Profile Plot]')
        profiles_html += make_profile_html(ad_profile, 'Axial Diffusivity', '[AD Profile Plot]')
    
    # Build tractogram visualization HTML
    def make_tractogram_html(img_data, title, placeholder_text, include_legend=False):
        legend_html = '''<div class="tractogram-legend">
                        <span><span class="legend-line-left"></span> Left CST</span>
                        <span><span class="legend-line-right"></span> Right CST</span>
                    </div>''' if include_legend else ''
        
        if img_data:
            return f'''<div class="tractogram-item">
                    <h4>{title}</h4>
                    <img src="{img_data}" alt="{title}" style="width:100%;height:auto">
                    {legend_html}
                </div>'''
        else:
            return f'''<div class="tractogram-item">
                    <h4>{title}</h4>
                    <div class="tractogram-placeholder">{placeholder_text}</div>
                    {legend_html}
                </div>'''
    
    tractograms_html = ''
    tractograms_html += make_tractogram_html(qc_axial, 'CST Tractogram (Axial)', '[Axial View]')
    tractograms_html += make_tractogram_html(qc_sagittal, 'CST Tractogram (Sagittal)', '[Sagittal View]')
    tractograms_html += make_tractogram_html(qc_coronal, 'CST Tractogram (Coronal)', '[Coronal View]', include_legend=True)
    
    # Get whole brain streamline count
    whole_brain = processing.get('whole_brain_streamlines', 'N/A')
    if whole_brain != 'N/A' and isinstance(whole_brain, (int, float)):
        whole_brain = f'{int(whole_brain):,} streamlines'
    
    # Build HTML document matching the mockup design
    html_content = f'''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CST Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: white;
            padding: 12px;
            max-width: 210mm;
            margin: 0 auto;
            font-size: 10px;
        }}

        @media print {{
            body {{
                padding: 0;
            }}

            @page {{
                size: A4 portrait;
                margin: 10mm;
            }}
        }}

        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 10px;
        }}

        .header h1 {{
            color: #1976D2;
            font-size: 22px;
            margin-bottom: 4px;
        }}

        .header .meta {{
            color: #555;
            font-size: 11px;
        }}

        .extraction-badge {{
            background: #E3F2FD;
            color: #1565C0;
            padding: 4px 12px;
            border-radius: 4px;
            display: inline-block;
            font-weight: 600;
            margin-top: 6px;
            font-size: 10px;
        }}

        /* Parameters Row */
        .params-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
        }}

        .param-box {{
            background: #F8F9FA;
            border-left: 3px solid #2196F3;
            padding: 6px 10px;
            border-radius: 3px;
        }}

        .param-box h3 {{
            color: #1976D2;
            font-size: 10px;
            margin-bottom: 4px;
            font-weight: 600;
        }}

        .param-grid {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 2px 8px;
            font-size: 8px;
        }}

        .param-label {{
            color: #666;
        }}

        .param-value {{
            color: #333;
            font-family: 'Consolas', 'Courier New', monospace;
        }}

        /* Section styling */
        .section {{
            margin-bottom: 10px;
        }}

        .section-title {{
            color: #1976D2;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
        }}

        /* Metrics Table */
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 4px;
            font-size: 10px;
        }}

        .metrics-table thead {{
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
        }}

        .metrics-table th {{
            padding: 6px 10px;
            text-align: center;
            font-weight: 600;
        }}

        .metrics-table th:first-child {{
            text-align: left;
        }}

        .metrics-table td {{
            padding: 5px 10px;
            border-bottom: 1px solid #E0E0E0;
            text-align: center;
        }}

        .metrics-table td:first-child {{
            text-align: left;
            font-weight: 500;
        }}

        .metrics-table tbody tr:nth-child(even) {{
            background: #FAFAFA;
        }}

        .li-positive {{
            color: #D32F2F;
            font-weight: 600;
        }}

        .li-negative {{
            color: #1976D2;
            font-weight: 600;
        }}

        .legend {{
            text-align: center;
            font-size: 9px;
            color: #666;
            margin-top: 4px;
        }}

        .legend .left-gt {{
            color: #D32F2F;
        }}

        .legend .right-gt {{
            color: #1976D2;
        }}

        /* Visualizations Section - Two Column Layout */
        .viz-section {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 12px;
            margin-top: 6px;
        }}

        /* Left column: Along-tract profiles - LARGER */
        .profiles-column {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .profile-plot {{
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            padding: 5px;
            background: white;
        }}

        .profile-plot h4 {{
            color: #1976D2;
            font-size: 10px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 3px;
        }}

        .profile-placeholder {{
            background: linear-gradient(180deg, #FAFAFA 0%, #F0F0F0 100%);
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 3px;
            color: #888;
            font-style: italic;
            font-size: 9px;
            border: 1px dashed #CCC;
            text-align: center;
        }}

        /* Right column: Tractograms - SMALLER */
        .tractogram-column {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .tractogram-item {{
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            padding: 5px;
            background: white;
        }}

        .tractogram-item h4 {{
            color: #1976D2;
            font-size: 9px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 3px;
        }}

        .tractogram-placeholder {{
            background: #2a2a2a;
            height: 112px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 3px;
            color: #888;
            font-style: italic;
            font-size: 8px;
        }}

        .tractogram-legend {{
            display: flex;
            justify-content: center;
            gap: 12px;
            font-size: 8px;
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid #E0E0E0;
        }}

        .tractogram-legend span {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .legend-line-left {{
            width: 16px;
            height: 2px;
            background: #2196F3;
        }}

        .legend-line-right {{
            width: 16px;
            height: 2px;
            background: #F44336;
        }}

        /* Footer */
        .footer {{
            margin-top: 8px;
            padding-top: 6px;
            border-top: 1px solid #E0E0E0;
            text-align: center;
            color: #666;
            font-size: 8px;
            line-height: 1.4;
        }}
    </style>
</head>

<body>
    <!-- Header -->
    <div class="header">
        <h1>CST Analysis Report</h1>
        <div class="meta">
            Subject: <strong>{subject_id}</strong> |
            Date: <strong>{datetime.now().strftime("%Y-%m-%d")}</strong> |
            csttool <strong>v{version}</strong>
        </div>
        <div class="extraction-badge">Metrics Extracted In: {space}</div>
    </div>

    <!-- Acquisition & Processing Parameters -->
    <div class="params-row">
        <div class="param-box">
            <h3>Acquisition Parameters</h3>
            <div class="param-grid">
                <span class="param-label">Protocol:</span><span class="param-value">{acquisition.get('protocol', 'Single-shell DTI')}</span>
                <span class="param-label">b-value:</span><span class="param-value">{acquisition.get('b_value', '1000 s/mm²')}</span>
                <span class="param-label">Directions:</span><span class="param-value">{acquisition.get('directions', '64')}</span>
                <span class="param-label">Resolution:</span><span class="param-value">{acquisition.get('resolution', '2.0 × 2.0 × 2.0 mm³')}</span>
            </div>
        </div>
        <div class="param-box">
            <h3>Processing Parameters</h3>
            <div class="param-grid">
                <span class="param-label">Denoising:</span><span class="param-value">{processing.get('denoising', 'nlmeans')}</span>
                <span class="param-label">Tracking:</span><span class="param-value">{processing.get('tracking_method', 'Deterministic (DTI)')}</span>
                <span class="param-label">ROI Approach:</span><span class="param-value">{processing.get('roi_approach', 'Atlas-to-Subject (HO)')}</span>
                <span class="param-label">Whole-brain:</span><span class="param-value">{whole_brain}</span>
            </div>
        </div>
    </div>

    <!-- Summary Metrics -->
    <div class="section">
        <h2 class="section-title">Summary Metrics</h2>

        <table class="metrics-table">
            <thead>
                <tr>
                    <th style="width: 40%;">Metric</th>
                    <th style="width: 20%;">Left</th>
                    <th style="width: 20%;">Right</th>
                    <th style="width: 20%;">LI</th>
                </tr>
            </thead>
            <tbody>
                {"".join(metrics_rows)}
            </tbody>
        </table>

        <div class="legend">
            Legend: <span class="left-gt">Left&gt;Right</span>, <span class="right-gt">Right&gt;Left</span>
        </div>
    </div>

    <!-- Visualizations: Profiles (large) + Tractograms (small) -->
    <div class="section">
        <h2 class="section-title">Visualizations</h2>

        <div class="viz-section">
            <!-- Left: Along-Tract Profiles (LARGER) -->
            <div class="profiles-column">
                {profiles_html}
            </div>

            <!-- Right: Tractograms (SMALLER) -->
            <div class="tractogram-column">
                {tractograms_html}
            </div>
        </div>
    </div>

    <!-- Footer -->
    <div class="footer">
        Method: Deterministic tractography, DTI model | Generated by csttool
    </div>
</body>

</html>'''
    
    html_path = output_dir / f"{subject_id}_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML report saved: {html_path}")
    return html_path


def save_pdf_report(
    comparison,
    visualization_paths,
    output_dir,
    subject_id,
    version=None,
    space="Native Space"
):
    """
    Generate single-page clinical PDF report with metrics and visualizations.
    
    Layout:
    - Header: Subject ID, Date, version, space declaration
    - Compact metrics table with RD/AD and color-coded LI
    - Side-by-side visualizations (profiles + tractogram QC)
    - Footer: method note
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    visualization_paths : dict
        Paths to generated visualization figures (stacked_profiles, tractogram_qc_*)
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
    version : str
        csttool version string
    space : str
        Space declaration (e.g. "Native Space", "MNI152")
        
    Returns
    -------
    pdf_path : Path
        Path to saved PDF file
    """
    
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        print("⚠️  reportlab not installed. Skipping PDF generation.")
        print("   Install with: pip install reportlab")
        return None
    
    # Use package version if not specified
    if version is None:
        version = __version__
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f"{subject_id}_report.pdf"

    
    # Create document with reduced margins for single-page fit
    doc = SimpleDocTemplate(
        str(pdf_path), 
        pagesize=A4,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.4*inch,
        bottomMargin=0.4*inch
    )
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2196F3'),
        spaceAfter=6,
        alignment=TA_CENTER
    )
    
    header_style = ParagraphStyle(
        'HeaderInfo',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=2,
        alignment=TA_CENTER
    )
    
    space_style = ParagraphStyle(
        'SpaceDeclaration',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1976D2'),
        spaceAfter=8,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#1976D2'),
        spaceAfter=6,
        spaceBefore=8
    )
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=7,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    
    # === HEADER BLOCK ===
    story.append(Paragraph("CST Analysis Report", title_style))
    story.append(Paragraph(
        f"Subject: {subject_id} | Date: {datetime.now().strftime('%Y-%m-%d')} | csttool v{version}",
        header_style
    ))
    story.append(Paragraph(f"<b>Metrics Extracted In: {space}</b>", space_style))
    story.append(Spacer(1, 0.1*inch))
    
    # === METRICS TABLE - FIXED VERSION ===
    story.append(Paragraph("Summary Metrics", heading_style))
    
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    def format_li(li_value):
        """Format LI with color indicator."""
        if abs(li_value) < 0.05:
            return f"{li_value:+.3f}"
        elif li_value > 0:
            return f"<font color='blue'>{li_value:+.3f}</font>"
        else:
            return f"<font color='red'>{li_value:+.3f}</font>"
    
    # Build table data - FIX COLUMN HEADERS
    table_data = [
        ['Metric', 'Left', 'Right', 'LI'],  # Simple strings, not Paragraphs for headers
    ]
    
    # Add rows - ensure proper formatting
    header = Paragraph("Streamlines", styles['Normal']) 
    table_data.append([
        header,
        str(left['morphology']['n_streamlines']),
        str(right['morphology']['n_streamlines']),
        Paragraph(format_li(asym['streamline_count']['laterality_index']), styles['Normal'])
    ])
    
    # Convert volume from mm³ to cm³ correctly
    left_volume_cm3 = left['morphology']['tract_volume'] / 1000.0
    right_volume_cm3 = right['morphology']['tract_volume'] / 1000.0
    
    header = Paragraph("Volume (cm³)", styles['Normal'])
    table_data.append([
        header,
        f"{left_volume_cm3:.2f}",
        f"{right_volume_cm3:.2f}",
        Paragraph(format_li(asym['volume']['laterality_index']), styles['Normal'])
    ])
    
    header = Paragraph("Length (mm)", styles['Normal'])
    table_data.append([
        header,
        f"{left['morphology']['mean_length']:.1f}",
        f"{right['morphology']['mean_length']:.1f}",
        Paragraph(format_li(asym['mean_length']['laterality_index']), styles['Normal'])
    ])
    
    # Add diffusion metrics if available
    if 'fa' in left:
        header = Paragraph("FA", styles['Normal'])
        table_data.append([
            header,
            f"{left['fa']['mean']:.3f}",
            f"{right['fa']['mean']:.3f}",
            Paragraph(format_li(asym['fa']['laterality_index']), styles['Normal'])
        ])
    
    if 'md' in left:
        # MD is typically in ×10⁻³ mm²/s
        header = Paragraph("MD (×10<sup>-3</sup> mm<sup>2</sup>/s)", styles['Normal'])
        table_data.append([
            header,
            f"{left['md']['mean']*1000:.2f}",
            f"{right['md']['mean']*1000:.2f}",
            Paragraph(format_li(asym['md']['laterality_index']), styles['Normal'])
        ])
    
    if 'rd' in left:
        header = Paragraph("RD (×10<sup>-3</sup> mm<sup>2</sup>/s)", styles['Normal'])
        table_data.append([
            header,
            f"{left['rd']['mean']*1000:.2f}",
            f"{right['rd']['mean']*1000:.2f}",
            Paragraph(format_li(asym['rd']['laterality_index']), styles['Normal'])
        ])
    
    if 'ad' in left:
        header = Paragraph("AD (×10<sup>-3</sup> mm<sup>2</sup>/s)", styles['Normal'])
        table_data.append([
            header,
            f"{left['ad']['mean']*1000:.2f}",
            f"{right['ad']['mean']*1000:.2f}",
            Paragraph(format_li(asym['ad']['laterality_index']), styles['Normal'])
        ])
    
    # Legend row - FIX: Use proper string formatting
    table_data.append([
        Paragraph("Legend: <font color='blue'>Left>Right</font>, <font color='red'>Right>Left</font>", 
                 ParagraphStyle('Legend', parent=styles['Normal'], fontSize=8)),
        '', '', ''
    ])
    
    # Create table with adjusted column widths
    col_widths = [1.8*inch, 1.2*inch, 1.2*inch, 0.8*inch]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Apply table style
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -2), 9),  # All data rows except legend
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -2), 0.5, colors.grey),
        ('SPAN', (0, -1), (-1, -1)),  # Legend spans all columns
        ('ALIGN', (0, -1), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#F5F5F5')),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.12*inch))
    
        # === VISUALIZATION ROW - FIXED VERSION ===
    story.append(Paragraph("Visualizations", heading_style))
    
    # Create a two-column layout
    viz_data = []
    
    # Left column: Stacked profiles
    left_column_content = []
    if visualization_paths and 'stacked_profiles' in visualization_paths:
        profile_path = visualization_paths['stacked_profiles']
        if profile_path and Path(profile_path).exists():
            try:
                # Scale image to fit within column - reduced height for single-page fit
                profile_img = Image(str(profile_path), width=3.8*inch, height=4.5*inch)
                left_column_content.append(profile_img)
            except Exception as e:
                print(f"⚠️  Could not add profile image: {e}")
                left_column_content.append(Paragraph("Profile plot unavailable", styles['Normal']))
        else:
            left_column_content.append(Paragraph("Profile plot not found", styles['Normal']))
    else:
        left_column_content.append(Paragraph("Profile plot unavailable", styles['Normal']))
    
    # Right column: QC images stacked vertically
    right_column_content = []
    qc_views = ['axial', 'sagittal', 'coronal']
    qc_images_added = 0
    
    for view in qc_views:
        key = f'tractogram_qc_{view}'
        if visualization_paths and key in visualization_paths:
            img_path = visualization_paths[key]
            if img_path and Path(img_path).exists():
                try:
                    # Scale QC images smaller for single-page fit
                    img = Image(str(img_path), width=2.5*inch, height=1.5*inch)
                    right_column_content.append(img)
                    # Only add spacing between images, not after the last one
                    if view != 'coronal':
                        right_column_content.append(Spacer(1, 0.05*inch))
                    qc_images_added += 1
                except Exception as e:
                    print(f"⚠️  Could not add {view} QC image: {e}")
    
    if qc_images_added == 0:
        right_column_content.append(Paragraph("Tractogram visualizations unavailable", styles['Normal']))
    
    # Create two-column table
    viz_table = Table([
        [left_column_content, right_column_content]
    ], colWidths=[4.2*inch, 3.3*inch])
    
    viz_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    story.append(viz_table)
    story.append(Spacer(1, 0.1*inch))
    
    # === FOOTER ===
    story.append(Paragraph(
        "Method: Deterministic tractography, DTI model | Generated by csttool",
        footer_style
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF report saved: {pdf_path}")
    return pdf_path

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
            slice_type=view
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
    report_paths = {
        'json': save_json_report(comparison, output_dir, subject_id),
        'csv': save_csv_summary(comparison, output_dir, subject_id),
        'pdf': save_pdf_report(comparison, pdf_viz_paths, output_dir, subject_id, version, space),
        'visualizations': viz_paths
    }
    
    print("\n✅ Complete report package generated!")
    return report_paths