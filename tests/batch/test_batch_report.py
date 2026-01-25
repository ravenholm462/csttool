import csv
import json
from pathlib import Path
import pytest
from csttool.batch.modules.report import write_batch_metrics_csv, METRICS_HEADERS
from csttool.batch.batch import SubjectResult

def test_write_batch_metrics_csv_comprehensive(tmp_path):
    """
    Test that batch metrics CSV includes all scalars and localized metrics
    as per the unified reporting requirement.
    """
    output_dir = tmp_path / "batch_out"
    output_dir.mkdir()
    
    subject_id = "sub-01"
    session_id = None
    
    # 1. Setup mock subject output structure
    # Structure: <out>/<sub_id>/metrics/<sub_id>_bilateral_metrics.json
    subj_dir = output_dir / subject_id
    metrics_dir = subj_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    
    # Create a comprehensive metrics JSON mock
    mock_metrics = {
        "metrics": {
            "left": {
                "morphology": {
                    "n_streamlines": 100,
                    "mean_length": 120.5,
                    "tract_volume": 5000.0
                },
                "fa": {
                    "mean": 0.65, "std": 0.1,
                    "pontine": 0.7, "plic": 0.6, "precentral": 0.5
                },
                "md": {
                    "mean": 0.0008, "std": 0.0001,
                    "pontine": 0.0009, "plic": 0.0008, "precentral": 0.0007
                },
                # rd, ad omitted to test partial data handling
            },
            "right": {
                "morphology": {
                    "n_streamlines": 110,
                    "mean_length": 122.0,
                    "tract_volume": 5100.0
                },
                "fa": {
                    "mean": 0.66, "std": 0.11,
                    "pontine": 0.71, "plic": 0.61, "precentral": 0.51
                },
                "md": {
                    "mean": 0.00081, "std": 0.00011,
                    "pontine": 0.00091, "plic": 0.00081, "precentral": 0.00071
                }
            },
            "asymmetry": {
                "volume": {"laterality_index": 0.02},
                "streamline_count": {"laterality_index": 0.09},
                "fa": {"laterality_index": 0.015},
                "md": {"laterality_index": 0.001},
                # Localized LIs
                "fa_pontine": {"laterality_index": 0.01},
                "fa_plic": {"laterality_index": 0.02},
                "fa_precentral": {"laterality_index": 0.03},
            }
        }
    }
    
    json_path = metrics_dir / f"{subject_id}_bilateral_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(mock_metrics, f)
        
    # 2. logical results
    results = [
        SubjectResult(
            subject_id=subject_id,
            session_id=session_id,
            status="success",
            duration_seconds=10.0
        ),
        SubjectResult(
            subject_id="sub-02", # Failed subject
            session_id=None,
            status="failed",
            error="Something broke"
        )
    ]
    
    # 3. Generate CSV
    write_batch_metrics_csv(results, output_dir)
    
    # 4. Verify
    csv_path = output_dir / "batch_metrics.csv"
    assert csv_path.exists()
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 2
    
    # Check Header consistency
    assert reader.fieldnames == METRICS_HEADERS
    
    # Validate sub-01 data
    r1 = rows[0]
    assert r1['subject_id'] == "sub-01"
    assert r1['left_n_streamlines'] == "100"
    assert r1['right_n_streamlines'] == "110"
    
    # Check Morphology units
    # Volume is source (mm3) / 1000 -> not done in batch report anymore?
    # Wait, my implementation in report.py reused the logic from single report?
    # No, existing code had /1000.0 under "cst_l_volume_cm3".
    # Implementation Plan says "matches single report units" which is mm3.
    # But wait, looking at my `report.py` modification:
    # 'left_tract_volume_mm3': left['morphology']['tract_volume']
    # If the source is mm3, this remains mm3.
    # The single report CSV uses mm3. HTML uses cm3.
    # I should verify if I kept it as mm3 in implementation.
    # Yes: 'left_tract_volume_mm3': left['morphology']['tract_volume']
    
    assert float(r1['left_tract_volume_mm3']) == 5000.0  # mm3
    
    # Check Scalars
    assert float(r1['left_fa_mean']) == 0.65
    assert float(r1['fa_laterality_index']) == 0.015
    
    # Check Localized
    assert float(r1['left_fa_pontine']) == 0.7
    assert float(r1['fa_pontine_laterality_index']) == 0.01
    
    # Check MD (Diffusivity)
    # Single report CSV uses RAW values.
    # My implementation: left[scalar].get(region) -> RAW
    assert float(r1['left_md_mean']) == 0.0008
    
    # Validate sub-02 data (failed)
    r2 = rows[1]
    assert r2['subject_id'] == "sub-02"
    assert r2['status'] == "failed"
    assert r2['left_n_streamlines'] == ""  # Should be empty
