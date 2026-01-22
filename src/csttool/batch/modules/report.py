import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Standard headers for batch_metrics.csv
METRICS_HEADERS = [
    "subject_id", "session_id", "status", "error_category", 
    "duration_seconds", "cst_l_streamline_count", "cst_r_streamline_count",
    "cst_l_volume_cm3", "cst_r_volume_cm3",
    "cst_l_mean_fa", "cst_r_mean_fa", "laterality_index", "error"
]

logger = logging.getLogger(__name__)

def generate_batch_reports(results: List[Any], output_dir: Path):
    """
    Orchestrates the generation of all batch-level reports.
    
    Args:
        results: List of SubjectResult objects
        output_dir: Path to the batch output root
    """
    write_batch_metrics_csv(results, output_dir)
    write_batch_summary_json(results, output_dir)

def write_batch_metrics_csv(results: List[Any], output_dir: Path):
    """
    Writes a CSV aggregate of all subject metrics.
    Numeric cells are empty (NaN) for failed or skipped subjects.
    """
    csv_path = output_dir / "batch_metrics.csv"
    
    rows = []
    for res in results:
        # Standard fields always populated
        row = {
            "subject_id": res.subject_id,
            "session_id": res.session_id or "",
            "status": res.status,
            "error_category": res.error_category or "",
            "duration_seconds": round(res.duration_seconds, 1),
            "error": res.error or ""
        }
        
        # If success, attempt to extract numeric metrics from the per-subject JSON
        if res.status == "success":
            try:
                # Construct path to the subject's JSON report
                # Final output structure: <out>/<sub_id>/<ses_id>/metrics/<sub_id>_<ses_id>_bilateral_metrics.json
                subj_path = output_dir / res.subject_id
                if res.session_id:
                    subj_path = subj_path / res.session_id
                
                # csttool metrics named the file {subject_id}_bilateral_metrics.json inside "metrics" subdirectory
                metrics_file = subj_path / "metrics" / f"{res.subject_id}_bilateral_metrics.json"
                
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        m = data.get("metrics", {})
                        
                        # Populate metrics with formatting
                        if 'left' in m and 'right' in m:
                            row.update({
                                "cst_l_streamline_count": m['left']['morphology']['n_streamlines'],
                                "cst_r_streamline_count": m['right']['morphology']['n_streamlines'],
                                "cst_l_volume_cm3": round(m['left']['morphology']['tract_volume'] / 1000.0, 3),
                                "cst_r_volume_cm3": round(m['right']['morphology']['tract_volume'] / 1000.0, 3),
                            })
                            
                            # Functional metrics (FA)
                            if 'fa' in m['left']:
                                row.update({
                                    "cst_l_mean_fa": round(m['left']['fa']['mean'], 3),
                                    "cst_r_mean_fa": round(m['right']['fa']['mean'], 3),
                                    "laterality_index": round(m['asymmetry'].get('fa', {}).get('laterality_index', 0), 3)
                                })
            except Exception as e:
                logger.warning(f"Could not extract metrics for {res.subject_id}: {e}")
                
        rows.append(row)
        
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_HEADERS, restval="")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"✓ Batch metrics CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to write batch_metrics.csv: {e}")

def write_batch_summary_json(results: List[Any], output_dir: Path):
    """
    Writes a versioned JSON summary of the entire batch run.
    Contains summary counters and individual status records.
    """
    summary_path = output_dir / "batch_summary.json"
    
    counts = {
        "total": len(results),
        "success": sum(1 for r in results if r.status == "success"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped")
    }
    
    summary = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "summary": counts,
        "results": [
            {
                "subject_id": r.subject_id,
                "session_id": r.session_id,
                "status": r.status,
                "error_category": r.error_category,
                "duration_seconds": round(r.duration_seconds, 1),
                "error": r.error
            } for r in results
        ]
    }
    
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Batch summary JSON: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to write batch_summary.json: {e}")
