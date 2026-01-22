"""
Validate CLI Command

Compare csttool-extracted CST tractograms against reference bundles.
"""

import argparse
from pathlib import Path

from csttool.validation import (
    compute_bundle_overlap,
    compute_overreach,
    mean_closest_distance,
    generate_validation_report,
)


def add_validate_parser(subparsers):
    """Add the validate subcommand parser."""
    parser = subparsers.add_parser(
        "validate",
        help="Compare extracted CST against reference tractograms",
        description=(
            "Validate csttool-extracted CST streamlines against reference "
            "tractograms (e.g., TractoInferno PYT bundles) using overlap "
            "and distance metrics."
        ),
    )
    
    parser.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        type=Path,
        help="Path(s) to candidate CST tractogram(s) (.trk)",
    )
    
    parser.add_argument(
        "--reference",
        nargs="+",
        required=True,
        type=Path,
        help="Path(s) to reference tractogram(s) (.trk)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "validation_output",
        help="Output directory for validation report (default: ./validation_output)",
    )
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["dice", "overreach", "mdf", "all"],
        default=["all"],
        help="Metrics to compute (default: all)",
    )
    
    parser.set_defaults(func=cmd_validate)
    
    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    print("\n=== CST Validation ===\n")
    
    candidate_paths = args.candidate
    reference_paths = args.reference
    output_dir = args.output_dir
    metrics_to_compute = args.metrics
    
    # Validate inputs
    for path in candidate_paths:
        if not path.exists():
            print(f"ERROR: Candidate tractogram not found: {path}")
            return 1
    
    for path in reference_paths:
        if not path.exists():
            print(f"ERROR: Reference tractogram not found: {path}")
            return 1
    
    if len(candidate_paths) != len(reference_paths):
        print(
            f"WARNING: Number of candidate ({len(candidate_paths)}) and "
            f"reference ({len(reference_paths)}) tractograms differ. "
            "Pairing in order."
        )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which metrics to compute
    if "all" in metrics_to_compute:
        compute_dice = True
        compute_or = True
        compute_mdf = True
    else:
        compute_dice = "dice" in metrics_to_compute
        compute_or = "overreach" in metrics_to_compute
        compute_mdf = "mdf" in metrics_to_compute
    
    # Process each pair
    all_metrics = {}
    pairs = list(zip(candidate_paths, reference_paths))
    
    for i, (cand_path, ref_path) in enumerate(pairs):
        pair_name = f"pair_{i+1}"
        print(f"Processing {pair_name}:")
        print(f"  Candidate: {cand_path.name}")
        print(f"  Reference: {ref_path.name}")
        
        pair_metrics = {}
        
        if compute_dice:
            print("  Computing Dice overlap...", end=" ")
            overlap = compute_bundle_overlap(cand_path, ref_path)
            pair_metrics["overlap"] = overlap
            print(f"Dice = {overlap['dice']:.4f}")
        
        if compute_or:
            print("  Computing overreach...", end=" ")
            overreach = compute_overreach(cand_path, ref_path)
            pair_metrics["overreach"] = overreach
            print(f"Overreach = {overreach['overreach']:.4f}")
        
        if compute_mdf:
            print("  Computing MDF...", end=" ")
            mdf = mean_closest_distance(cand_path, ref_path)
            pair_metrics["mdf"] = mdf
            print(f"MDF = {mdf['mdf_mean']:.2f} mm (±{mdf['mdf_std']:.2f})")
        
        all_metrics[pair_name] = {
            "candidate": str(cand_path),
            "reference": str(ref_path),
            "metrics": pair_metrics,
        }
        print()
    
    # Generate report
    report_path = output_dir / "validation_report.json"
    generate_validation_report(
        all_metrics,
        report_path,
        candidate_paths=candidate_paths,
        reference_paths=reference_paths,
    )
    
    print(f"Validation report saved to: {report_path}")
    print("\n=== Validation Complete ===\n")
    
    return 0
