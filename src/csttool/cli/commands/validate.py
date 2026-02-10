"""
Validate CLI Command

Compare csttool-extracted CST tractograms against reference bundles.
"""

import argparse
import sys
from pathlib import Path

from csttool.validation import (
    compute_bundle_overlap,
    compute_overreach,
    compute_coverage,
    mean_closest_distance,
    streamline_count_ratio,
    generate_validation_report,
    check_spatial_compatibility,
    check_hemisphere_alignment,
    SpatialMismatchError,
)

try:
    from csttool.validation.visualization import save_overlap_maps, save_validation_snapshots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


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
    
    # Candidate Inputs
    parser.add_argument(
        "--cand-left",
        required=True,
        type=Path,
        help="Path to candidate Left CST tractogram (.trk)",
    )
    parser.add_argument(
        "--cand-right",
        required=True,
        type=Path,
        help="Path to candidate Right CST tractogram (.trk)",
    )
    
    # Reference Inputs
    parser.add_argument(
        "--ref-left",
        required=True,
        type=Path,
        help="Path to reference Left CST tractogram (.trk)",
    )
    parser.add_argument(
        "--ref-right",
        required=True,
        type=Path,
        help="Path to reference Right CST tractogram (.trk)",
    )
    parser.add_argument(
        "--ref-space",
        required=True,
        type=Path,
        help="Reference NIfTI image defining the coordinate grid (e.g. FA map)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "validation_output",
        help="Output directory for validation report (default: ./validation_output)",
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visual reports (overlays and snapshots)",
    )
    
    parser.add_argument(
        "--disable-hemisphere-check",
        action="store_true",
        help="Disable warning for bundle centroid hemisphere mismatch",
    )
    
    parser.set_defaults(func=cmd_validate)
    
    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    print("=" * 60)
    print("CST VALIDATION")
    print("=" * 60)
    
    output_dir = args.output_dir
    ref_space = args.ref_space
    visualize = args.visualize
    
    # Pairs to process
    pairs = [
        ("Left", args.cand_left, args.ref_left),
        ("Right", args.cand_right, args.ref_right),
    ]
    
    # 1. Validation & Setup
    if not ref_space.exists():
        print(f"  ✗ Reference space image not found: {ref_space}")
        return 1

    for side, cand, ref in pairs:
        if not cand.exists():
            print(f"  ✗ {side} candidate not found: {cand}")
            return 1
        if not ref.exists():
            print(f"  ✗ {side} reference not found: {ref}")
            return 1

    if visualize and not VISUALIZATION_AVAILABLE:
        print("  ⚠️ Visualization libraries (nilearn/matplotlib) not found, skipping")
        visualize = False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    
    # 2. Process each hemisphere
    for side, cand_path, ref_path in pairs:
        print(f"\n  → Processing {side} hemisphere...")
        print(f"  Candidate: {cand_path.name}")
        print(f"  Reference: {ref_path.name}")

        # A. Check Spatial Compatibility
        try:
            check_spatial_compatibility(
                cand_path, ref_path, ref_space,
                tol_trans=1.0, tol_rot=1e-3
            )
            print(f"  ✓ Spatial compatibility verified")
        except SpatialMismatchError as e:
            print(f"  ✗ Spatial mismatch detected: {e}")
            return 1
        except Exception as e:
            print(f"  ✗ Error checking space: {e}")
            return 1

        # A.5 Hemisphere Check
        if not args.disable_hemisphere_check:
            try:
                hs_warn = check_hemisphere_alignment(cand_path, ref_path)
                if hs_warn:
                    print(f"  ⚠️ {hs_warn}")
            except Exception as e:
                if visualize:
                    print(f"  ⚠️ Hemisphere check failed: {e}")

        # B. Compute Metrics
        side_metrics = {}

        # Overlap (Dice)
        d_res = compute_bundle_overlap(cand_path, ref_path, ref_space)
        side_metrics.update(d_res)

        # Coverage
        cov_res = compute_coverage(cand_path, ref_path, ref_space)
        side_metrics.update(cov_res)

        # Overreach
        ov_res = compute_overreach(cand_path, ref_path, ref_space)
        side_metrics.update(ov_res)

        # MDF
        mdf_res = mean_closest_distance(cand_path, ref_path, step_size_mm=2.0)
        side_metrics.update(mdf_res)

        # Count Ratio
        cnt_res = streamline_count_ratio(cand_path, ref_path)
        side_metrics.update(cnt_res)

        print(f"  Dice:          {d_res['dice']:.4f}")
        print(f"  Coverage:      {cov_res['coverage']:.4f}")
        print(f"  Overreach:     {ov_res['overreach']:.4f}")
        print(f"  MDF symmetric: {mdf_res['mdf_symmetric']:.2f} mm")
        print(f"  Count ratio:   {cnt_res['streamline_count_ratio']:.2f}")

        all_metrics[side.lower()] = side_metrics

        # C. Visualization
        if visualize:
            try:
                vis_dir = output_dir / "visualizations" / side.lower()
                maps = save_overlap_maps(cand_path, ref_path, ref_space, vis_dir)
                save_validation_snapshots(ref_space, maps, vis_dir)
                print(f"  ✓ Visualizations saved")
            except Exception as e:
                print(f"  ⚠️ Visualization failed: {e}")

    # 3. Generate Report
    report_path = output_dir / "validation_report.json"
    generate_validation_report(
        metrics=all_metrics,
        output_path=report_path,
        candidate_paths=[p[1] for p in pairs],
        reference_paths=[p[2] for p in pairs],
        ref_anatomy_path=ref_space
    )
    
    print(f"\n✓ Validation complete")
    print(f"  Report: {report_path}")
    
    return 0
