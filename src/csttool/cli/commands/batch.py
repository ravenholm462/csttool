import argparse
import logging
import sys
from pathlib import Path
from typing import List

from ...batch.batch import BatchConfig, run_batch, SubjectSpec
from ...batch.modules.manifest import load_manifest, StudyInfo
from ...batch.modules.validation import validate_batch_preflight, PreflightError
from ...batch.modules.report import generate_batch_reports
from ...batch.modules.discover import (
    discover_subjects, 
    detect_input_type, 
    find_bval_bvec, 
    find_json_sidecar,
    validate_single_series
)

logger = logging.getLogger(__name__)

def cmd_batch(args: argparse.Namespace) -> None:
    """
    CLI handler for the 'batch' command.
    """
    verbose = getattr(args, 'verbose', False)
    if verbose:
        logging.getLogger("csttool").setLevel(logging.INFO)
    
    # 1. Prepare Subjects List
    subjects: List[SubjectSpec] = []
    global_options = {}
    
    if getattr(args, 'manifest', None):
        # Load from manifest file
        logger.info(f"Loading manifest: {args.manifest}")
        study, subj_configs = load_manifest(args.manifest)
        global_options = study.global_options
        
        for s in subj_configs:
            # Determine input type automatically if not specified (though manifest usually has it)
            series_uid = None
            if s.get('nifti'):
                input_path = Path(s['nifti'])
                input_type = "nifti"
                bval, bvec = find_bval_bvec(input_path)
                json_sidecar = find_json_sidecar(input_path)
            else:
                input_path = Path(s['dicom'])
                input_type = "dicom"
                bval, bvec, json_sidecar = None, None, None
                
                # Enforce series disambiguation
                series_uid = s.get('series_uid')
                try:
                    # If uid provided, validate existence. If not, validate uniqueness.
                    # Returns the valid UID.
                    series_uid = validate_single_series(input_path, series_uid)
                except Exception as e:
                    logger.error(f"Subject {s['id']} DICOM validation failed: {e}")
                    raise 

            subjects.append(SubjectSpec(
                subject_id=s['id'],
                session_id=s.get('session'),
                input_path=input_path,
                input_type=input_type,
                bval_path=bval,
                bvec_path=bvec,
                json_path=json_sidecar,
                series_uid=series_uid,
                options=s.get('options', {})
            ))
    elif getattr(args, 'bids_dir', None):
        # Auto-discover from BIDS directory
        logger.info(f"Auto-discovering subjects in: {args.bids_dir}")
        discovered = discover_subjects(
            args.bids_dir,
            include_subjects=getattr(args, 'include', None),
            exclude_subjects=getattr(args, 'exclude', None)
        )
        
        for item in discovered:
            try:
                input_type, input_path = detect_input_type(item['dir'])
                bval, bvec = None, None
                json_sidecar = None
                series_uid = None
                
                if input_type == "nifti":
                    bval, bvec = find_bval_bvec(input_path)
                    json_sidecar = find_json_sidecar(input_path)
                elif input_type == "dicom":
                    # Check for ambiguity
                    try:
                        # Returns UID if unique or specified (here None), raises error if ambiguous
                        series_uid = validate_single_series(input_path) 
                    except Exception as e:
                        logger.warning(f"Skipping {item['id']} (DICOM ambiguity): {e}")
                        continue
                
                subjects.append(SubjectSpec(
                    subject_id=item['id'],
                    session_id=item['session'],
                    input_path=input_path,
                    input_type=input_type,
                    bval_path=bval,
                    bvec_path=bvec,
                    json_path=json_sidecar,
                    series_uid=series_uid,
                    options={}
                ))
            except Exception as e:
                logger.warning(f"Skipping {item['id']}: {e}")
    else:
        print("Error: Must provide either --manifest or --bids-dir")
        sys.exit(1)

    if not subjects:
        print("Error: No subjects found to process.")
        sys.exit(1)

    # 2. Setup Batch Configuration
    config = BatchConfig(
        out=args.out,
        force=args.force,
        timeout_minutes=args.timeout_minutes,
        keep_work=args.keep_work,
        # Pipeline options (inherit from global options then CLI)
        denoise_method=getattr(args, 'denoise_method', global_options.get('denoise_method', 'patch2self')),
        preprocessing=getattr(args, 'preprocessing', global_options.get('preprocessing', True)),
        generate_pdf=getattr(args, 'generate_pdf', global_options.get('generate_pdf', False)),
        # Additional options can be added here
    )

    # 3. Preflight Validation
    logger.info("Running preflight validation...")
    errors = validate_batch_preflight(subjects, config)
    
    if errors:
        print("\nBatch Preflight Validation Failed:")
        print("="*40)
        for err in errors:
            sub_id = f"[{err.subject_id}] " if err.subject_id else ""
            print(f"- {err.category}: {sub_id}{err.message}")
        print("="*40)
        
        if not getattr(args, 'dry_run', False):
            sys.exit(1)

    if getattr(args, 'validate_only', False):
        print("\nValidation successful.")
        return

    # 4. Dry Run Logic
    if getattr(args, 'dry_run', False):
        print("\nBatch Execution Plan:")
        print("="*40)
        print(f"Total subjects: {len(subjects)}")
        print(f"Output root:    {config.out}")
        print(f"Config hash:    (will be computed at runtime)")
        print("\nSubjects to process:")
        for i, s in enumerate(subjects[:10], 1):
            ses = f", session {s.session_id}" if s.session_id else ""
            print(f"  {i}. {s.subject_id}{ses} ({s.input_type})")
        if len(subjects) > 10:
            print(f"  ... and {len(subjects)-10} more")
        print("="*40)
        return

    # 5. Execute Batch
    print(f"\nStarting batch processing for {len(subjects)} subjects...")
    results = run_batch(subjects, config, verbose=verbose)

    # 6. Generate Reports
    print("\nGenerating final batch reports...")
    generate_batch_reports(results, config.out)
    
    # 7. Summary
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    
    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Total:      {len(results)}")
    print(f"Success:    {success}")
    print(f"Failed:     {failed}")
    print(f"Skipped:    {skipped}")
    print(f"\nMain metrics: {config.out / 'batch_metrics.csv'}")
    print("="*40)
