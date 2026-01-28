
import argparse
import json
from pathlib import Path
from datetime import datetime
from time import time

from csttool import __version__
from ..utils import extract_stem_from_filename, save_pipeline_report

from .check import cmd_check
from .import_cmd import cmd_import
from .preprocess import cmd_preprocess
from .track import cmd_track
from .extract import cmd_extract, run_roi_seeded_extraction
from .metrics import cmd_metrics

def cmd_run(args: argparse.Namespace) -> None:
    """
    Run complete CST analysis pipeline.
    
    Steps:
        1. check     - Verify environment and dependencies
        2. import    - Convert DICOM to NIfTI (or validate existing NIfTI)
        3. preprocess - Denoise, skull strip, (optional) motion correct
        4. track     - Whole-brain deterministic tractography
        5. extract   - Atlas-based bilateral CST extraction
        6. metrics   - Compute metrics and generate reports
    """
    
    verbose = getattr(args, 'verbose', False)
    quiet = getattr(args, 'quiet', False)
    continue_on_error = getattr(args, 'continue_on_error', False)

    # Quiet overrides verbose
    if quiet:
        verbose = False

    # Create output directory structure
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Determine subject ID
    subject_id = args.subject_id
    if not subject_id:
        if args.dicom:
            subject_id = args.dicom.name
        elif args.nifti:
            subject_id = extract_stem_from_filename(str(args.nifti))
        else:
            subject_id = "subject"
    
    # Initialize pipeline tracking
    pipeline_start = time()
    step_times = {}
    step_results = {}
    failed_steps = []
    pipeline_metadata = {}  # Initialize here to ensure it's available throughout
    
    if not quiet:
        print("\n" + "="*70)
        print("CSTTOOL - COMPLETE CST ANALYSIS PIPELINE")
        print("="*70)
        print(f"Subject ID:     {subject_id}")
        print(f"Output:         {args.out}")
        print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    # =========================================================================
    # STEP 1: CHECK
    # =========================================================================
    if not getattr(args, 'skip_check', False):
        if not quiet:
            print("\n" + "▶"*3 + " STEP 1/6: ENVIRONMENT CHECK " + "◀"*3)
        t0 = time()
        
        try:
            # Create a mock args object for cmd_check
            check_args = argparse.Namespace()
            check_ok = cmd_check(check_args)
            step_results['check'] = {'success': check_ok}
            
            if not check_ok and not continue_on_error:
                print("\n✗ Environment check failed. Fix dependencies and retry.")
                return
                
        except Exception as e:
            print(f"✗ Check failed: {e}")
            failed_steps.append('check')
            step_results['check'] = {'success': False, 'error': str(e)}
            if not continue_on_error:
                return
        
        step_times['check'] = time() - t0
    else:
        if not quiet:
            print("\n" + "⏭ STEP 1/6: SKIPPING ENVIRONMENT CHECK")
        step_results['check'] = {'success': True, 'skipped': True}
    
    # =========================================================================
    # STEP 2: IMPORT
    # =========================================================================
    if not quiet:
        print("\n" + "▶"*3 + " STEP 2/6: IMPORT DATA " + "◀"*3)
    t0 = time()
    
    nifti_path = None
    
    try:
        if args.nifti:
            # Check if file exists (handles broken symlinks too)
            if not args.nifti.exists():
                # Check if it's a broken symlink (e.g., git-annex)
                if args.nifti.is_symlink():
                    raise FileNotFoundError(
                        f"NIfTI file is a broken symlink: {args.nifti}\n"
                        f"If using git-annex, run: git annex get {args.nifti}"
                    )
                else:
                    raise FileNotFoundError(f"NIfTI file does not exist: {args.nifti}")
            
            # Use provided NIfTI directly
            nifti_path = args.nifti
            print(f"Using existing NIfTI: {nifti_path}")
            step_results['import'] = {'success': True, 'nifti_path': str(nifti_path)}
            
            # Extract acquisition metadata from NIfTI
            try:
                from ..utils import load_with_preproc
                from csttool.ingest import extract_acquisition_metadata
                
                _, _, hdr, gtab, bids_json = load_with_preproc(nifti_path)
                voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
                
                # Build CLI overrides
                overrides = {}
                if getattr(args, 'field_strength', None):
                    overrides['field_strength_T'] = args.field_strength
                if getattr(args, 'echo_time', None):
                    overrides['echo_time_ms'] = args.echo_time
                
                acquisition = extract_acquisition_metadata(
                    bvecs=gtab.bvecs,
                    bvals=gtab.bvals,
                    voxel_size=voxel_size,
                    bids_json=bids_json,
                    overrides=overrides
                )
                
                pipeline_metadata['acquisition'] = acquisition
                
            except Exception as e:
                if verbose:
                    print(f"Note: Could not extract acquisition metadata: {e}")
                    
        elif args.dicom:
            # Run import from DICOM
            import_args = argparse.Namespace(
                dicom=args.dicom,
                nifti=None,
                out=args.out,
                subject_id=subject_id,
                series=getattr(args, 'series', None),
                scan_only=False,
                verbose=verbose,
                field_strength=getattr(args, 'field_strength', None),
                echo_time=getattr(args, 'echo_time', None)
            )
            
            import_result = cmd_import(import_args)
            
            if import_result and import_result.get('nifti_path'):
                nifti_path = Path(import_result['nifti_path'])
                step_results['import'] = {'success': True, 'result': import_result}
                
                # capture metadata from import
                if 'metadata' in import_result:
                     pipeline_metadata['acquisition'] = import_result['metadata']
            else:
                raise RuntimeError("Import failed to produce NIfTI file")
        else:
            raise ValueError("Must provide --dicom or --nifti")
            
    except Exception as e:
        print(f"✗ Import failed: {e}")
        failed_steps.append('import')
        step_results['import'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['import'] = time() - t0
    
    # =========================================================================
    # STEP 3: PREPROCESS
    # =========================================================================
    
    if getattr(args, 'preprocess', False):
        if not quiet:
            print("\n" + "▶"*3 + " STEP 3/6: PREPROCESSING " + "◀"*3)
        t0 = time()
        
        preproc_path = None
        
        try:
            if nifti_path is None:
                raise RuntimeError("No NIfTI available from import step")
            
            preproc_out = args.out / "preprocessing"
            preproc_args = argparse.Namespace(
                dicom=None,
                nifti=nifti_path,
                out=preproc_out,
                coil_count=getattr(args, 'coil_count', 4),
                denoise_method=getattr(args, 'denoise_method', 'patch2self'),
                show_plots=getattr(args, 'show_plots', False),
                save_visualizations=getattr(args, 'save_visualizations', False),
                unring=getattr(args, 'unring', False),
                perform_motion_correction=getattr(args, 'perform_motion_correction', False),
                target_voxel_size=getattr(args, 'target_voxel_size', None),
                verbose=verbose
            )
            
            preproc_result = cmd_preprocess(preproc_args)
            
            if preproc_result and preproc_result.get('preprocessed_path'):
                preproc_path = Path(preproc_result['preprocessed_path'])
                step_results['preprocess'] = {'success': True, 'result': preproc_result}
            else:
                raise RuntimeError("Preprocessing failed")
                
        except Exception as e:
            print(f"✗ Preprocessing failed: {e}")
            failed_steps.append('preprocess')
            step_results['preprocess'] = {'success': False, 'error': str(e)}
            if not continue_on_error:
                save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
                return
        
        step_times['preprocess'] = time() - t0
        
        # Record metadata for report
        pipeline_metadata['preprocessing'] = {
            'status': 'Executed',
            'method': getattr(args, 'denoise_method', 'patch2self'),
            'unring': getattr(args, 'unring', False),
            'motion_correction': getattr(args, 'perform_motion_correction', False)
        }
        
    else:
        if not quiet:
            print("\n" + "⏭ STEP 3/6: SKIPPING PREPROCESSING (Pass-through)")
        t0 = time()
        # Pass through the input NIfTI as the "preprocessed" data
        preproc_path = nifti_path
        step_results['preprocess'] = {'success': True, 'skipped': True, 'passthrough_path': str(preproc_path)}
        step_times['preprocess'] = time() - t0
        
        # Record metadata for report
        pipeline_metadata['preprocessing'] = {
            'status': 'Skipped (External Preprocessing Used)'
        }
    
    # =========================================================================
    # STEP 4: TRACK
    # =========================================================================
    if not quiet:
        print("\n" + "▶"*3 + " STEP 4/6: TRACTOGRAPHY " + "◀"*3)
    t0 = time()
    
    tractogram_path = None
    fa_path = None
    md_path = None
    rd_path = None
    ad_path = None
    
    try:
        if preproc_path is None:
            raise RuntimeError("No preprocessed data available")
        
        track_out = args.out / "tracking"
        track_args = argparse.Namespace(
            nifti=preproc_path,
            subject_id=subject_id,
            fa_thr=getattr(args, 'fa_thr', 0.2),
            seed_density=getattr(args, 'seed_density', 1),
            step_size=getattr(args, 'step_size', 0.5),
            sh_order=getattr(args, 'sh_order', 6),
            save_visualizations=getattr(args, 'save_visualizations', False),
            show_plots=getattr(args, 'show_plots', False),
            verbose=verbose,
            out=track_out
        )
        
        track_result = cmd_track(track_args)
        
        if track_result:
            tractogram_path = Path(track_result['tractogram_path'])
            fa_path = Path(track_result['fa_path'])
            md_path = Path(track_result['md_path'])
            # RD and AD maps (if available)
            rd_path = Path(track_result['rd_path']) if 'rd_path' in track_result else None
            ad_path = Path(track_result['ad_path']) if 'ad_path' in track_result else None
            step_results['track'] = {'success': True, 'result': track_result}
            
            # Capture tracking parameters
            if 'tracking_params' in track_result:
                pipeline_metadata['tracking'] = track_result['tracking_params']
        else:
            raise RuntimeError("Tracking failed")
            
    except Exception as e:
        print(f"✗ Tracking failed: {e}")
        failed_steps.append('track')
        step_results['track'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['track'] = time() - t0
    
    # =========================================================================
    # STEP 5: EXTRACT
    # =========================================================================
    if not quiet:
        print("\n" + "▶"*3 + " STEP 5/6: CST EXTRACTION " + "◀"*3)
    t0 = time()
    
    cst_left_path = None
    cst_right_path = None
    
    extraction_method = getattr(args, 'extraction_method', 'passthrough')
    
    try:
        if tractogram_path is None or fa_path is None:
            raise RuntimeError("No tractogram or FA map available")
        
        extract_out = args.out / "extraction"
        
        if extraction_method == "roi-seeded":
            # ROI-seeded requires raw DWI data - use dedicated function
            extract_result = run_roi_seeded_extraction(
                preproc_path=preproc_path,
                fa_path=fa_path,
                output_dir=extract_out,
                subject_id=subject_id,
                args=args,
                verbose=verbose
            )
        else:
            # endpoint or passthrough - use cmd_extract with tractogram
            extract_args = argparse.Namespace(
                tractogram=tractogram_path,
                fa=fa_path,
                subject_id=subject_id,
                dilate_brainstem=getattr(args, 'dilate_brainstem', 2),
                dilate_motor=getattr(args, 'dilate_motor', 1),
                min_length=getattr(args, 'min_length', 20.0),
                max_length=getattr(args, 'max_length', 200.0),
                extraction_method=extraction_method,
                fast_registration=getattr(args, 'fast_registration', False),
                save_visualizations=getattr(args, 'save_visualizations', False),
                verbose=verbose,
                out=extract_out
            )
            
            extract_result = cmd_extract(extract_args)
        
        if extract_result:
            cst_left_path = extract_result.get('cst_left_path')
            cst_right_path = extract_result.get('cst_right_path')
            step_results['extract'] = {'success': True, 'result': extract_result}
            
            if extract_result['stats']['cst_total_count'] == 0:
                print("⚠ Warning: No CST streamlines extracted")
        else:
            raise RuntimeError("CST extraction failed or produced no streamlines")
            
    except Exception as e:
        print(f"✗ CST extraction failed: {e}")
        failed_steps.append('extract')
        step_results['extract'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['extract'] = time() - t0
    
    # =========================================================================
    # STEP 6: METRICS
    # =========================================================================
    if not quiet:
        print("\n" + "▶"*3 + " STEP 6/6: METRICS & REPORTS " + "◀"*3)
    t0 = time()
    
    try:
        if cst_left_path is None or cst_right_path is None:
            raise RuntimeError("No CST tractograms available")
        
        metrics_out = args.out / "metrics"
        metrics_args = argparse.Namespace(
            cst_left=cst_left_path,
            cst_right=cst_right_path,
            fa=fa_path,
            md=md_path,
            rd=rd_path,
            ad=ad_path,
            subject_id=subject_id,
            generate_pdf=getattr(args, 'generate_pdf', False),
            save_visualizations=getattr(args, 'save_visualizations', False),
            verbose=verbose,
            out=metrics_out,
            space=getattr(args, 'space', "Native Space"),
            # internal: pass pipeline metadata
            pipeline_metadata=pipeline_metadata if 'pipeline_metadata' in locals() else {}
        )
        
        metrics_result = cmd_metrics(metrics_args)
        
        if metrics_result:
            step_results['metrics'] = {'success': True, 'result': metrics_result}
        else:
            raise RuntimeError("Metrics computation failed")
            
    except Exception as e:
        print(f"✗ Metrics failed: {e}")
        failed_steps.append('metrics')
        step_results['metrics'] = {'success': False, 'error': str(e)}
    
    step_times['metrics'] = time() - t0

    # Clean up empty intermediate directory
    intermediate_dir = args.out / "intermediate"
    if intermediate_dir.exists() and not any(intermediate_dir.iterdir()):
        intermediate_dir.rmdir()
        if verbose:
            print("✓ Cleaned up empty intermediate directory")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time() - pipeline_start
    
    # Save pipeline report
    report_path = save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
    
    if not quiet:
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Subject ID:     {subject_id}")
        print(f"Total time:     {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
        print(f"\nStep timing:")
        for step, elapsed in step_times.items():
            status = "✓" if step not in failed_steps else "✗"
            print(f"  {status} {step:12s}: {elapsed:.1f}s")

        if failed_steps:
            print(f"\n  Failed steps: {', '.join(failed_steps)}")
        else:
            print(f"\n✓ All steps completed successfully!")

        print(f"\nOutputs:")
        print(f"  Pipeline report: {report_path}")
        if step_results.get('metrics', {}).get('success'):
            print(f"  Metrics:         {args.out / 'metrics'}")
        if step_results.get('extract', {}).get('success'):
            print(f"  CST tractograms: {args.out / 'extraction'}")

        print("="*70)
    elif failed_steps:
        # Even in quiet mode, report failures
        print(f"Pipeline completed with failures: {', '.join(failed_steps)}")
