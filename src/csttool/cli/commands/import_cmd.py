
import argparse
import shutil
from pathlib import Path

from ..utils import resolve_nifti, load_with_preproc


def cmd_import(args: argparse.Namespace) -> dict | None:
    """Import DICOM data or load an existing NIfTI dataset."""

    # Route to raw-BIDS import when --raw-bids is specified
    if getattr(args, 'raw_bids', None):
        return cmd_import_raw_bids(args)

    try:
        from csttool.ingest import run_ingest_pipeline, scan_study
        USE_INGEST = True
    except ImportError:
        USE_INGEST = False
        print("  ⚠️ Ingest module not available, using legacy import")

    if USE_INGEST and args.dicom:
        args.out.mkdir(parents=True, exist_ok=True)

        if getattr(args, 'scan_only', False):
            print(f"  → Scanning DICOM directory: {args.dicom}")
            series_list = scan_study(args.dicom)
            if not series_list:
                print("  ⚠️ No valid DICOM series found")
                return None
            print(f"  ✓ Found {len(series_list)} series")
            return {'series': series_list, 'scan_only': True}

        result = run_ingest_pipeline(
            study_dir=args.dicom,
            output_dir=args.out,
            series_index=getattr(args, 'series', None),
            series_uid=getattr(args, 'series_uid', None),
            subject_id=args.subject_id,
            verbose=getattr(args, 'verbose', False),
            field_strength=getattr(args, 'field_strength', None),
            echo_time=getattr(args, 'echo_time', None),
        )

        if result and result.get('nifti_path'):
            print(f"\n✓ Import complete")
            print(f"  {result['nifti_path']}")
            _report_converter(result)
            return result
        else:
            print("  ✗ Import failed")
            return None

    else:
        return cmd_import_legacy(args)


def cmd_import_raw_bids(args: argparse.Namespace) -> dict | None:
    """
    Scanner-dump → raw BIDS dataset.

    Places dcm2niix output in:
        <raw_bids>/sub-<id>/ses-<session>/dwi/
    and writes dataset_description.json + participants.tsv at the dataset root.

    Anonymisation is ON by default.  Pass --keep-phi to disable.
    """
    from csttool.ingest import run_ingest_pipeline, scan_study
    from csttool.bids.output import (
        write_dataset_description,
        update_participants_tsv,
        parse_dicom_age,
        hash_patient_id,
        bids_filename,
        sanitize_bids_label,
    )

    raw_bids = Path(args.raw_bids)
    keep_phi = getattr(args, 'keep_phi', False)

    if not args.dicom:
        print("  ✗ --raw-bids requires --dicom")
        return None

    # ------------------------------------------------------------------
    # 1. Determine subject / session labels
    # ------------------------------------------------------------------
    subject_label = getattr(args, 'subject_id', None)
    session_label = getattr(args, 'session_id', None)

    patient_id = None
    patient_age = None
    patient_sex = None
    study_date = None

    if not subject_label or not session_label:
        patient_id, patient_age, patient_sex, study_date = _read_dicom_patient_tags(
            args.dicom
        )

    if not subject_label:
        if patient_id is None:
            print(
                "  ✗ Could not read PatientID from DICOM. "
                "Use --subject-id to specify it explicitly."
            )
            return None
        if keep_phi:
            print(
                "  ⚠️  WARNING: --keep-phi is set. "
                "Output subject label contains PHI (PatientID). "
                "This dataset is NOT de-identified."
            )
            subject_label = sanitize_bids_label(patient_id)
        else:
            subject_label = hash_patient_id(patient_id)

    if not session_label:
        if study_date:
            session_label = study_date.replace("-", "")
        else:
            session_label = "01"

    # Ensure labels don't already include the prefix
    if subject_label.startswith("sub-"):
        subject_label = subject_label[4:]
    if session_label.startswith("ses-"):
        session_label = session_label[4:]

    subject_id = f"sub-{subject_label}"
    session_id = f"ses-{session_label}"

    # ------------------------------------------------------------------
    # 2. Run ingest pipeline to a temp directory
    # ------------------------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory(prefix="csttool_import_") as tmpdir:
        tmp_out = Path(tmpdir) / "convert"
        result = run_ingest_pipeline(
            study_dir=args.dicom,
            output_dir=tmp_out,
            subject_id=subject_label,
            verbose=getattr(args, 'verbose', False),
            field_strength=getattr(args, 'field_strength', None),
            echo_time=getattr(args, 'echo_time', None),
        )

        if not result or not result.get('nifti_path'):
            print("  ✗ DICOM conversion failed")
            return None

        series_desc = ""
        if result.get('series_analysis'):
            series_desc = result['series_analysis'].series_description or ""

        # ------------------------------------------------------------------
        # 3. Build BIDS output paths
        # ------------------------------------------------------------------
        dwi_dir = raw_bids / subject_id / session_id / "dwi"
        dwi_dir.mkdir(parents=True, exist_ok=True)

        # Base BIDS name for the DWI file (no desc entity on raw)
        bids_stem = bids_filename(
            subject=subject_label,
            suffix="dwi",
            extension="",
            session=session_label,
        )
        # Remove trailing underscore before suffix
        bids_stem = bids_stem.rstrip("_")

        # Copy files with BIDS names
        nii_src = Path(result['nifti_path'])
        nii_dst = dwi_dir / f"{bids_stem}.nii.gz"
        shutil.copy2(nii_src, nii_dst)

        bval_dst = bvec_dst = json_dst = None

        if result.get('bval_path'):
            bval_dst = dwi_dir / f"{bids_stem}.bval"
            shutil.copy2(result['bval_path'], bval_dst)

        if result.get('bvec_path'):
            bvec_dst = dwi_dir / f"{bids_stem}.bvec"
            shutil.copy2(result['bvec_path'], bvec_dst)

        if result.get('json_path'):
            json_dst = dwi_dir / f"{bids_stem}.json"
            shutil.copy2(result['json_path'], json_dst)

    # ------------------------------------------------------------------
    # 4. Write BIDS dataset-level files
    # ------------------------------------------------------------------
    write_dataset_description(raw_bids, dataset_type="raw")

    age_val = parse_dicom_age(patient_age) if patient_age else None
    update_participants_tsv(
        raw_bids,
        subject_id,
        metadata={"age": age_val, "sex": patient_sex},
    )

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n✓ Raw BIDS import complete")
    print(f"  Subject:  {subject_id}")
    print(f"  Session:  {session_id}")
    print(f"  DWI:      {nii_dst}")
    if bval_dst:
        print(f"  bval:     {bval_dst}")
    if json_dst:
        print(f"  Sidecar:  {json_dst}")
    print(f"  Dataset:  {raw_bids}")
    _report_converter(result)

    return {
        'nifti_path': nii_dst,
        'bval_path': bval_dst,
        'bvec_path': bvec_dst,
        'json_path': json_dst,
        'subject_id': subject_id,
        'session_id': session_id,
        'raw_bids': raw_bids,
        'metadata': result.get('metadata', {}),
    }


def cmd_import_legacy(args: argparse.Namespace) -> dict | None:
    """Legacy import using preproc functions."""
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return None

    data, _affine, hdr, gtab, bids_json = load_with_preproc(nii)

    print(f"\n✓ Dataset loaded")
    print(f"  File:       {nii}")
    print(f"  Shape:      {data.shape}")
    print(f"  Directions: {len(gtab.bvals)}")
    voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
    print(f"  Voxel size: {voxel_size[0]:.2f} x {voxel_size[1]:.2f} x {voxel_size[2]:.2f} mm")
    print(f"  B-values:   {sorted(set(gtab.bvals.astype(int)))}")

    overrides = {}
    if getattr(args, 'field_strength', None):
        overrides['field_strength_T'] = args.field_strength
    if getattr(args, 'echo_time', None):
        overrides['echo_time_ms'] = args.echo_time

    from csttool.ingest import extract_acquisition_metadata
    acquisition = extract_acquisition_metadata(
        bvecs=gtab.bvecs,
        bvals=gtab.bvals,
        voxel_size=voxel_size,
        bids_json=bids_json,
        overrides=overrides,
    )

    return {
        'nifti_path': nii,
        'data_shape': data.shape,
        'n_gradients': len(gtab.bvals),
        'metadata': acquisition,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_dicom_patient_tags(dicom_dir: Path):
    """
    Read PatientID, PatientAge, PatientSex, StudyDate from the first DICOM
    file in *dicom_dir*.  Returns (patient_id, age_str, sex_str, study_date)
    with None for missing fields.
    """
    try:
        import pydicom
    except ImportError:
        return None, None, None, None

    dicom_dir = Path(dicom_dir)
    candidates = (
        list(dicom_dir.glob("*.dcm"))
        + list(dicom_dir.glob("*.DCM"))
        + [f for f in dicom_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    )
    if not candidates:
        # Recurse one level (study root with series subdirs)
        for sub in dicom_dir.iterdir():
            if sub.is_dir():
                candidates = list(sub.glob("*.dcm")) or list(sub.glob("*.DCM"))
                if candidates:
                    break

    if not candidates:
        return None, None, None, None

    try:
        ds = pydicom.dcmread(str(candidates[0]), stop_before_pixels=True)
        patient_id = str(getattr(ds, "PatientID", "") or "").strip() or None
        age_str = str(getattr(ds, "PatientAge", "") or "").strip() or None
        sex_str = str(getattr(ds, "PatientSex", "") or "").strip() or None
        study_date_raw = str(getattr(ds, "StudyDate", "") or "").strip()
        # DICOM StudyDate is YYYYMMDD — keep as-is for session label
        study_date = study_date_raw if len(study_date_raw) == 8 else None
        return patient_id, age_str, sex_str, study_date
    except Exception:
        return None, None, None, None


def _report_converter(result: dict) -> None:
    """Print a one-line converter note if a fallback was used."""
    if not result:
        return
    converter = result.get('converter')
    fallback = result.get('fallback_used', False)
    if converter and fallback:
        print(f"  ⚠️  Used dicom2nifti fallback (dcm2niix unavailable or failed)")
    elif converter:
        print(f"  ✓  Converter: {converter}")
