#!/usr/bin/env bash
# run_determinism_test.sh — Run csttool N times and validate determinism.
#
# Runs the full csttool pipeline repeatedly with identical parameters and
# a controlled environment, then compares all outputs pairwise.
#
# Single-subject usage:
#   ./scripts/run_determinism_test.sh --subject-id sub-1204 --nifti PATH
#
# Multi-subject usage:
#   ./scripts/run_determinism_test.sh \
#       --subjects sub-1204 sub-1025 sub-1031 \
#       --nifti-pattern /mnt/neurodata/tractoinferno_raw/ds003900/derivatives/trainset/{subject}/dwi/{subject}__dwi.nii.gz
#
# Options:
#   --n-runs N             Number of runs per subject (default: 3)
#   --subject-id ID        Single subject identifier (default: sub-1204)
#   --nifti PATH           Single subject NIfTI file
#   --subjects ID...       Space-separated list of subject IDs (multi-subject mode)
#   --nifti-pattern PAT    Path pattern with {subject} placeholder (used with --subjects)
#   --field-strength F     Field strength in Tesla (default: 3.0)
#   --outdir DIR           Root output directory (default: auto-generated)
#   --reuse-run DIR        Existing run to reuse as run_0 (single-subject only)
#   --help                 Show this help

set -euo pipefail

# -----------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------
N_RUNS=3
SUBJECT_ID="sub-1204"
NIFTI="/mnt/neurodata/tractoinferno_raw/ds003900/derivatives/trainset/sub-1204/dwi/sub-1204__dwi.nii.gz"
NIFTI_PATTERN="/mnt/neurodata/tractoinferno_raw/ds003900/derivatives/trainset/{subject}/dwi/{subject}__dwi.nii.gz"
FIELD_STRENGTH="3.0"
OUTDIR=""
REUSE_RUN=""
SUBJECTS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-runs)         N_RUNS="$2";          shift 2 ;;
        --nifti)          NIFTI="$2";           shift 2 ;;
        --subject-id)     SUBJECT_ID="$2";      shift 2 ;;
        --nifti-pattern)  NIFTI_PATTERN="$2";   shift 2 ;;
        --field-strength) FIELD_STRENGTH="$2";  shift 2 ;;
        --outdir)         OUTDIR="$2";          shift 2 ;;
        --reuse-run)      REUSE_RUN="$2";       shift 2 ;;
        --subjects)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                SUBJECTS+=("$1")
                shift
            done
            ;;
        --help)
            head -30 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTDIR" ]]; then
    OUTDIR="/home/alem/data/thesis/determinism/$(date +%Y-%m-%d)_seed42"
fi

# If --subjects provided, use multi-subject mode; otherwise single-subject
if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
    MULTI_SUBJECT=1
else
    MULTI_SUBJECT=0
    SUBJECTS=("$SUBJECT_ID")
fi

# -----------------------------------------------------------------------
# Environment control — pin all threading to 1 for determinism
# -----------------------------------------------------------------------
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

echo "========================================================================"
echo "DETERMINISM VALIDATION TEST"
echo "========================================================================"
echo "  Subjects:       ${SUBJECTS[*]}"
echo "  Seed:           42 (default)"
echo "  N runs:         $N_RUNS"
echo "  Output dir:     $OUTDIR"
echo ""
echo "  Environment:"
echo "    PYTHONHASHSEED=$PYTHONHASHSEED"
echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "    OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "    MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "    NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
echo "========================================================================"

mkdir -p "$OUTDIR"

# -----------------------------------------------------------------------
# Helper: capture provenance for a run
# -----------------------------------------------------------------------
capture_provenance() {
    local run_idx="$1"
    local run_dir="$2"
    local wall_time="$3"
    local exit_code="$4"
    local subject_outdir="$5"
    local subject_id="$6"
    local nifti_path="$7"

    python3 -c "
import json, sys
sys.path.insert(0, '$(cd "$SCRIPT_DIR/.." && pwd)/src')
from csttool.reproducibility.provenance import get_provenance_dict
try:
    import csttool
    version = getattr(csttool, '__version__', 'unknown')
except Exception:
    version = 'unknown'
prov = get_provenance_dict()
prov['csttool_version'] = version
prov['run_index'] = $run_idx
prov['run_dir'] = '$run_dir'
prov['wall_time_seconds'] = $wall_time
prov['exit_code'] = $exit_code
prov['subject_id'] = '$subject_id'
prov['environment_variables'] = {
    'PYTHONHASHSEED': '${PYTHONHASHSEED}',
    'OMP_NUM_THREADS': '${OMP_NUM_THREADS}',
    'OPENBLAS_NUM_THREADS': '${OPENBLAS_NUM_THREADS}',
    'MKL_NUM_THREADS': '${MKL_NUM_THREADS}',
    'NUMEXPR_NUM_THREADS': '${NUMEXPR_NUM_THREADS}',
    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS}',
}
prov['command'] = 'csttool run --nifti $nifti_path --out $run_dir --subject-id $subject_id --field_strength $FIELD_STRENGTH --generate-pdf --save-visualizations --verbose'
# Capture BLAS backend (important for floating-point reproducibility)
try:
    import numpy as np
    blas_info = np.__config__.blas_opt_info if hasattr(np.__config__, 'blas_opt_info') else {}
    prov['blas_backend'] = {
        'libraries': blas_info.get('libraries', []),
        'library_dirs': blas_info.get('library_dirs', []),
    }
    try:
        import numpy.core._multiarray_umath as _mu
        prov['blas_backend']['cpu_features'] = getattr(_mu, '__cpu_features__', 'unknown')
    except Exception:
        pass
except Exception as e:
    prov['blas_backend'] = {'error': str(e)}
json.dump(prov, open('${subject_outdir}/run_${run_idx}_provenance.json', 'w'), indent=2)
"
}

# -----------------------------------------------------------------------
# Per-subject determinism test
# -----------------------------------------------------------------------
run_subject_test() {
    local sid="$1"
    local nifti_path="$2"
    local subject_outdir="$3"
    local reuse_run="$4"

    echo ""
    echo "###################################################################"
    echo "  SUBJECT: $sid"
    echo "  Input:   $nifti_path"
    echo "  Dir:     $subject_outdir"
    echo "###################################################################"

    if [[ ! -f "$nifti_path" ]]; then
        echo "ERROR: Input file not found: $nifti_path"
        return 1
    fi

    mkdir -p "$subject_outdir/comparisons"

    # --- Determine starting run index (reuse logic) ---
    local start_idx=0

    if [[ -n "$reuse_run" ]]; then
        echo "Checking provenance of existing run: $reuse_run"

        PROV_CHECK=$(python3 -c "
import json, sys, pathlib
sys.path.insert(0, '$(cd "$SCRIPT_DIR/.." && pwd)/src')
report_files = list(pathlib.Path('$reuse_run').glob('*_pipeline_report.json'))
if not report_files:
    print('NO_REPORT')
    sys.exit(0)
with open(report_files[0]) as f:
    report = json.load(f)
existing_version = report.get('pipeline_version', 'unknown')
try:
    import csttool
    current_version = getattr(csttool, '__version__', 'unknown')
except Exception:
    current_version = 'unknown'
if existing_version != current_version:
    print(f'VERSION_MISMATCH:{existing_version}:{current_version}')
else:
    print('OK')
" 2>&1)

        case "$PROV_CHECK" in
            OK)
                echo "  Provenance matches. Symlinking as run_0."
                ln -sfn "$(cd "$reuse_run" && pwd)" "$subject_outdir/run_0"
                capture_provenance 0 "$reuse_run" 0 0 "$subject_outdir" "$sid" "$nifti_path"
                start_idx=1
                ;;
            VERSION_MISMATCH*)
                echo "  WARNING: $PROV_CHECK — skipping reuse, running fresh."
                ;;
            NO_REPORT)
                echo "  WARNING: No pipeline report found in existing run — skipping reuse."
                ;;
            *)
                echo "  WARNING: Provenance check failed: $PROV_CHECK — skipping reuse."
                ;;
        esac
    fi

    # --- Execute runs ---
    for i in $(seq "$start_idx" $((N_RUNS - 1))); do
        local run_dir="$subject_outdir/run_$i"

        echo ""
        echo "========================================================================"
        echo "  [$sid] RUN $i / $((N_RUNS - 1))"
        echo "========================================================================"
        mkdir -p "$run_dir"

        local start_time
        start_time=$(date +%s)

        csttool run \
            --nifti "$nifti_path" \
            --out "$run_dir" \
            --subject-id "$sid" \
            --field-strength "$FIELD_STRENGTH" \
            --generate-pdf \
            --save-visualizations \
            --verbose \
            2>&1 | tee "$subject_outdir/run_${i}.log"

        local exit_code=${PIPESTATUS[0]}
        local end_time
        end_time=$(date +%s)
        local wall_time=$((end_time - start_time))

        echo ""
        echo "  Run $i completed in ${wall_time}s (exit code: $exit_code)"

        capture_provenance "$i" "$run_dir" "$wall_time" "$exit_code" "$subject_outdir" "$sid" "$nifti_path"

        if [[ "$exit_code" -ne 0 ]]; then
            echo "ERROR: Run $i failed (exit $exit_code). Check: $subject_outdir/run_${i}.log"
            return 1
        fi

        # Check pipeline report for silent step failures
        local report
        report=$(ls "$run_dir"/${sid}*_pipeline_report.json 2>/dev/null | head -1 || true)
        if [[ -n "$report" ]]; then
            local failed_steps
            failed_steps=$(python3 -c "
import json
with open('$report') as f:
    r = json.load(f)
failed = r.get('failed_steps', [])
success = r.get('execution', {}).get('success', True)
if failed or not success:
    print('FAILED: ' + (', '.join(failed) if failed else 'pipeline reported failure'))
" 2>/dev/null || true)
            if [[ -n "$failed_steps" ]]; then
                echo "ERROR: $failed_steps — Check: $subject_outdir/run_${i}.log"
                return 1
            fi
        fi
    done

    # --- Pairwise comparisons ---
    echo ""
    echo "========================================================================"
    echo "  [$sid] PAIRWISE COMPARISONS"
    echo "========================================================================"

    for i in $(seq 0 $((N_RUNS - 2))); do
        for j in $(seq $((i + 1)) $((N_RUNS - 1))); do
            echo ""
            echo "--- run_$i vs run_$j ---"
            python3 "$SCRIPT_DIR/compare_outputs.py" \
                "$subject_outdir/run_$i" "$subject_outdir/run_$j" \
                --subject-id "$sid" \
                --output "$subject_outdir/comparisons/run_${i}_vs_run_${j}.json"
        done
    done

    # --- Per-subject aggregate report ---
    echo ""
    echo "========================================================================"
    echo "  [$sid] AGGREGATE REPORT"
    echo "========================================================================"
    python3 "$SCRIPT_DIR/determinism_report.py" "$subject_outdir"

    echo ""
    echo "  [$sid] Done. Summary: $subject_outdir/determinism_summary.txt"
}

# -----------------------------------------------------------------------
# Main: run for each subject
# -----------------------------------------------------------------------
SUBJECT_VERDICTS=()

for sid in "${SUBJECTS[@]}"; do
    # Resolve NIfTI path
    if [[ "$MULTI_SUBJECT" -eq 1 ]]; then
        nifti_path="${NIFTI_PATTERN//\{subject\}/$sid}"
    else
        nifti_path="$NIFTI"
    fi

    # Per-subject output dir: flat if single-subject, nested if multi-subject
    if [[ "$MULTI_SUBJECT" -eq 1 ]]; then
        subject_outdir="$OUTDIR/$sid"
    else
        subject_outdir="$OUTDIR"
    fi

    reuse=""
    if [[ "$MULTI_SUBJECT" -eq 0 && -n "$REUSE_RUN" ]]; then
        reuse="$REUSE_RUN"
    fi

    run_subject_test "$sid" "$nifti_path" "$subject_outdir" "$reuse"

    # Collect per-subject verdict for multi-subject summary
    verdict=$(python3 -c "
import json
with open('$subject_outdir/determinism_summary.json') as f:
    s = json.load(f)
print(s.get('overall_verdict', 'UNKNOWN'))
" 2>/dev/null || echo "UNKNOWN")

    SUBJECT_VERDICTS+=("$sid:$verdict")
done

# -----------------------------------------------------------------------
# Multi-subject summary
# -----------------------------------------------------------------------
if [[ "$MULTI_SUBJECT" -eq 1 ]]; then
    echo ""
    echo "========================================================================"
    echo "  MULTI-SUBJECT SUMMARY"
    echo "========================================================================"
    echo ""

    ALL_BITWISE=1
    ANY_NONDET=0

    for entry in "${SUBJECT_VERDICTS[@]}"; do
        subject="${entry%%:*}"
        verdict="${entry##*:}"
        printf "  %-20s %s\n" "$subject" "$verdict"
        [[ "$verdict" != "BITWISE_IDENTICAL" ]] && ALL_BITWISE=0
        [[ "$verdict" == "NON_DETERMINISTIC" ]] && ANY_NONDET=1
    done

    echo ""
    if [[ "$ANY_NONDET" -eq 1 ]]; then
        echo "  OVERALL: NON_DETERMINISTIC (one or more subjects failed)"
    elif [[ "$ALL_BITWISE" -eq 1 ]]; then
        echo "  OVERALL: BITWISE_IDENTICAL (all subjects)"
    else
        echo "  OVERALL: TOLERANCE_IDENTICAL (all within tolerance, not all bitwise)"
    fi

    # Write to file
    {
        echo "Multi-Subject Determinism Summary"
        echo "Date: $(date +%Y-%m-%d)"
        echo "N subjects: ${#SUBJECTS[@]}"
        echo "N runs per subject: $N_RUNS"
        echo ""
        for entry in "${SUBJECT_VERDICTS[@]}"; do
            subject="${entry%%:*}"
            verdict="${entry##*:}"
            printf "%-20s %s\n" "$subject" "$verdict"
        done
        echo ""
        if [[ "$ANY_NONDET" -eq 1 ]]; then
            echo "OVERALL: NON_DETERMINISTIC"
        elif [[ "$ALL_BITWISE" -eq 1 ]]; then
            echo "OVERALL: BITWISE_IDENTICAL"
        else
            echo "OVERALL: TOLERANCE_IDENTICAL"
        fi
    } > "$OUTDIR/multi_subject_summary.txt"

    echo ""
    echo "Multi-subject summary: $OUTDIR/multi_subject_summary.txt"
fi

echo ""
echo "========================================================================"
echo "All done. Results in: $OUTDIR"
echo "========================================================================"
