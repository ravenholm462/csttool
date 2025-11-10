#!/usr/bin/env bash
# ==============================================
# Test script for csttool's CLI
# ==============================================
set -euo pipefail  # safer execution
IFS=$'\n\t'

echo "Starting csttool test..."

# Input dir paths
read -rp "NIfTI file path: " NIFTI_FILE
read -rp "DICOM dir path: " DICOM_DIR
read -rp "Out dir path: " OUTDIR

echo "=============================================="
echo "Verifying output directory..."
echo "=============================================="

if [[ -z "$OUTDIR" ]]; then
    echo "Error: output directory path cannot be empty."
    exit 1
fi

if [[ -d "$OUTDIR" ]]; then
    echo "Output directory exists: $OUTDIR"
else
    echo "Output directory not found — creating: $OUTDIR"
    if mkdir -p "$OUTDIR"; then
        echo "Successfully created $OUTDIR"
    else
        echo "Error: could not create $OUTDIR"
        exit 1
    fi
fi

echo "=============================================="
echo "Checking tool availability..."
echo "=============================================="

# Environment test
if ! command -v csttool >/dev/null 2>&1; then
    echo "Error: csttool command not found. Did you install the package?"
    exit 1
fi

echo 
echo "Running environment check..."
csttool check

# NIfTI import test
echo
echo "Running csttool import..."
if [ -f "$NIFTI_FILE" ]; then
    csttool import --nifti "$NIFTI_FILE" --out "$OUTDIR"
else
    echo "Warning: sample NIfTI not found at $NIFTI_FILE"
fi

# Preprocessing test
echo
echo "Running csttool preprocess (no motion correction)..."
if [ -f "$NIFTI_FILE" ]; then
    csttool preprocess \
        --nifti "$NIFTI_FILE" \
        --out "$OUTDIR" \
        --coil-count 4 \
        --skip-motion-correction \
        --show-plots
else
    echo "Skipping preprocess (no NIfTI file present)."
fi

# DICOM test
echo
echo "Running csttool import (DICOM)"
if [ -d "$DICOM_DIR" ]; then
    csttool import --dicom "$DICOM_DIR" --out "$OUTDIR"
else
    echo "Warning: test DICOM folder not found at $DICOM_DIR"
fi

# Done
echo
echo "=============================================="
echo "CLI test script completed."
echo "Output directory: $OUTDIR"
echo "=============================================="