"""
CLI command to fetch FSL-licensed atlas data (Tier 2).
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

from ...data.manifest import get_tier2_files, verify_checksum, get_manifest_entry
from ...data.loader import get_user_data_dir


FSL_LICENSE_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════╗
║                       FSL LICENSE ACKNOWLEDGMENT                         ║
╔══════════════════════════════════════════════════════════════════════════╗

This command will download the following FSL-licensed data:
  • FMRIB58_FA template and skeleton
  • Harvard-Oxford cortical and subcortical atlases

LICENSE: FSL Non-Commercial Use Only

The FSL Software Library and associated data are provided free of charge for
NON-COMMERCIAL use only.

DEFINITION OF COMMERCIAL USE:
Commercial use includes, but is not limited to:
  • Use in connection with development, testing, or commercialization of
    products or services
  • Use in a commercial organization where the work contributes to
    commercial activities
  • Any use that results in monetary compensation

For academic research and non-commercial use, you may use this data freely.
For commercial use, you MUST obtain a commercial license from the University
of Oxford.

SOURCE: University of Oxford, FMRIB Centre
FULL LICENSE: https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html

╚══════════════════════════════════════════════════════════════════════════╝
"""


def _prompt_license_acceptance() -> bool:
    """
    Interactively prompt user to accept FSL license.

    Returns
    -------
    bool
        True if user accepts, False otherwise
    """
    print(FSL_LICENSE_SUMMARY)
    print("\nDo you accept the FSL non-commercial license terms?")
    print("By typing 'yes', you confirm that your use is non-commercial.")
    print()

    response = input("Accept FSL license? (yes/no): ").strip().lower()
    return response in ("yes", "y")


def _download_file(url: str, dest_path: Path, expected_sha256: str) -> bool:
    """
    Download file from URL and verify checksum.

    Parameters
    ----------
    url : str
        Source URL
    dest_path : Path
        Destination file path
    expected_sha256 : str
        Expected SHA256 checksum

    Returns
    -------
    bool
        True if download and verification successful, False otherwise
    """
    try:
        # Download to temporary file first
        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading {dest_path.name}...")
        urllib.request.urlretrieve(url, temp_path)

        # Verify checksum
        print(f"  Verifying checksum...")
        if not verify_checksum(temp_path, expected_sha256):
            print(f"  ✗ Checksum verification failed for {dest_path.name}")
            print(f"    This may indicate a corrupted download or modified source file.")
            temp_path.unlink()
            return False

        # Move to final location
        temp_path.rename(dest_path)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded and verified {dest_path.name} ({size_mb:.2f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Error downloading {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def cmd_fetch_data(args) -> dict | None:
    """
    Fetch FSL-licensed atlas data (FMRIB58_FA, Harvard-Oxford).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    dict or None
        Result dictionary with download info, or None on failure
    """
    print("=" * 76)
    print("FETCH FSL-LICENSED ATLAS DATA")
    print("=" * 76)
    print()

    # Check license acceptance
    accept_license = getattr(args, "accept_fsl_license", False)

    if not accept_license:
        # Interactive prompt
        if not _prompt_license_acceptance():
            print("\n  ✗ License not accepted. Data download cancelled.")
            return None
    else:
        print("FSL license accepted via --accept-fsl-license flag.")
        print()

    # Get user data directory
    data_dir = get_user_data_dir()
    print(f"Installation directory: {data_dir}")
    print()

    # Get Tier 2 files from manifest
    tier2_files = get_tier2_files()
    print(f"Files to download: {len(tier2_files)}")
    print()

    # Download files
    results = {}
    failed_downloads = []

    for manifest_key, manifest_entry in tier2_files.items():
        url = manifest_entry["source_url"]
        sha256 = manifest_entry["sha256"]

        # Construct destination path
        dest_path = data_dir / manifest_key

        # Check if already exists and valid
        if dest_path.exists():
            if verify_checksum(dest_path, sha256):
                print(f"  → {dest_path.name} already exists and is valid (skipping)")
                results[manifest_key] = {
                    "source_url": url,
                    "sha256": sha256,
                    "sha256_verified": True,
                    "size_bytes": dest_path.stat().st_size,
                    "status": "already_installed",
                }
                continue

        # Download and verify
        success = _download_file(url, dest_path, sha256)

        if success:
            results[manifest_key] = {
                "source_url": url,
                "sha256": sha256,
                "sha256_verified": True,
                "size_bytes": dest_path.stat().st_size,
                "status": "downloaded",
            }
        else:
            failed_downloads.append(manifest_key)

        print()

    # Check for failures
    if failed_downloads:
        print("=" * 76)
        print(f"✗ Download failed for {len(failed_downloads)} file(s):")
        for key in failed_downloads:
            print(f"  - {key}")
        print("=" * 76)
        return None

    # Write .metadata.json
    metadata_path = data_dir / ".metadata.json"
    metadata = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "csttool_version": version("csttool"),
        "fsl_data_standard_tag": "2208.0",
        "fsl_data_atlases_tag": "2103.0",
        "files": results,
        "license_accepted": "FSL non-commercial",
        "license_accepted_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata written to {metadata_path}")

    # Write .validated stamp
    from ...data.loader import _save_validation_stamp

    stamp_data = {}
    for manifest_key in results.keys():
        file_path = data_dir / manifest_key
        stamp_data[str(file_path)] = {
            "size": file_path.stat().st_size,
            "mtime": file_path.stat().st_mtime,
            "sha256_ok": True,
        }

    _save_validation_stamp(stamp_data)
    print(f"  ✓ Validation stamp written")
    print()

    # Summary
    print("=" * 76)
    print("✓ DATA FETCH COMPLETE")
    print("=" * 76)
    print()
    print(f"Downloaded {len(results)} file(s) to {data_dir}")
    total_size_mb = sum(r["size_bytes"] for r in results.values()) / (1024 * 1024)
    print(f"Total size: {total_size_mb:.2f} MB")
    print()
    print("You can now run csttool commands that require FSL atlas data.")
    print()

    return {
        "data_dir": str(data_dir),
        "files_downloaded": len(results),
        "total_size_bytes": sum(r["size_bytes"] for r in results.values()),
    }
