# fetch-data Command

The `fetch-data` command downloads FSL-licensed atlas data required for CST extraction. This includes the FMRIB58_FA template and Harvard-Oxford atlases used for anatomical registration and region-of-interest definition.

## Overview

CST extraction requires anatomical reference atlases to define motor cortex and brainstem regions. These atlases are licensed by the University of Oxford under the FSL Non-Commercial License and must be downloaded separately due to licensing restrictions.

The `fetch-data` command automates:

- License acknowledgment and acceptance
- Downloading atlas files from official FSL sources
- SHA256 checksum verification for data integrity
- Installation to the user data directory
- Metadata tracking for reproducibility

## Usage

### Interactive Mode (Default)

```bash
csttool fetch-data
```

This will display the FSL license terms and prompt for acceptance. You must type `yes` to proceed with the download.

### Non-Interactive Mode

```bash
csttool fetch-data --accept-fsl-license
```

Use this flag to accept the license without an interactive prompt. Suitable for automated pipelines or scripted installations.

## FSL License Summary

**License Type**: FSL Non-Commercial Use Only

**Permitted Uses**:

- Academic research
- Educational purposes
- Non-commercial scientific work

**Prohibited Uses**:

- Commercial product development or testing
- Use in commercial organizations where work contributes to commercial activities
- Any use resulting in monetary compensation

**Full License**: <https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html>

By accepting this license, you confirm that your use is non-commercial. For commercial use, you must obtain a commercial license from the University of Oxford.

## Downloaded Data

The command downloads the following files:

| File | Description | Size |
|------|-------------|------|

| `FMRIB58_FA_1mm.nii.gz` | FMRIB58 FA template (1mm resolution) | ~1 MB |
| `FMRIB58_FA-skeleton_1mm.nii.gz` | FMRIB58 FA skeleton mask | ~0.1 MB |
| `HarvardOxford-cort-maxprob-thr25-1mm.nii.gz` | Harvard-Oxford cortical atlas | ~0.5 MB |
| `HarvardOxford-sub-maxprob-thr25-1mm.nii.gz` | Harvard-Oxford subcortical atlas | ~0.5 MB |

**Total Download Size**: ~2 MB

## Installation Directory

Data is installed to the platform-specific user data directory:

- **Linux**: `~/.local/share/csttool/`
- **macOS**: `~/Library/Application Support/csttool/`
- **Windows**: `%LOCALAPPDATA%\csttool\`

## Checksum Verification

All downloaded files are verified using SHA256 checksums to ensure data integrity. If verification fails, the download is rejected and you'll see an error message:

```text
✗ Checksum verification failed for FMRIB58_FA_1mm.nii.gz
  This may indicate a corrupted download or modified source file.
```

If this occurs, try running the command again. Persistent failures may indicate network issues or upstream changes to the source files.

## Skipping Already-Downloaded Files

If you run `fetch-data` multiple times, it will detect existing files and skip re-downloading them:

```text
→ FMRIB58_FA_1mm.nii.gz already exists and is valid (skipping)
```

This allows safe re-runs without wasting bandwidth.

## Example Output

```text
============================================================================
FETCH FSL-LICENSED ATLAS DATA
============================================================================

╔══════════════════════════════════════════════════════════════════════════╗
║                       FSL LICENSE ACKNOWLEDGMENT                         ║
╔══════════════════════════════════════════════════════════════════════════╗

This command will download the following FSL-licensed data:
  • FMRIB58_FA template and skeleton
  • Harvard-Oxford cortical and subcortical atlases

LICENSE: FSL Non-Commercial Use Only
[... license text ...]

Do you accept the FSL non-commercial license terms?
By typing 'yes', you confirm that your use is non-commercial.

Accept FSL license? (yes/no): yes

Installation directory: /home/user/.local/share/csttool

Files to download: 4

  Downloading FMRIB58_FA_1mm.nii.gz...
  Verifying checksum...
  ✓ Downloaded and verified FMRIB58_FA_1mm.nii.gz (0.95 MB)

  Downloading FMRIB58_FA-skeleton_1mm.nii.gz...
  Verifying checksum...
  ✓ Downloaded and verified FMRIB58_FA-skeleton_1mm.nii.gz (0.08 MB)

  Downloading HarvardOxford-cort-maxprob-thr25-1mm.nii.gz...
  Verifying checksum...
  ✓ Downloaded and verified HarvardOxford-cort-maxprob-thr25-1mm.nii.gz (0.51 MB)

  Downloading HarvardOxford-sub-maxprob-thr25-1mm.nii.gz...
  Verifying checksum...
  ✓ Downloaded and verified HarvardOxford-sub-maxprob-thr25-1mm.nii.gz (0.45 MB)

  ✓ Metadata written to /home/user/.local/share/csttool/.metadata.json
  ✓ Validation stamp written

============================================================================
✓ DATA FETCH COMPLETE
============================================================================

Downloaded 4 file(s) to /home/user/.local/share/csttool
Total size: 1.99 MB

You can now run csttool commands that require FSL atlas data.
```

## Error Handling

### Download Failure

If a file fails to download, the command will report the failure and exit:

```text
✗ Error downloading FMRIB58_FA_1mm.nii.gz: Connection timeout
```

Possible causes:

- Network connectivity issues
- Firewall blocking HTTPS connections
- Upstream server unavailable

### License Declined

If you decline the license terms, the command exits immediately:

```text
✗ License not accepted. Data download cancelled.
```

## Metadata and Validation

After successful download, the command writes two tracking files:

1. **`.metadata.json`**: Records download timestamp, csttool version, FSL data versions, file checksums, and license acceptance
2. **`.validated` stamp**: Enables fast validation on subsequent runs

These files ensure data provenance and enable reproducible research by tracking exactly which atlas versions were used.

## Integration with Extract Pipeline

Once atlas data is downloaded, it's automatically detected by the `extract` command:

```bash
# This will now work without additional setup
csttool extract \
    --tractogram whole_brain.trk \
    --fa subject_FA.nii.gz \
    --out extracted_cst/
```

If atlas data is missing when you run `extract`, you'll see a helpful error:

```text
✗ Required atlas data not found. Please run: csttool fetch-data
```

## Troubleshooting

### Permission Errors

If you encounter permission errors during installation:

```bash
# Check user data directory permissions
ls -ld ~/.local/share/csttool
```

Ensure the directory is writable by your user account.

### Disk Space

Verify you have sufficient disk space (~10 MB recommended to account for temporary download files):

```bash
df -h ~/.local/share
```

### Network Issues Behind Proxy

If you're behind a corporate proxy, ensure your environment variables are set:

```bash
export http_proxy="http://proxy.example.com:8080"
export https_proxy="http://proxy.example.com:8080"
csttool fetch-data --accept-fsl-license
```

### Offline Installation

The `fetch-data` command requires internet connectivity. For offline or air-gapped environments, you can manually place the atlas files in the user data directory, but you must ensure:

1. Files are named exactly as specified in the manifest
2. SHA256 checksums match expected values
3. `.metadata.json` is created with proper structure

Contact the csttool developers for offline installation bundles.

## See Also

- [extract](extract.md) — CST extraction command that uses downloaded atlases
- [check](check.md) — Environment validation (does not check atlas data)
- [Installation Guide](../../getting-started/installation.md) — Initial setup instructions
- [Data Requirements](../../getting-started/data-requirements.md) — Overview of required data files
