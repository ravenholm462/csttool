# Batch Processing Feature for csttool

## Design Decisions
- **Execution**: Sequential, one child process per subject (v1); pool-based parallelism (v2)
- **Process isolation**: multiprocessing.Process for safe timeout and memory reclamation
- **Manifest**: JSON format with optional series_uid for DICOM disambiguation
- **Input/Output**: BIDS-compliant with session support (`sub-*/ses-*/dwi/`)
- **Output**: BIDS-derivative-like (not fully compliant)
- **Resume**: By default via `_done.json` completion markers (source of truth)
- **Input types**: NIfTI and DICOM (with series disambiguation)
- **DICOM conversion**: Existing `ingest.convert_series`; explicit series_uid if ambiguous
- **Pipeline integration**: Internal Python API called from child process
- **Subject timeout**: 120 minutes default; enforced via process termination + 30s grace
- **Locking**: fcntl.flock() as sole authority
- **Output atomicity**: Atomic rename from _work/; _done.json written last

## Tasks

### Planning Phase ✓
- [x] Explore existing codebase architecture
- [x] Understand single-subject pipeline (`cmd_run`)
- [x] Draft implementation plan
- [x] Create batch module directory structure
- [x] Add BIDS session support to plan
- [x] Add DICOM discovery to plan
- [x] Add resume-by-default behavior
- [x] Add error handling strategy with categories
- [x] Add batch_metrics.csv schema
- [x] Add work directory strategy (`_work/` + `_failed_attempts/`)
- [x] Add locking mechanism
- [x] Add logging architecture
- [x] Add BIDS derivative metadata spec
- [x] Add batch_summary.json schema with versioning
- [x] Add subject sanitization
- [x] Cross-reference with 3 engineer reviews
- [x] Get user approval on revised plan
- [x] Add process isolation architecture
- [x] Add signal handling with grace period
- [x] Add DICOM series disambiguation
- [x] Add atomic output promotion
- [x] Add BIDS derivative scope clarification
- [x] Fix CLI preprocessing flag bug
- [x] Add metrics CSV NaN policy

### Implementation Phase
- [x] `modules/locking.py` - flock-based locking (metadata informational only)
- [x] `modules/logging_setup.py` - Per-subject log isolation (child process setup)
- [x] `modules/discover.py` - BIDS discovery + DICOM series detection
- [x] `modules/validation.py` - Preflight validation (environment, inputs, outputs)
- [x] `modules/manifest.py` - JSON manifest with series_uid support
- [x] `batch.py` - Process isolation, signal handling, atomic promotion
- [x] `modules/report.py` - Batch summary and metrics (NaN semantics)
- [x] CLI integration (`cli/commands/batch.py`)
- [x] Unit tests in `tests/batch/`

### Verification Phase
- [x] Run unit tests for each module
- [ ] Manual verification with mock BIDS dataset
- [ ] Peer review of `batch.py` orchestrator
- [x] Finalize `walkthrough.md`t validation (--validate-only, --dry-run)
- [ ] Test manifest loading/validation (with series_uid)
- [ ] Test process isolation and timeout handling
- [ ] Test signal handling with grace period (SIGINT/SIGTERM)
- [ ] Test atomic output promotion
- [ ] Integration test with multi-subject, multi-series dataset
