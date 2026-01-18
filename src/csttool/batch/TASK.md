# Batch Processing Feature for csttool

## Design Decisions
- **Execution**: Sequential (v1), threading later (v2)
- **Manifest**: JSON format
- **Input/Output**: BIDS-compliant with session support (`sub-*/ses-*/dwi/`)

## Tasks

### Planning Phase
- [x] Explore existing codebase architecture
- [x] Understand single-subject pipeline (`cmd_run`)
- [x] Draft implementation plan
- [x] Create batch module directory structure
- [x] Add BIDS session support to plan
- [ ] Get user approval on revised plan

### Implementation Phase
- [ ] `modules/discover.py` - BIDS subject/session discovery
- [ ] `modules/manifest.py` - JSON manifest loading
- [ ] `batch.py` - Sequential batch orchestrator
- [ ] `modules/report.py` - Batch summary generation
- [ ] CLI integration in `cli.py`
- [ ] Unit tests in `tests/batch/`

### Verification Phase
- [ ] Test BIDS discovery (with/without sessions)
- [ ] Test manifest loading/validation
- [ ] Integration test with multi-subject data
