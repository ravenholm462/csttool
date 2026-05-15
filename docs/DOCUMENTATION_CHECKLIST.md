# csttool Documentation Checklist

**Last Updated**: 2026-05-15

This checklist tracks documentation progress for all pages defined in `mkdocs.yml`. The full pass writing every empty page completed on 2026-05-15.

---

## Progress Summary

| Section | Complete | Total | Status |
|---------|----------|-------|--------|
| Getting Started | 3 | 3 | ✅ Completed |
| Tutorials | 2 | 2 | ✅ Completed |
| How-To Guides | 3 | 3 | ✅ Completed |
| CLI Reference | 12 | 12 | ✅ Completed |
| API Reference | 4 | 4 | ✅ Completed (mkdocstrings auto-generated) |
| Explanation | 5 | 5 | ✅ Completed (+ References page) |
| Contributing | 3 | 3 | ✅ Completed |

---

## Site build

The site builds cleanly under `mkdocs build --strict`. Run:

```bash
source .venv/bin/activate
mkdocs build --strict
```

Internal-only pages (seminar/, fixes/, DOCUMENTATION_CHECKLIST.md, v0.4.0-updates.md, section landing pages) are filtered via `exclude_docs:` in `mkdocs.yml`.

---

## Citations needing sources

The Explanation pages cite from `thesis/references.bib`. No `[citation needed]` markers were introduced during the 2026-05-15 pass — every claim that needed support resolved to an existing bib entry. If a future contributor adds material that cannot be sourced from `references.bib`, mark it inline as `[citation needed]` and record it here so it can be sourced later. Do not fabricate citations.

---

## Writing Guidelines (kept for contributors)

1. **Keep it concise** — Users want answers, not essays.
2. **Show, don't tell** — Use code examples liberally.
3. **Test your examples** — Every command should work.
4. **Link related pages** — Help users navigate.
5. **Use admonitions** — `!!! note`, `!!! warning`, `!!! tip`.
6. **Cite from the bib** — For scholarly statements on Explanation pages, link to anchors on `explanation/references.md`. New citations need a matching entry in `thesis/references.bib` first.
