# Contributing

Thanks for considering a contribution to csttool. Two practical pages cover the basics:

- [Development setup](development-setup.md) — clone, virtualenv, dev install, test runner, local docs.
- [Code style](code-style.md) — formatting, linting, naming conventions.
- [Architecture](architecture.md) — a guided tour of `src/csttool/`.

## Workflow

1. Fork the repo and clone your fork.
2. Create a topic branch (`feat/short-description`, `fix/short-description`).
3. Run `pytest` locally and confirm tests pass.
4. Build the docs locally (`mkdocs serve`) and confirm any pages you touched render.
5. Open a pull request against `main`. Reference any related issue.

## What we look for in PRs

- A clear, scoped change. One feature / one fix per PR.
- Tests for any new behaviour. New CLI flags should be exercised by the integration suite under `tests/`.
- Documentation updates: every new flag should land with a matching row in [`reference/parameters.md`](../reference/parameters.md) and a sentence in the relevant CLI-reference page.
- For changes to the science (extraction methods, tracking parameters, registration profiles): a short justification in the PR description and, if relevant, a note in [`explanation/design-decisions.md`](../explanation/design-decisions.md).

## Bug reports

Please include:

- The exact `csttool` command and flags you ran.
- The full traceback or last 20 lines of CLI output.
- `csttool --version` and `python --version`.
- The dataset shape (DWI dimensions, b-values, gradient count) — output from `csttool check-dataset` is ideal.
