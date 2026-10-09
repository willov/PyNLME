# Contributing to PyNLME

## Setup

Python 3.11-3.14 and [uv](https://docs.astral.sh/uv/). pynlme is pure Python (NumPy,
SciPy); no compiler is needed except by SUND, which compiles model files.

```bash
git clone https://github.com/willov/PyNLME && cd PyNLME
uv sync --dev --extra sund     # leave out --extra sund to skip SUND
uv run pytest                  # the fast tests, as CI runs them
```

`uv sync` installs exactly what is requested; without `--extra sund` it uninstalls SUND.

## Tests

| Command | What runs |
|---|---|
| `uv run pytest` | Fast tests: unit tests and the stored MATLAB/R/saemix reference cases (CI, every push to `main` and pull requests) |
| `uv run pytest --runslow` | Also the simulation studies (bias, coverage) |
| with R (`nlme`, `jsonlite`) installed | Also the randomized comparison with R nlme |
| with `PYNLME_MATLAB` set | Also the randomized comparison with MATLAB nlmefit |

Run the slow checks by hand when changing algorithms; see
[validation/README.md](validation/README.md). In CI the R validation job runs only when
started from the Actions tab.

`compat="matlab"` and `compat="r"` follow MATLAB's and R's iterations step by step
(`src/pynlme/matlab_lb.py`, `fit_lindstrom_bates_nlme` in `lindstrom_bates.py`). A
change there must keep the randomized comparisons agreeing; see
[docs/implementation.md](docs/implementation.md).

## Code style

- `uv run ruff check src tests` (line length 88, Python 3.11 syntax).
- Docstrings for public functions; MATLAB and Python option names as in
  `docs/api_reference.md`.
- New behaviour comes with a test, against a reference (MATLAB, R) where one exists.
- Never copy code from MATLAB or R (their algorithms and formulas, not their code).

## Documentation

- `docs/api_reference.md` for new functions and options, `docs/implementation.md` for
  algorithm changes, `validation/README.md` for validation findings.
- `CHANGELOG.md` under `[Unreleased]`.
- Releases: [docs/release-workflow.md](docs/release-workflow.md).

## Reporting issues

Include the pynlme, Python, NumPy and SciPy versions, the operating system, a minimal
reproducible example, and the expected and actual results.
