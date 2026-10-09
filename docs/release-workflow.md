# Release workflow

Releases are created automatically when the version in `pyproject.toml` changes on
`main`.

1. Update `version` in `pyproject.toml` and move the `[Unreleased]` notes in
   `CHANGELOG.md` under a heading `## [x.y.z] - YYYY-MM-DD`.
2. Commit and push (or merge) to `main`.
3. GitHub Actions (`.github/workflows/main.yml`):
   - runs the tests on Linux, Windows and macOS with Python 3.11-3.14 (the fast tests;
     the slow validation against R, the `validation` job, runs only when the workflow
     is started by hand: Actions -> CI/CD Pipeline -> Run workflow);
   - if the version changed and the tag does not exist yet, tags `vx.y.z`, builds the
     wheel and sdist (`uv build`; pynlme is pure Python, so one `py3-none-any` wheel
     works everywhere) and creates a GitHub release (marked pre-release) with them and
     the changelog section.

No release is made if the version is unchanged, the tag exists, or tests fail.

Users install from the release:

```bash
pip install https://github.com/willov/PyNLME/releases/download/vx.y.z/pynlme-x.y.z-py3-none-any.whl
# with SUND models:
pip install "pynlme[sund] @ https://github.com/willov/PyNLME/releases/download/vx.y.z/pynlme-x.y.z-py3-none-any.whl"
```

or from source: `pip install git+https://github.com/willov/PyNLME`.

Reference results in `validation/reference/` are regenerated separately (see
[validation/README.md](../validation/README.md)); they need R and MATLAB, not CI.
