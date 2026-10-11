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
   - if the version changed and the tag does not exist yet, tags `vx.y.z`, builds one
     wheel per platform (Linux x86_64 and aarch64, Windows x64, macOS arm64 and
     x86_64; each an abi3 wheel for Python 3.11+, with the compiled Rust kernels) and
     the sdist with maturin, and creates a GitHub release (marked pre-release) with
     them and the changelog section.

To check the wheel builds before a release: Actions -> Build wheels -> Run workflow
(`.github/workflows/wheels.yml`; builds without releasing).

No release is made if the version is unchanged, the tag exists, or tests fail.

The same run publishes the wheels and the sdist to PyPI (job `publish-pypi`) by
trusted publishing: no token, PyPI trusts this repository's `main.yml` in the GitHub
environment `pypi`. Set up once:

- PyPI: Account settings -> Publishing -> Add a new pending publisher: project
  `pynlme`, owner `willov`, repository `pynlme`, workflow `main.yml`, environment
  `pypi`.
- GitHub: Settings -> Environments -> New environment `pypi`, with a required
  reviewer, so that each upload waits for approval (a version uploaded to PyPI can
  never be replaced).

Users install from PyPI (`pip install pynlme`) or from the release:

```bash
# the wheel for your platform from the release page, e.g. Linux x86_64:
pip install pynlme-x.y.z-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# with SUND models:
pip install "pynlme[sund] @ file:///path/to/pynlme-x.y.z-cp311-abi3-<platform>.whl"
```

or from source (needs a Rust toolchain): `pip install git+https://github.com/willov/pynlme`.

Reference results in `validation/reference/` are regenerated separately (see
[validation/README.md](../validation/README.md)); they need R and MATLAB, not CI.
