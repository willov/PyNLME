# Release workflow

Merges to `main` only run the tests. To release, on an up-to-date `main`:

```bash
git checkout main && git pull
uv run python scripts/release.py x.y.z            # --dry-run shows the changes only
```

The script checks that `main` is clean and matches `origin/main` and that the tag is
new; sets the version in `pyproject.toml` and `rust/Cargo.toml`; dates the heading
`## [x.y.z] - unreleased` in `CHANGELOG.md` and `CITATION.cff`; shows the changes,
asks, then commits "Release vx.y.z", tags `vx.y.z` and pushes both.

The pushed tag starts the release in GitHub Actions (`.github/workflows/main.yml`):

- check that the tag is `v` plus `pyproject.toml`'s version and on `main`;
- run the tests (Linux, Windows, macOS; Python 3.11-3.14);
- build one wheel per platform (Linux x86_64 and aarch64, Windows x64, macOS arm64
  and x86_64; abi3 wheels for Python 3.11+ with the compiled Rust kernels) and the
  sdist with maturin;
- create a GitHub release (pre-release) with them and the changelog section;
- publish them to PyPI once you approve the deployment ("Review deployments" in the
  run; environment `pypi`).

Between releases, add notes under a heading `## [next version] - unreleased` at the
top of `CHANGELOG.md`.

Nothing is published if the tag does not match the version or is not on `main`, or if
the tests fail. The slow validation against R runs only when the workflow is started
by hand (Actions -> CI/CD Pipeline -> Run workflow). To check the wheel builds before
tagging: Actions -> Build wheels -> Run workflow (`.github/workflows/wheels.yml`).

PyPI publishing is trusted publishing: no token; PyPI trusts this repository's
`main.yml` in the environment `pypi`. Set up once:

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
