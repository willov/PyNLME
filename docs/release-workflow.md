# Release workflow

Merges to `main` only run the tests. A release is made when you push a tag:

1. On `main`: set `version` in `pyproject.toml` and put the notes in `CHANGELOG.md`
   under a heading `## [x.y.z] - unreleased`; merge.
2. Tag the merged commit and push the tag:

   ```bash
   git checkout main && git pull
   git tag vx.y.z
   git push origin vx.y.z
   ```

3. GitHub Actions (`.github/workflows/main.yml`) then
   - checks that the tag is `v` plus `pyproject.toml`'s version and on `main`;
   - runs the tests (Linux, Windows, macOS; Python 3.11-3.14);
   - builds one wheel per platform (Linux x86_64 and aarch64, Windows x64, macOS arm64
     and x86_64; abi3 wheels for Python 3.11+ with the compiled Rust kernels) and the
     sdist with maturin;
   - creates a GitHub release (pre-release) with them and the changelog section;
   - publishes them to PyPI once you approve the deployment ("Review deployments" in
     the run; environment `pypi`);
   - commits the release date to `main` (the version's heading in `CHANGELOG.md` and
     `date-released` in `CITATION.cff`). If `main` is protected, allow GitHub Actions
     to push to it.

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
