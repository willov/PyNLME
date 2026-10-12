"""
nlmefit (compat="matlab", the default) against MATLAB nlmefit on freshly
simulated datasets.

Needs MATLAB: set PYNLME_MATLAB to the command that starts it, e.g.
``PYNLME_MATLAB="/usr/local/MATLAB/R2023b/bin/matlab"`` (environment
assignments in front are allowed, e.g. ``LD_PRELOAD=... /path/to/matlab``).
PYNLME_RANDOM_SEED and PYNLME_RANDOM_N as in test_random_vs_r.py.

compat="matlab" follows nlmefit's iteration step by step with its default
options (matlab_lb.py). On most datasets the fit is MATLAB's to ~1e-10. Two
things limit this, both inside MATLAB itself:

- nlmefit's LME step runs fminsearch to a tolerance of 1e-2 (100 x TolX), and
  where it stops can flip with perturbations at the 1e-9 level (seed 1007,
  logistic_6: rounding MATLAB's own starting theta to 8 digits moves its
  estimates by 2e-5). Such fits agree to MATLAB's TolX (1e-4) only.
  Psi then agrees to ~1% (the tolerance is on the log-variance scale).
- Where a random-effects variance collapses to zero during the iteration
  (pynlme warns, or logs at INFO if it recovers), the likelihood is flat along
  it and rounding decides where both stop, and whether either finds its way
  off the plateau: MATLAB's own result changes with the number of BLAS threads
  (seed 1001 logistic_7 RELME: -308.544 with one thread, -310.691 with four;
  over 3200 fits pynlme ended higher or lower by up to 2.1 in log-likelihood).
  Those fits are only checked for gross errors.
"""

import json
import logging
import os
import secrets
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from pynlme import nlmefit

from ..conftest import matlab_command
from .simulate import FAMILIES, simulate, write_for_r

pytestmark = pytest.mark.skipif(
    matlab_command() is None, reason="set PYNLME_MATLAB to run against MATLAB"
)

SEED = int(os.environ.get("PYNLME_RANDOM_SEED", secrets.randbelow(2**31)))
N_PER_FAMILY = int(os.environ.get("PYNLME_RANDOM_N", 5))
NAMES = [f"{family}_{k}" for family in FAMILIES for k in range(N_PER_FAMILY)]
SCRIPT_DIR = Path(__file__).resolve().parents[2] / "validation" / "matlab"


def _run_matlab(manifest: Path, results: Path, approx: str) -> None:
    command = matlab_command()
    env = dict(os.environ)
    while command and "=" in command[0]:
        key, value = command.pop(0).split("=", 1)
        env[key] = value
    call = (
        f"addpath('{SCRIPT_DIR}'); fit_manifest('{manifest}', '{results}', '{approx}')"
    )
    # Output to a file, not a pipe: MATLAB starts background helpers (MathWorks
    # ServiceHost, CrashReporter) that inherit stdout/stderr and keep a pipe open
    # after MATLAB exits, so reading it to the end would never return.
    with open(results.with_suffix(".log"), "w") as log:
        subprocess.run(
            [*command, "-batch", call],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=1800,
        )


@pytest.fixture(scope="module", params=["LME", "RELME"])
def runs(request, tmp_path_factory):
    approx = request.param
    rng = np.random.default_rng(SEED)
    datasets = [simulate(name.rsplit("_", 1)[0], rng, name=name) for name in NAMES]
    directory = tmp_path_factory.mktemp(f"random_vs_matlab_{approx}")
    results = directory / "matlab.json"
    _run_matlab(write_for_r(datasets, directory), results, approx)
    return approx, {d.name: d for d in datasets}, json.loads(results.read_text())


@pytest.mark.parametrize("name", NAMES)
def test_matches_matlab(runs, name, caplog):
    approx, datasets, results = runs
    d, r = datasets[name], results[name]
    context = f"seed={SEED} dataset={name} {approx}"
    with (
        warnings.catch_warnings(record=True) as caught,
        caplog.at_level(logging.INFO, logger="pynlme"),
    ):
        warnings.simplefilter("always")
        try:
            beta, psi, stats, _ = nlmefit(
                *d.fit_args(), **d.kwargs, ApproximationType=approx
            )
        except Exception as error:
            if isinstance(error, FloatingPointError) and "error" in r:
                # MATLAB failed too ("MYFUN has returned Inf or NaN values"): the
                # model is not finite where the iteration leads.
                return
            raise AssertionError(f"{context}: {error!r}") from error  # with the seed
    collapsed = any(
        "variance was estimated as zero" in str(w.message) for w in caught
    ) or any(
        "variance was zero during the iterations" in rec.getMessage()
        for rec in caplog.records
    )
    assert np.all(np.isfinite(beta)) and np.all(np.isfinite(psi)), context
    if "error" in r:
        return  # MATLAB failed; pynlme only needs finite estimates.
    if collapsed:
        assert stats.logl >= r["logl"] - 2.0, (
            f"{context}: logl {stats.logl} vs {r['logl']} (flat path)"
        )
        return
    ref_psi = np.atleast_2d(r["psi"])
    assert stats.logl == pytest.approx(r["logl"], abs=1e-3), (
        f"{context}: logl {stats.logl} vs {r['logl']}"
    )
    scale = np.maximum(np.abs(r["beta"]), 1.0)
    assert np.all(np.abs(beta - r["beta"]) <= 1e-4 * scale), (
        f"{context}: beta={beta} MATLAB={r['beta']}"
    )
    np.testing.assert_allclose(
        psi, ref_psi, rtol=1e-2, atol=1e-3 * np.max(np.diag(ref_psi)), err_msg=context
    )
    # Record how often the fit is MATLAB's to rounding (see the module docstring).
    print(
        f"{context}: max |dbeta|/scale = {np.max(np.abs(beta - r['beta']) / scale):.1e}"
    )
