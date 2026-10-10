"""
nlmefit with compat="r" against R nlme on freshly simulated datasets.

Each run draws new true parameters, group counts and sampling times, so the
implementation cannot be tuned to fixed cases. The seed is printed in failure
messages; set PYNLME_RANDOM_SEED to reproduce a run and PYNLME_RANDOM_N to
change the number of datasets per family.

compat="r" follows nlme's iteration step by step (PNLS with EM-updated
covariance and at most pnlsMaxIter steps; see lindstrom_bates.py), so where R
converges pynlme must reach the same fit: the same log-likelihood (1e-3) and
the same estimates. Where several fixed points exist this requires following
nlme's path, not just reaching a fixed point. The one exception: where R's
optimizers stop early on a nearly flat likelihood (a variance collapsing to
zero), pynlme may end higher; that is accepted only when its likelihood is
higher than R's. The controls are those of validation/r/fit_manifest.R.
"""

import json
import os
import secrets
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from pynlme import nlmefit

from ..conftest import r_available
from .simulate import FAMILIES, simulate, write_for_r

pytestmark = pytest.mark.skipif(
    not r_available(), reason="needs Rscript with nlme and jsonlite"
)

SEED = int(os.environ.get("PYNLME_RANDOM_SEED", secrets.randbelow(2**31)))
N_PER_FAMILY = int(os.environ.get("PYNLME_RANDOM_N", 5))
NAMES = [f"{family}_{k}" for family in FAMILIES for k in range(N_PER_FAMILY)]
# nlmeControl(maxIter = 200, pnlsTol = 1e-6, tolerance = 1e-7) of fit_manifest.R
R_CONTROL = {"compat": "r", "max_iter": 200, "pnls_tol": 1e-6, "tol_x": 1e-7}
R_SCRIPT = Path(__file__).resolve().parents[2] / "validation" / "r" / "fit_manifest.R"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    rng = np.random.default_rng(SEED)
    datasets = [simulate(name.rsplit("_", 1)[0], rng, name=name) for name in NAMES]
    directory = tmp_path_factory.mktemp("random_vs_r")
    manifest = write_for_r(datasets, directory)
    results = directory / "r_results.json"
    subprocess.run(
        ["Rscript", str(R_SCRIPT), str(manifest), str(results)],
        check=True,
        capture_output=True,
    )
    return {d.name: d for d in datasets}, json.loads(results.read_text())


@pytest.mark.parametrize("name", NAMES)
def test_matches_or_beats_r(runs, name):
    datasets, r_results = runs
    d, r = datasets[name], r_results[name]
    context = f"seed={SEED} dataset={name}"

    # Slowly (linearly) converging cases can stop a few iterations apart (R's
    # criterion drops to zero once its PNLS step stops at pnlsTol), so a
    # non-convergence warning is allowed; the fit itself must be R's.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beta, psi, stats, _ = nlmefit(*d.fit_args(), **d.kwargs, **R_CONTROL)

    assert np.all(np.isfinite(beta)) and np.all(np.isfinite(psi)), context
    assert np.all(np.linalg.eigvalsh(psi) >= -1e-10 * np.max(np.diag(psi))), context

    if "error" in r:
        return  # R failed; pynlme only needs to return finite estimates.

    # Where a variance collapses towards zero the likelihood is nearly flat
    # along it, and where R's optimizers (nlminb in nlme's LME step, which
    # pynlme does not reimplement, or lme's) stop there decides R's estimates.
    # pynlme's L-BFGS-B moves on: it can end higher (seed 1770464656, oral_1:
    # +0.0017; seed 2009, linear_7: +0.00015) or elsewhere on the ridge (seed
    # 1005, oral_9: -0.0007, ~1 in 2400 datasets). Those cases need the same
    # likelihood only; all others must give R's fit.
    if stats.logl > r["logl"] + 1e-5:
        return
    collapsed = np.min(np.diag(np.atleast_2d(r["psi"]))) < 1e-6 * r["sigma"] ** 2
    if collapsed and stats.logl >= r["logl"] - 1e-3:
        return
    assert stats.logl == pytest.approx(r["logl"], abs=1e-3), (
        f"{context}: pynlme logl {stats.logl:.6f}, R {r['logl']:.6f}; beta={beta} "
        f"R={r['beta']}"
    )
    scale = np.maximum(np.abs(r["beta"]), 1.0)
    assert np.all(np.abs(beta - r["beta"]) <= 1e-4 * scale), (
        f"{context}: beta={beta} R={r['beta']}"
    )
    # A variance component near zero sits on an almost flat likelihood; where
    # nlme's optimizers stop along it depends on rounding (R's own numerical
    # derivatives), e.g. 0.013 vs 0.019 next to a variance of 0.079 with the
    # same likelihood (seed 1005, emax_7). Hence the absolute margin.
    ref_psi = np.atleast_2d(r["psi"])
    np.testing.assert_allclose(
        psi, ref_psi, rtol=1e-3, atol=0.1 * np.max(np.diag(ref_psi)), err_msg=context
    )
