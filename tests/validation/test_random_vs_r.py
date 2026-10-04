"""
nlmefit against R nlme on freshly simulated datasets.

Each run draws new true parameters, group counts and sampling times, so the
implementation cannot be tuned to fixed cases. The seed is printed in failure
messages; set PYNLME_RANDOM_SEED to reproduce a run and PYNLME_RANDOM_N to
change the number of datasets per family.

A dataset passes if pynlme's maximised log-likelihood is at least as high as
R's, to within 0.01. Where both reach the same optimum the estimates agree to
~1e-6. The Lindstrom-Bates iteration can have several fixed points; where the
two implementations end at different ones, differences below 0.01 in the
log-likelihood (0.02 in deviance) are of no practical consequence, and larger
differences have so far all been in pynlme's favour. pynlme must converge
whenever R does, unless it found a better log-likelihood than R.

RefineBeta0 is switched off because R has no such step; comparing like with
like. (Over 640 datasets, refining never gave a better fit by more than 0.01
and gave a worse one in 3 cases; it stays on by default for MATLAB parity.)
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

pytestmark = pytest.mark.skipif(not r_available(), reason="needs Rscript with nlme and jsonlite")

SEED = int(os.environ.get("PYNLME_RANDOM_SEED", secrets.randbelow(2**31)))
N_PER_FAMILY = int(os.environ.get("PYNLME_RANDOM_N", 5))
NAMES = [f"{family}_{k}" for family in FAMILIES for k in range(N_PER_FAMILY)]
R_SCRIPT = Path(__file__).resolve().parents[2] / "validation" / "r" / "fit_manifest.R"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    rng = np.random.default_rng(SEED)
    datasets = [simulate(name.rsplit("_", 1)[0], rng, name=name) for name in NAMES]
    directory = tmp_path_factory.mktemp("random_vs_r")
    manifest = write_for_r(datasets, directory)
    results = directory / "r_results.json"
    subprocess.run(["Rscript", str(R_SCRIPT), str(manifest), str(results)], check=True, capture_output=True)
    return {d.name: d for d in datasets}, json.loads(results.read_text())


@pytest.mark.parametrize("name", NAMES)
def test_matches_or_beats_r(runs, name):
    datasets, r_results = runs
    d, r = datasets[name], r_results[name]
    context = f"seed={SEED} dataset={name}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beta, psi, stats, _ = nlmefit(*d.fit_args(), **d.kwargs, RefineBeta0="off")
    converged = not any("did not converge" in str(w.message) for w in caught)

    assert np.all(np.isfinite(beta)) and np.all(np.isfinite(psi)), context
    assert np.all(np.linalg.eigvalsh(psi) >= -1e-10 * np.max(np.diag(psi))), context

    if "error" in r:
        return  # R failed; pynlme only needs to return finite estimates.

    assert converged or stats.logl > r["logl"], f"{context}: R converged but pynlme did not"
    assert stats.logl >= r["logl"] - 0.01, (
        f"{context}: pynlme logl {stats.logl:.6f} below R {r['logl']:.6f}; "
        f"beta={beta} R={r['beta']}"
    )
