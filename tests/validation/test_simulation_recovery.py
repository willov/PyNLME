"""
Simulation studies: fit many datasets simulated from known parameters.

For the linear model the LME approximation is exact, so ML theory applies:
fixed effects are unbiased and Wald intervals have close to nominal coverage.
These are checked against Monte Carlo error.

For nonlinear models the linearisation biases the estimates somewhat (R nlme,
which pynlme matches to ~1e-6, has the same bias). In a 200-replicate study
with 30 groups the largest bias was -13% for log(ka) in the oral model, with
90% coverage; others were within 1%. The checks for those models only catch
gross errors; agreement with R is tested in test_random_vs_r.py.

Run with: pytest tests/validation/test_simulation_recovery.py --runslow
"""

import warnings

import numpy as np
import pytest

from pynlme import nlmefit, nlmefitsa

from .simulate import Truth, simulate

pytestmark = pytest.mark.slow

N_GROUPS = 30

TRUTHS = {
    "linear": Truth(np.array([1.0, 0.5]), np.array([[0.25, 0.03], [0.03, 0.04]]), 0.4),
    "oral": Truth(np.log([0.1, 1.5, 0.04]), np.diag([0.3**2, 0.25**2]), 0.5),
    "logistic": Truth(
        np.array([200.0, 700.0, 300.0]), np.diag([20.0**2, 40.0**2]), 6.0
    ),
    "emax": Truth(np.array([10.0, 50.0, np.log(8.0)]), np.diag([2.0**2, 0.25**2]), 2.0),
}


def run_study(family: str, replicates: int, seed: int = 1):
    truth = TRUTHS[family]
    rng = np.random.default_rng(seed)
    beta, psi, sigma, covered = [], [], [], []
    for _ in range(replicates):
        d = simulate(family, rng, m=N_GROUPS, truth=truth)
        with warnings.catch_warnings():
            # Non-convergence fails; MATLAB's informational PNLS warnings do not.
            warnings.filterwarnings(
                "error", message="nlmefit did not converge", category=RuntimeWarning
            )
            b, p, stats, _ = nlmefit(*d.fit_args(), **d.kwargs)
        beta.append(b)
        psi.append(p)
        sigma.append(stats.errorparam[0])
        covered.append(np.abs(b - truth.beta) <= 1.96 * stats.sebeta)
    return truth, np.array(beta), np.array(psi), np.array(sigma), np.array(covered)


def test_linear_model_matches_ml_theory():
    replicates = 200
    truth, beta, psi, sigma, covered = run_study("linear", replicates)

    mcse = beta.std(axis=0, ddof=1) / np.sqrt(replicates)
    assert np.all(np.abs(beta.mean(axis=0) - truth.beta) < 3 * mcse)

    # Binomial Monte Carlo error on 95% coverage is ~1.5% with 200 replicates.
    binom = 3 * np.sqrt(0.95 * 0.05 / replicates)
    assert np.all(np.abs(covered.mean(axis=0) - 0.95) < binom)

    # ML variance estimates are biased downwards by O(1/m); sigma is not.
    ratio = np.diag(psi.mean(axis=0)) / np.diag(truth.psi)
    assert np.all((ratio > 0.8) & (ratio < 1.05))
    assert abs(sigma.mean() / truth.sigma - 1) < 0.02


@pytest.mark.parametrize("family", ["oral", "logistic", "emax"])
def test_nonlinear_models_have_no_gross_errors(family):
    truth, beta, psi, sigma, covered = run_study(family, replicates=100)

    rel_bias = np.abs(beta.mean(axis=0) - truth.beta) / np.abs(truth.beta)
    assert np.all(rel_bias < 0.2)
    assert np.all(covered.mean(axis=0) > 0.85)
    ratio = np.diag(psi.mean(axis=0)) / np.diag(truth.psi)
    assert np.all((ratio > 0.7) & (ratio < 1.3))
    assert abs(sigma.mean() / truth.sigma - 1) < 0.05


def test_saem_removes_linearisation_bias():
    """SAEM maximises the exact likelihood, so the LME bias in log(ka) goes away.

    In a 50-replicate study LME's relative bias in log(ka) was -15% and
    SAEM's +1.4%, with similar root-mean-square errors.
    """
    truth = TRUTHS["oral"]
    rng = np.random.default_rng(1)
    estimates = []
    for k in range(40):
        d = simulate("oral", rng, m=N_GROUPS, truth=truth)
        beta, _, _, _ = nlmefitsa(*d.fit_args(), **d.kwargs, random_state=k)
        estimates.append(beta)
    rel_bias = (np.mean(estimates, axis=0) - truth.beta) / np.abs(truth.beta)
    assert np.all(np.abs(rel_bias) < 0.05)
