"""
nlmefitsa (SAEM) against exact answers, R saemix and MATLAB nlmefitsa.

SAEM is stochastic, so comparisons are statistical. For the linear model the
exact maximum-likelihood estimates are known (nlmefit's LME is exact there),
and SAEM must reach them within its Monte Carlo error. Against saemix (10
seeds) and MATLAB (5 seeds) the mean estimates over several seeds must agree
within 4 standard errors of the difference.
"""

import numpy as np
import pytest

from pynlme import nlmefit, nlmefitsa

from .cases import linear_sim, load_reference, orange, theoph


@pytest.fixture(scope="module")
def linear():
    c = linear_sim()
    ml = nlmefit(c.X, c.y, c.group, None, c.modelfun, c.beta0, **c.kwargs)
    return c, ml


def test_linear_model_reaches_exact_ml(linear):
    c, (beta_ml, psi_ml, stats_ml, _) = linear
    runs = [nlmefitsa(c.X, c.y, c.group, None, c.modelfun, c.beta0, **c.kwargs, random_state=s) for s in range(3)]
    beta = np.mean([r[0] for r in runs], axis=0)
    psi = np.mean([r[1] for r in runs], axis=0)
    sigma = np.mean([r[2].errorparam[0] for r in runs])
    # Monte Carlo error of SAEM is a small fraction of the statistical error.
    assert np.all(np.abs(beta - beta_ml) < 0.2 * stats_ml.sebeta)
    np.testing.assert_allclose(np.diag(psi), np.diag(psi_ml), rtol=0.1)
    assert sigma == pytest.approx(stats_ml.errorparam[0], rel=0.02)


def test_loglikelihood_and_standard_errors(linear):
    c, (_, _, stats_ml, _) = linear
    out = {}
    for method in ("is", "lin"):
        _, _, stats, _ = nlmefitsa(
            c.X, c.y, c.group, None, c.modelfun, c.beta0, **c.kwargs,
            random_state=1, LogLikMethod=method, ComputeStdErrors=True,
        )
        out[method] = stats
    # The linearisation is exact for a linear model, so importance sampling
    # must reproduce it within its Monte Carlo error (SD ~0.01); the exact
    # value is at most slightly below the maximum.
    assert out["is"].logl == pytest.approx(out["lin"].logl, abs=0.05)
    assert stats_ml.logl - 0.1 < out["lin"].logl <= stats_ml.logl + 1e-6
    np.testing.assert_allclose(out["lin"].sebeta, stats_ml.sebeta, rtol=0.1)


def test_reproducible_with_seed(linear):
    c, _ = linear
    a = nlmefitsa(c.X, c.y, c.group, None, c.modelfun, c.beta0, **c.kwargs, random_state=7)
    b = nlmefitsa(c.X, c.y, c.group, None, c.modelfun, c.beta0, **c.kwargs, random_state=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_outputs_follow_matlab():
    c = theoph()
    beta, psi, stats, b = nlmefitsa(c.X, c.y, c.group, c.V, c.modelfun, c.beta0, **c.kwargs, random_state=1, n_chains=1)
    assert beta.shape == (3,) and psi.shape == (2, 2) and b.shape == (2, 12)
    assert stats.logl is None and stats.aic is None and stats.sebeta is None  # MATLAB defaults
    assert stats.dfe == len(c.y) - 6


@pytest.mark.parametrize("option", [{"FEObsDesign": np.zeros((3, 3, 35))}, {"LogLikMethod": "gq"}])
def test_unimplemented_options_raise(option):
    c = orange()
    with pytest.raises(NotImplementedError):
        nlmefitsa(c.X, c.y, c.group, None, c.modelfun, c.beta0, **c.kwargs, **option, NIterations=[2, 2, 2])


@pytest.mark.slow
@pytest.mark.parametrize("name, factory", [("linear", linear_sim), ("orange", orange), ("theoph", theoph)])
def test_agrees_with_saemix_and_matlab(name, factory):
    c = factory()
    runs = [nlmefitsa(c.X, c.y, c.group, c.V, c.modelfun, c.beta0, **c.kwargs, random_state=s) for s in range(5)]
    ours = np.array([r[0] for r in runs])
    mean, var = ours.mean(axis=0), ours.var(axis=0, ddof=1) / len(ours)

    saemix = load_reference(f"r_saemix_{name}.json")
    se = np.sqrt(var + np.array(saemix["beta_sd"]) ** 2 / saemix["n_seeds"])
    assert np.all(np.abs(mean - saemix["beta_mean"]) < 4 * se), "saemix"

    matlab = np.array([load_reference(f"matlab_{name}_saem_seed{s}.json")["beta"] for s in range(1, 6)])
    se = np.sqrt(var + matlab.var(axis=0, ddof=1) / len(matlab))
    assert np.all(np.abs(mean - matlab.mean(axis=0)) < 4 * se), "MATLAB"


SAEM_ERROR_CASES = [(n, e) for n in ("theoph", "orange") for e in ("proportional", "combined", "exponential")]


@pytest.mark.slow
@pytest.mark.parametrize("name, error_model", SAEM_ERROR_CASES, ids=["-".join(c) for c in SAEM_ERROR_CASES])
def test_error_models_agree_with_matlab(name, error_model):
    """Mean estimates over 3 seeds within 4 standard errors of MATLAB's (3 seeds).

    As MATLAB's references, time 0 and non-positive responses are dropped.
    """
    from .cases import vectorized

    case = vectorized({"theoph": theoph, "orange": orange}[name]())
    keep = (case.y > 0) & (np.ravel(case.X) > 0)
    case.X, case.y, case.group = case.X[keep], case.y[keep], case.group[keep]
    runs = [
        nlmefitsa(case.X, case.y, case.group, case.V, case.modelfun, case.beta0, **case.kwargs,
                  ErrorModel=error_model, random_state=s)
        for s in range(3)
    ]
    ours = np.array([r[0] for r in runs])
    matlab = np.array([load_reference(f"matlab_{name}_saem_{error_model}_seed{s}.json")["beta"] for s in range(1, 4)])
    se = np.sqrt(ours.var(axis=0, ddof=1) / 3 + matlab.var(axis=0, ddof=1) / 3)
    assert np.all(np.abs(ours.mean(axis=0) - matlab.mean(axis=0)) < 4 * se)
