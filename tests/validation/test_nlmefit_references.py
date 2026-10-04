"""
nlmefit against reference results from R nlme and the MATLAB documentation.

R nlme runs the same Lindstrom-Bates algorithm, and both sides converge tightly
(nlme pnlsTol 1e-6; pynlme tol_x 1e-6), so estimates agree to about 1e-6
relative. The tolerances below leave roughly a 100x margin over that.
"""

import numpy as np
import pytest

from pynlme import nlmefit

from .cases import R_NLME_CASES, indometh, load_reference

RTOL = 1e-4


@pytest.fixture(scope="module", params=list(R_NLME_CASES))
def fit(request):
    factory, ref_file, extra = R_NLME_CASES[request.param]
    case = factory()
    # compat="r": R's reporting conventions, so outputs compare directly.
    beta, psi, stats, b = nlmefit(
        case.X, case.y, case.group, case.V, case.modelfun, case.beta0,
        **case.kwargs, **extra, compat="r",
    )
    return case, load_reference(ref_file), beta, psi, stats, b


def test_fixed_effects(fit):
    _, ref, beta, *_ = fit
    np.testing.assert_allclose(beta, ref["beta"], rtol=RTOL)


def test_random_effects_covariance(fit):
    _, ref, _, psi, _, _ = fit
    ref_psi = np.atleast_2d(ref["psi"])
    # Absolute tolerance relative to the largest variance, so variance
    # components estimated at zero are compared on a sensible scale.
    np.testing.assert_allclose(psi, ref_psi, rtol=RTOL, atol=RTOL * np.max(np.diag(ref_psi)))


def test_error_sd_and_loglikelihood(fit):
    _, ref, _, _, stats, _ = fit
    np.testing.assert_allclose(stats.errorparam, [ref["sigma"]], rtol=RTOL)
    assert stats.logl == pytest.approx(ref["logl"], abs=1e-3)


def test_standard_errors(fit):
    _, ref, _, _, stats, _ = fit
    np.testing.assert_allclose(stats.sebeta, ref["sebeta"], rtol=1e-3)


def test_bic_follows_r(fit):
    """R's BIC uses the number of observations (N - f for REML)."""
    case, ref, beta, psi, stats, _ = fit
    n = len(case.y) - (len(beta) if ref["method"] == "REML" else 0)
    num_param = stats.dfe and len(case.y) - stats.dfe
    assert stats.bic == pytest.approx(-2 * ref["logl"] + np.log(n) * num_param, abs=2e-3)


def test_random_effects(fit):
    _, ref, _, _, _, b = fit
    ref_b = np.atleast_2d(ref["b"])
    assert b.shape == ref_b.shape
    np.testing.assert_allclose(b, ref_b, atol=RTOL * np.max(np.abs(ref_b)))


def test_matlab_doc_indomethacin():
    """MATLAB nlmefit doc example, from its own start values [1 1 1 1].

    That start is symmetric in the two exponential terms, so which term ends up
    as the fast one depends on rounding. The fit is accepted with the terms in
    either order; the published values are printed to 4 decimals.
    """
    doc = load_reference("matlab_docs.json")["cases"]["indometh_lme"]
    case = indometh(beta0=(1, 1, 1, 1))
    beta, psi, _, _ = nlmefit(case.X, case.y, case.group, None, case.modelfun, case.beta0, **case.kwargs)
    order = [0, 1, 2, 3] if beta[0] < beta[2] else [2, 3, 0, 1]
    half_ulp = 0.5 * 10 ** -doc["decimals"] + 1e-9
    np.testing.assert_allclose(beta[order], doc["beta"], rtol=0, atol=half_ulp)
    np.testing.assert_allclose(np.diag(psi)[order], doc["psi_diag"], rtol=0, atol=half_ulp)
