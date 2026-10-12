"""
nlmefit against MATLAB R2023b nlmefit (validation/matlab/run_references.m).

The references use tight tolerances (statset TolX = TolFun = 1e-8), so both
implementations are converged and agree to ~1e-6; the tolerances below leave a
margin of about 100x. All outputs are compared, including MATLAB's reporting
conventions (rmse = sqrt(sum(ires^2)/dfe); for RELME the error variance as
RSS/N and BIC with log(m - f)).
"""

import numpy as np
import pytest

from pynlme import nlmefit

from .cases import indometh, linear_sim, load_reference, orange, theoph, theoph_wt

# statset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxIter', 1000) of run_references.m
TIGHT = {"Options": {"TolX": 1e-8, "TolFun": 1e-8, "MaxIter": 1000}}

CASES = {
    # The MATLAB documentation start values for indomethacin.
    "indometh": lambda: indometh(beta0=(1, 1, 1, 1)),
    "theoph": theoph,
    "theoph_wt": theoph_wt,
    "orange": orange,
    "linear": linear_sim,
}
PARAMS = [(name, approx) for name in CASES for approx in ("lme", "relme")]


@pytest.fixture(scope="module", params=PARAMS, ids=[f"{n}-{a}" for n, a in PARAMS])
def fit(request):
    name, approx = request.param
    ref = load_reference(f"matlab_{name}_{approx}_tight.json")
    case = CASES[name]()
    beta, psi, stats, b = nlmefit(
        case.X,
        case.y,
        case.group,
        case.V,
        case.modelfun,
        case.beta0,
        **case.kwargs,
        ApproximationType=approx.upper(),
        **TIGHT,
    )
    if name == "indometh" and abs(beta[0] - ref["beta"][0]) > 0.5:
        # Symmetric start: the two exponential terms may be labelled the other way.
        order = [2, 3, 0, 1]
        beta, psi, b = beta[order], psi[np.ix_(order, order)], b[order]
        stats.sebeta = stats.sebeta[order]
    return ref, beta, psi, stats, b


@pytest.mark.parametrize("name, approx", PARAMS, ids=[f"{n}-{a}" for n, a in PARAMS])
def test_identical_to_matlab_at_its_default_options(name, approx):
    """compat="matlab" follows nlmefit's iteration step by step (matlab_lb.py),
    so with MATLAB's default statset the result is MATLAB's own, not just the
    same optimum: estimates agree to ~1e-10 where MATLAB stops at TolX = 1e-4."""
    ref = load_reference(f"matlab_{name}_{approx}_default.json")
    case = CASES[name]()
    beta, psi, stats, b = nlmefit(
        case.X,
        case.y,
        case.group,
        case.V,
        case.modelfun,
        case.beta0,
        **case.kwargs,
        ApproximationType=approx.upper(),
    )
    ref_beta, ref_psi = np.array(ref["beta"]), np.atleast_2d(ref["psi"])
    if name == "indometh" and abs(beta[0] - ref_beta[0]) > 0.5:
        # Symmetric start: rounding decides which exponential term comes first.
        order = [2, 3, 0, 1]
        beta, psi = beta[order], psi[np.ix_(order, order)]
    tol = 1e-6 if name == "indometh" else 1e-8
    np.testing.assert_allclose(beta, ref_beta, rtol=tol, atol=tol)
    np.testing.assert_allclose(
        psi, ref_psi, rtol=tol, atol=tol * np.max(np.diag(ref_psi))
    )
    assert stats.logl == pytest.approx(ref["stats"]["logl"], abs=1e-6)


def test_estimates(fit):
    ref, beta, psi, _, b = fit
    ref_psi = np.atleast_2d(ref["psi"])
    np.testing.assert_allclose(beta, ref["beta"], rtol=1e-4)
    np.testing.assert_allclose(
        psi, ref_psi, rtol=1e-3, atol=1e-4 * np.max(np.diag(ref_psi))
    )
    ref_b = np.atleast_2d(ref["b"])
    np.testing.assert_allclose(b, ref_b, atol=1e-4 * np.max(np.abs(ref_b)))


def test_stats(fit):
    ref, _, _, stats, _ = fit
    s = ref["stats"]
    assert stats.dfe == s["dfe"]
    for field in ("logl", "aic", "bic"):
        assert getattr(stats, field) == pytest.approx(s[field], abs=2e-3), field
    np.testing.assert_allclose(stats.rmse, s["rmse"], rtol=1e-4)
    np.testing.assert_allclose(stats.errorparam, np.ravel(s["errorparam"]), rtol=1e-4)
    np.testing.assert_allclose(stats.sebeta, s["sebeta"], rtol=1e-3)
    np.testing.assert_allclose(
        stats.ires, s["ires"], atol=1e-4 * np.max(np.abs(s["ires"]))
    )
    np.testing.assert_allclose(
        stats.pres, s["pres"], atol=1e-4 * np.max(np.abs(s["pres"]))
    )
    for field in ("iwres", "pwres", "cwres"):
        np.testing.assert_allclose(
            getattr(stats, field), s[field], atol=1e-3, err_msg=field
        )


FIRST_ORDER = [(name, approx) for name in CASES for approx in ("fo", "foce")]


@pytest.fixture(scope="module")
def first_order_fits():
    fits = {}
    for name, approx in FIRST_ORDER:
        case = CASES[name]()
        fits[name, approx] = nlmefit(
            case.X,
            case.y,
            case.group,
            case.V,
            case.modelfun,
            case.beta0,
            **case.kwargs,
            ApproximationType=approx.upper(),
        )
    return fits


@pytest.mark.parametrize(
    "name, approx", FIRST_ORDER, ids=[f"{n}-{a}" for n, a in FIRST_ORDER]
)
def test_first_order_at_least_as_good_as_matlab(first_order_fits, name, approx):
    """pynlme must reach MATLAB's FO/FOCE log-likelihood or a higher one.

    MATLAB's FO collapses every covariance to zero on these datasets (and on the
    linear model runs for 1.5 hours without returning a log-likelihood), so for
    FO pynlme is typically far better. Where both reach the same optimum, the
    estimates must agree.
    """
    ref = load_reference(f"matlab_{name}_{approx}_tight.json")
    beta, psi, stats, _ = first_order_fits[name, approx]
    if ref["stats"]["logl"] is None:
        pytest.skip("MATLAB returned no log-likelihood")
    assert stats.logl >= ref["stats"]["logl"] - 1e-3
    if abs(stats.logl - ref["stats"]["logl"]) < 1e-3:
        if name == "indometh" and abs(beta[0] - ref["beta"][0]) > 0.5:
            order = [2, 3, 0, 1]
            beta, psi = beta[order], psi[np.ix_(order, order)]
            stats.sebeta = stats.sebeta[order]
        ref_psi = np.atleast_2d(ref["psi"])
        np.testing.assert_allclose(beta, ref["beta"], rtol=1e-4)
        np.testing.assert_allclose(
            psi, ref_psi, rtol=1e-3, atol=1e-4 * np.max(np.diag(ref_psi))
        )
        np.testing.assert_allclose(stats.rmse, ref["stats"]["rmse"], rtol=1e-4)
        # Standard errors are not compared: see test_matlab_first_order_standard_errors.


@pytest.mark.parametrize("name", ["orange", "indometh", "linear"])
def test_fo_equals_foce_when_random_effects_enter_linearly(first_order_fits, name):
    """Orange (asymptote), indomethacin (amplitudes; the rate variances are zero)
    and the linear model are linear in their random effects, so the
    linearisation is exact at any point and FO and FOCE coincide."""
    fo = first_order_fits[name, "fo"][2].logl
    foce = first_order_fits[name, "foce"][2].logl
    assert fo == pytest.approx(foce, abs=1e-4)


@pytest.mark.parametrize("name", list(CASES))
def test_matlab_first_order_standard_errors(name):
    """Documents a MATLAB quirk that pynlme does not copy.

    MATLAB's FO/FOCE standard errors come from the LME stage it runs first,
    rescaled by the final error SD (nlmefit.m computes covb from that stage's
    R00), rather than being evaluated at the FO/FOCE estimates. pynlme
    evaluates them at its own estimates, which differs by a few percent.
    """
    lme = load_reference(f"matlab_{name}_lme_tight.json")["stats"]
    foce = load_reference(f"matlab_{name}_foce_tight.json")["stats"]
    ratio = np.ravel(foce["errorparam"])[0] / np.ravel(lme["errorparam"])[0]
    np.testing.assert_allclose(
        foce["sebeta"], np.array(lme["sebeta"]) * ratio, rtol=1e-4
    )


ERROR_MODEL_CASES = [
    (name, approx, em)
    for name in ("theoph", "orange")
    for em in ("proportional", "combined", "exponential")
    for approx in ("lme", "relme")
]


def _positive(case):
    """MATLAB's error-model references drop non-positive responses and time 0."""
    keep = (case.y > 0) & (np.ravel(case.X) > 0)
    case.X, case.y, case.group = case.X[keep], case.y[keep], case.group[keep]
    return case


@pytest.mark.parametrize(
    "name, approx, error_model",
    ERROR_MODEL_CASES,
    ids=["-".join(c) for c in ERROR_MODEL_CASES],
)
def test_error_models(name, approx, error_model):
    """Proportional, combined and exponential error models against MATLAB.

    Observations at time 0 are dropped: the oral model predicts exactly 0
    there while some observed concentrations are positive, which makes the
    proportional and exponential models ill-posed (MATLAB then returns an
    error SD of ~1e7 or ~110; pynlme reproduces those degenerate fits too).
    """
    ref = load_reference(f"matlab_{name}_{approx}_{error_model}_tight.json")
    case = _positive(CASES[name]())
    beta, psi, stats, b = nlmefit(
        case.X,
        case.y,
        case.group,
        case.V,
        case.modelfun,
        case.beta0,
        **case.kwargs,
        ApproximationType=approx.upper(),
        ErrorModel=error_model,
        **TIGHT,
    )
    s = ref["stats"]
    ref_psi = np.atleast_2d(ref["psi"])
    errorparam = np.ravel(s["errorparam"])
    np.testing.assert_allclose(beta, ref["beta"], rtol=1e-4)
    atol = max(1e-4 * np.max(np.diag(ref_psi)), 1e-8 * errorparam[0] ** 2)
    np.testing.assert_allclose(psi, ref_psi, rtol=1e-3, atol=atol)
    np.testing.assert_allclose(stats.errorparam, errorparam, rtol=1e-3)
    assert stats.logl == pytest.approx(s["logl"], abs=2e-3)
    assert stats.dfe == s["dfe"]
    np.testing.assert_allclose(stats.rmse, s["rmse"], rtol=1e-4)
    np.testing.assert_allclose(stats.sebeta, s["sebeta"], rtol=1e-3)
    for field in ("iwres", "pwres", "cwres"):
        if len(s[field]) == 0:
            assert getattr(stats, field) is None, field
        else:
            np.testing.assert_allclose(
                getattr(stats, field), s[field], atol=1e-3, err_msg=field
            )


FOCE_ERROR_CASES = [
    (name, em)
    for name in ("theoph", "orange")
    for em in ("proportional", "combined", "exponential")
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, error_model", FOCE_ERROR_CASES, ids=["-".join(c) for c in FOCE_ERROR_CASES]
)
def test_foce_error_models_at_least_as_good_as_matlab(name, error_model):
    """FOCE with non-constant error models: MATLAB's log-likelihood or a higher
    one, and MATLAB's estimates where both reach the same optimum.

    MATLAB's FOCE log-likelihood for the exponential model leaves out the
    -sum(log y) term that its LME includes (FOCE +1.6 vs LME -182.6 on
    theophylline). The default compat="matlab" reproduces that; compat="r"
    reports the full likelihood (checked below).
    """
    from .cases import with_derivatives

    ref = load_reference(f"matlab_{name}_foce_{error_model}_tight.json")
    case = _positive(with_derivatives(CASES[name]()))
    beta, psi, stats, _ = nlmefit(
        case.X,
        case.y,
        case.group,
        case.V,
        case.modelfun,
        case.beta0,
        **case.kwargs,
        ApproximationType="FOCE",
        ErrorModel=error_model,
    )
    ref_logl = ref["stats"]["logl"]
    assert stats.logl >= ref_logl - 1e-3
    if abs(stats.logl - ref_logl) < 1e-3:
        np.testing.assert_allclose(beta, ref["beta"], rtol=1e-4)
        np.testing.assert_allclose(
            stats.errorparam, np.ravel(ref["stats"]["errorparam"]), rtol=1e-3
        )


def test_exponential_foce_loglikelihood_convention():
    """compat="r" adds the -sum(log y) term that MATLAB's FO/FOCE leave out."""
    case = _positive(CASES["orange"]())
    args = (case.X, case.y, case.group, case.V, case.modelfun, case.beta0)
    kw = dict(case.kwargs, ApproximationType="FO", ErrorModel="exponential")
    matlab = nlmefit(*args, **kw)[2].logl
    r = nlmefit(*args, **kw, compat="r")[2].logl
    assert r == pytest.approx(matlab - np.sum(np.log(case.y)), abs=1e-6)
