"""Analytic model derivatives (pynlme.with_jacobian) instead of finite differences."""

import numpy as np
import pytest

from pynlme import nlmefit, nlmefitsa
from pynlme.data_types import NLMEOptions
from pynlme.nlmefit import _build_model, _parse_options
from pynlme.utils import validate_inputs

from .validation.cases import (
    indometh,
    linear_sim,
    load_reference,
    orange,
    theoph,
    vectorized,
    with_derivatives,
)

FACTORIES = {
    "indometh": indometh,
    "theoph": theoph,
    "orange": orange,
    "linear": linear_sim,
}


# Converged tightly, so differences in the derivatives do not show.
TIGHT = {"Options": {"TolX": 1e-8, "TolFun": 1e-8, "MaxIter": 1000}}


def grouped_model(case):
    parsed, matlab_style = _parse_options(case.kwargs)
    options = NLMEOptions(**parsed)
    X, y, g, V, b0 = validate_inputs(case.X, case.y, case.group, case.V, case.beta0)
    return _build_model(X, y, g, V, case.modelfun, b0, options, matlab_style)


@pytest.mark.parametrize("name", list(FACTORIES))
def test_analytic_jacobian_matches_finite_differences(name):
    """Checks the test models' derivatives and the ParamTransform chain rule."""
    analytic = grouped_model(with_derivatives(FACTORIES[name]()))
    numeric = grouped_model(vectorized(FACTORIES[name]()))
    rng = np.random.default_rng(0)
    beta = np.asarray(FACTORIES[name]().beta0, dtype=float)
    etas = np.array([analytic.eta(i, beta, None) for i in range(analytic.n_groups)])
    etas = etas + rng.normal(0, 0.05, etas.shape) * np.maximum(np.abs(etas), 1)
    np.testing.assert_allclose(
        analytic.jacobian_padded(etas),
        numeric.jacobian_padded(etas),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("approx", ["LME", "RELME", "FOCE"])
@pytest.mark.parametrize("full", [True, False], ids=["Full", "SinglePhi"])
def test_fits_agree_with_finite_difference_fits(approx, full):
    numeric = vectorized(theoph())
    analytic = with_derivatives(theoph(), full=full)
    a = nlmefit(
        numeric.X,
        numeric.y,
        numeric.group,
        numeric.V,
        numeric.modelfun,
        numeric.beta0,
        **numeric.kwargs,
        ApproximationType=approx,
        **TIGHT,
    )
    b = nlmefit(
        analytic.X,
        analytic.y,
        analytic.group,
        analytic.V,
        analytic.modelfun,
        analytic.beta0,
        **analytic.kwargs,
        ApproximationType=approx,
        **TIGHT,
    )
    # FOCE maximises with L-BFGS-B, whose stopping point moves by ~1e-5 with the
    # derivatives used (and between SciPy versions); LME/RELME agree to ~1e-7.
    np.testing.assert_allclose(b[0], a[0], rtol=1e-4 if approx == "FOCE" else 1e-5)
    np.testing.assert_allclose(np.diag(b[1]), np.diag(a[1]), rtol=1e-4)
    # compat="matlab" uses MATLAB's forward differences (step eps^(1/3)),
    # which move the linearised likelihood by ~1e-4 against exact derivatives.
    assert b[2].logl == pytest.approx(a[2].logl, abs=5e-4)


def test_still_matches_matlab_with_analytic_derivatives():
    c = with_derivatives(theoph())
    beta, psi, stats, _ = nlmefit(
        c.X, c.y, c.group, c.V, c.modelfun, c.beta0, **c.kwargs, **TIGHT
    )
    ref = load_reference("matlab_theoph_lme_tight.json")
    np.testing.assert_allclose(beta, ref["beta"], rtol=1e-5)
    np.testing.assert_allclose(stats.sebeta, ref["stats"]["sebeta"], rtol=1e-4)


def test_saem_with_analytic_derivatives():
    """Theophylline has a parameter without random effects (log ke), whose
    SAEM update uses the derivatives."""
    numeric = vectorized(theoph())
    analytic = with_derivatives(theoph())
    a = nlmefitsa(
        numeric.X,
        numeric.y,
        numeric.group,
        numeric.V,
        numeric.modelfun,
        numeric.beta0,
        **numeric.kwargs,
        random_state=2,
    )
    b = nlmefitsa(
        analytic.X,
        analytic.y,
        analytic.group,
        analytic.V,
        analytic.modelfun,
        analytic.beta0,
        **analytic.kwargs,
        random_state=2,
    )
    np.testing.assert_allclose(b[0], a[0], atol=2e-3)


@pytest.mark.parametrize("method", ["LSODA", "BDF", "Radau"])
def test_ode_model_implicit_methods_match_dop853(method):
    """ode_model's implicit methods (given the systems' block-diagonal Jacobian)
    give the explicit method's solution."""
    from pynlme import ode_model

    def rhs(t, y, phi, v):
        return np.column_stack(
            [-phi[:, 0] * y[:, 0], phi[:, 0] * y[:, 0] - phi[:, 1] * y[:, 1]]
        )

    def y0(phi, v):
        return np.column_stack([v[:, 0], np.zeros(len(v))])

    def output(y, phi, v):
        return y[:, 1]

    t = np.tile([0.5, 1.0, 2.0, 4.0, 8.0], 4)[:, None]
    phi = np.repeat([[1.0, 0.1], [1.5, 0.2], [0.8, 0.05], [2.0, 0.3]], 5, axis=0)
    v = np.repeat([[10.0], [20.0], [5.0], [8.0]], 5, axis=0)
    expected = ode_model(rhs, y0, output)(phi, t, v)
    got = ode_model(rhs, y0, output, method=method)(phi, t, v)
    np.testing.assert_allclose(got, expected, rtol=1e-7)
