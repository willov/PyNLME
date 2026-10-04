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

FACTORIES = {"indometh": indometh, "theoph": theoph, "orange": orange, "linear": linear_sim}


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
    np.testing.assert_allclose(analytic.jacobian_padded(etas), numeric.jacobian_padded(etas), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("approx", ["LME", "RELME", "FOCE"])
@pytest.mark.parametrize("full", [True, False], ids=["Full", "SinglePhi"])
def test_fits_agree_with_finite_difference_fits(approx, full):
    numeric = vectorized(theoph())
    analytic = with_derivatives(theoph(), full=full)
    a = nlmefit(numeric.X, numeric.y, numeric.group, numeric.V, numeric.modelfun, numeric.beta0,
                **numeric.kwargs, ApproximationType=approx)
    b = nlmefit(analytic.X, analytic.y, analytic.group, analytic.V, analytic.modelfun, analytic.beta0,
                **analytic.kwargs, ApproximationType=approx)
    np.testing.assert_allclose(b[0], a[0], rtol=1e-5)
    np.testing.assert_allclose(np.diag(b[1]), np.diag(a[1]), rtol=1e-4)
    assert b[2].logl == pytest.approx(a[2].logl, abs=1e-4)


def test_still_matches_matlab_with_analytic_derivatives():
    c = with_derivatives(theoph())
    beta, psi, stats, _ = nlmefit(c.X, c.y, c.group, c.V, c.modelfun, c.beta0, **c.kwargs)
    ref = load_reference("matlab_theoph_lme_tight.json")
    np.testing.assert_allclose(beta, ref["beta"], rtol=1e-5)
    np.testing.assert_allclose(stats.sebeta, ref["stats"]["sebeta"], rtol=1e-4)


def test_saem_with_analytic_derivatives():
    """Theophylline has a parameter without random effects (log ke), whose
    SAEM update uses the derivatives."""
    numeric = vectorized(theoph())
    analytic = with_derivatives(theoph())
    a = nlmefitsa(numeric.X, numeric.y, numeric.group, numeric.V, numeric.modelfun, numeric.beta0,
                  **numeric.kwargs, random_state=2)
    b = nlmefitsa(analytic.X, analytic.y, analytic.group, analytic.V, analytic.modelfun, analytic.beta0,
                  **analytic.kwargs, random_state=2)
    np.testing.assert_allclose(b[0], a[0], atol=2e-3)
