"""Calling conventions, options handling and invariances of nlmefit."""

import numpy as np
import pytest

from pynlme import nlmefit

from .validation.cases import orange, theoph


@pytest.fixture(scope="module")
def theoph_fit():
    c = theoph()
    return c, nlmefit(c.X, c.y, c.group, c.V, c.modelfun, c.beta0, **c.kwargs)


def test_python_option_names_use_zero_based_indices(theoph_fit):
    c, (beta, psi, _, _) = theoph_fit
    beta2, psi2, _, _ = nlmefit(
        c.X, c.y, c.group, c.V, c.modelfun, c.beta0,
        param_transform=[1, 1, 1], re_params_select=[1, 2],
    )
    np.testing.assert_allclose(beta2, beta, rtol=1e-10)
    np.testing.assert_allclose(psi2, psi, rtol=1e-10)


def test_logical_parameter_selection(theoph_fit):
    c, (beta, _, _, _) = theoph_fit
    beta2, _, _, _ = nlmefit(
        c.X, c.y, c.group, c.V, c.modelfun, c.beta0,
        ParamTransform=[1, 1, 1], REParamsSelect=np.array([False, True, True]),
    )
    np.testing.assert_allclose(beta2, beta, rtol=1e-10)


def test_output_shapes_follow_matlab(theoph_fit):
    c, (beta, psi, stats, b) = theoph_fit
    m = len(np.unique(c.group))
    assert beta.shape == (3,)
    assert psi.shape == (2, 2)
    assert b.shape == (2, m)
    assert stats.ires.shape == stats.pres.shape == c.y.shape
    num_param = 3 + 2 + 1  # fixed effects, diagonal covariance, error sd
    assert stats.dfe == len(c.y) - num_param
    assert stats.aic == pytest.approx(-2 * stats.logl + 2 * num_param)
    assert stats.bic == pytest.approx(-2 * stats.logl + np.log(m) * num_param)


def test_invariant_to_row_order_and_group_labels(theoph_fit):
    c, (beta, psi, _, b) = theoph_fit
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(c.y))
    # Relabel groups with strings whose sorted order matches the original.
    labels = np.array([f"subject-{int(g):02d}" for g in c.group])
    beta2, psi2, _, b2 = nlmefit(
        c.X[perm], c.y[perm], labels[perm], c.V, c.modelfun, c.beta0, **c.kwargs
    )
    np.testing.assert_allclose(beta2, beta, rtol=1e-6)
    np.testing.assert_allclose(psi2, psi, rtol=1e-5)
    np.testing.assert_allclose(b2, b, atol=1e-6)


def test_model_without_v_argument_is_called_with_two_arguments():
    c = orange()
    calls = []

    def model(phi, t):
        calls.append(t.shape)
        return phi[0] / (1 + np.exp(-(t - phi[1]) / phi[2]))

    nlmefit(c.X, c.y, c.group, None, model, c.beta0, **c.kwargs)
    assert calls and all(shape[1] == 1 for shape in calls)


def test_model_requiring_third_argument_gets_none():
    c = orange()

    def model(phi, t, v):
        assert v is None
        return phi[0] / (1 + np.exp(-(t - phi[1]) / phi[2]))

    nlmefit(c.X, c.y, c.group, None, model, c.beta0, **c.kwargs)


@pytest.mark.parametrize(
    "option",
    [
        {"ErrorModel": "proportional", "ApproximationType": "FOCE"},
        {"FEObsDesign": np.zeros((3, 3, 35))},
        {"CovPattern": np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])},
    ],
)
def test_unimplemented_options_raise(option):
    c = orange()
    kwargs = {k: v for k, v in c.kwargs.items() if not ("CovPattern" in option and k == "REParamsSelect")}
    with pytest.raises(NotImplementedError):
        nlmefit(c.X, c.y, c.group, None, c.modelfun, c.beta0, **kwargs, **option)


def test_unknown_option_raises():
    c = orange()
    with pytest.raises(TypeError):
        nlmefit(c.X, c.y, c.group, None, c.modelfun, c.beta0, NotAnOption=1)


def test_statset_options():
    c = orange()
    with pytest.warns(RuntimeWarning, match="did not converge"):
        nlmefit(c.X, c.y, c.group, None, c.modelfun, c.beta0, Options={"MaxIter": 1}, **c.kwargs)


@pytest.mark.parametrize("approx", ["LME", "RELME", "FOCE"])
def test_full_vectorization_gives_identical_fits(approx):
    """Vectorization="Full" evaluates many groups per model call; the
    arithmetic is the same, so the estimates must be too."""
    from .validation.cases import vectorized

    c = theoph()
    cv = vectorized(c)
    a = nlmefit(c.X, c.y, c.group, c.V, c.modelfun, c.beta0, **c.kwargs, ApproximationType=approx)
    b = nlmefit(cv.X, cv.y, cv.group, cv.V, cv.modelfun, cv.beta0, **cv.kwargs, ApproximationType=approx)
    np.testing.assert_allclose(b[0], a[0], rtol=1e-8)
    np.testing.assert_allclose(b[1], a[1], rtol=1e-6, atol=1e-12)
    assert b[2].logl == pytest.approx(a[2].logl, abs=1e-8)


def test_full_vectorization_saem_identical_with_seed():
    from pynlme import nlmefitsa

    from .validation.cases import vectorized

    c = orange()
    cv = vectorized(c)
    a = nlmefitsa(c.X, c.y, c.group, None, c.modelfun, c.beta0, **c.kwargs, random_state=3, NIterations=[20, 20, 20])
    b = nlmefitsa(cv.X, cv.y, cv.group, None, cv.modelfun, cv.beta0, **cv.kwargs, random_state=3, NIterations=[20, 20, 20])
    np.testing.assert_allclose(b[0], a[0], rtol=1e-8)
    np.testing.assert_allclose(b[1], a[1], rtol=1e-6)


def test_compat_modes_differ_only_in_conventions(theoph_fit):
    c, (beta, psi, stats, _) = theoph_fit
    beta_r, psi_r, stats_r, _ = nlmefit(c.X, c.y, c.group, c.V, c.modelfun, c.beta0, **c.kwargs, compat="r")
    np.testing.assert_allclose(beta_r, beta, rtol=1e-6)
    np.testing.assert_allclose(psi_r, psi, rtol=1e-5)
    n, f = len(c.y), len(beta)
    np.testing.assert_allclose(stats_r.sebeta, stats.sebeta * np.sqrt(n / (n - f)), rtol=1e-6)
    num_param = n - stats.dfe
    assert stats_r.bic - stats.bic == pytest.approx(num_param * (np.log(n) - np.log(12)))


def test_bad_compat_raises():
    c = orange()
    with pytest.raises(ValueError):
        nlmefit(c.X, c.y, c.group, None, c.modelfun, c.beta0, compat="python")
