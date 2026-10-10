"""
The Rust kernels (rust/, module pynlme._kernels) against the NumPy
code they replace, on random problems of random sizes: they must do the same
arithmetic for any data, not just the validation cases.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import pynlme.lindstrom_bates as lb

pytestmark = pytest.mark.skipif(
    lb._kernels is None, reason="compiled kernels not built"
)


class _Model:
    """Just what LinearizedLME reads from a model."""

    def __init__(self, A, B, n_obs):
        self.A, self.B, self.n_obs = A, B, n_obs
        self.n_groups, self.n_fixed, self.n_random = A.shape[0], A.shape[2], B.shape[2]


def random_problem(rng, scale=None):
    m = int(rng.integers(2, 40))
    p = int(rng.integers(1, 6))
    r = int(rng.integers(1, p + 1))
    f = int(rng.integers(1, p + 2))
    n_max = int(rng.integers(1, 15))
    A = rng.normal(size=(m, p, f))
    B = np.zeros((m, p, r))
    B[:, :r, :] = np.eye(r)
    J = rng.normal(size=(m, n_max, p)) * (
        rng.uniform(0.01, 100) if scale is None else scale
    )
    res = rng.normal(size=(m, n_max))
    lengths = rng.integers(1, n_max + 1, size=m)  # unbalanced groups: zero-padded rows
    mask = np.arange(n_max)[None, :] < lengths[:, None]
    J, res = J * mask[..., None], res * mask
    lme = lb.LinearizedLME(
        _Model(A, B, int(mask.sum())),
        rng.normal(size=f),
        rng.normal(size=(r, m)) * 0.1,
        res,
        J,
        reml=bool(rng.integers(0, 2)),
    )
    full = bool(rng.integers(0, 2))
    L = np.diag(np.exp(rng.normal(size=r)))
    if full and r > 1:
        L[np.tril_indices(r, -1)] = rng.normal(size=r * (r - 1) // 2) * 0.3
    return lme, L, full


@pytest.mark.parametrize("seed", range(40))
def test_kernels_match_numpy(seed):
    rng = np.random.default_rng(seed)
    lme, L, full = random_problem(rng)
    kernels = lb._kernels
    try:
        out_k = lme.solve(L)
        em_k = lb._em_factor(lme, L, full)
        lb._kernels = None
        out_p = lme.solve(L)
        em_p = lb._em_factor(lme, L, full)
    finally:
        lb._kernels = kernels
    logl_k, beta_k, s2_k, A_k, u_k = out_k
    logl_p, beta_p, s2_p, A_p, u_p = out_p
    assert logl_k == pytest.approx(logl_p, rel=1e-10, abs=1e-8)
    np.testing.assert_allclose(
        beta_k, beta_p, rtol=1e-8, atol=1e-10 * np.max(np.abs(beta_p))
    )
    np.testing.assert_allclose(s2_k, s2_p, rtol=1e-10)
    np.testing.assert_allclose(A_k, A_p, rtol=1e-9, atol=1e-12 * np.max(np.abs(A_p)))
    np.testing.assert_allclose(
        u_k, u_p, rtol=1e-8, atol=1e-10 * max(np.max(np.abs(u_p)), 1.0)
    )
    # nlme's EM works with the groups' cross-products, which square the
    # conditioning: on ill-conditioned random problems NumPy's own result moves
    # by up to ~1e-5 when the inputs change by 1e-15. The kernel must agree
    # within that sensitivity (or 1e-8 on well-conditioned problems).
    scale = np.max(np.abs(em_p))
    sensitivity = _em_sensitivity(lme, L, full, em_p) / scale
    assert np.max(np.abs(em_k - em_p)) / scale <= max(1e-8, 50 * sensitivity)


def _em_sensitivity(lme, L, full, em_p, repeats=4):
    """How much the NumPy EM result moves when Z changes by 1e-15 (relative)."""
    kernels, Z, ZtZ = lb._kernels, lme.Z, lme.ZtZ
    worst = 0.0
    try:
        lb._kernels = None
        for k in range(repeats):
            lme.Z = Z * (1 + 1e-15 * np.random.default_rng(k).standard_normal(Z.shape))
            lme.ZtZ = np.einsum("mnj,mnk->mjk", lme.Z, lme.Z)
            worst = max(
                worst, float(np.max(np.abs(lb._em_factor(lme, L, full) - em_p)))
            )
    finally:
        lb._kernels, lme.Z, lme.ZtZ = kernels, Z, ZtZ
    return worst


@pytest.mark.parametrize("seed", range(40))
def test_gradient_kernel(seed):
    lme, L, _ = random_problem(np.random.default_rng(seed))
    lme.reml = False
    kernels = lb._kernels
    try:
        lme.solve(L)
        g_k = lme.gradient()
        lb._kernels = None
        lme.solve(L)
        g_p = lme.gradient()
    finally:
        lb._kernels = kernels
    np.testing.assert_allclose(g_k, g_p, rtol=1e-6, atol=1e-9 * np.max(np.abs(g_p)))


SMOOTH = {
    "rosenbrock": (
        lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2,
        [-1.2, 1.0],
    ),
    "quadratic": (
        lambda v: float(np.sum((np.asarray(v) - [1, -2, 3]) ** 2 * [1, 10, 100])),
        [0.0, 0.0, 0.0],
    ),
    "exp-sum": (
        lambda v: float(np.exp(v[0]) + np.exp(-v[1]) + (v[0] - v[1]) ** 2),
        [2.0, -1.0],
    ),
}


@pytest.mark.parametrize("name", SMOOTH)
def test_optif9_rust_vs_python(name):
    """The Rust optif9 takes the Python version's steps (up to BFGS-update rounding)."""
    f, x0 = SMOOTH[name]
    for itnlim in (1, 2, 5, 20):
        a = np.asarray(lb._kernels.optif9_py(f, list(x0), itnlim))
        b = lb._optif9(f, np.asarray(x0, dtype=float), itnlim)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", range(40))
def test_nlm_kernel(seed):
    """nlme's nlm refinement of the covariance factor, on problems scaled like real
    linearisations: the same objective value as the Python version (to 1e-5).
    (On extreme random problems both can wander far, along different paths.)"""
    rng = np.random.default_rng(seed)
    lme, L, full = random_problem(rng, scale=rng.uniform(0.3, 3))
    lme.reml = False

    def objective(Lx):
        lme.solve(Lx)
        return 0.5 * (lme.n * np.log(lme._last[3]) + lme._logdet)

    kernels = lb._kernels
    try:
        L_k = lb._nlm_factor(lme, L, full)
        lb._kernels = None
        L_p = lb._nlm_factor(lme, L, full)
    finally:
        lb._kernels = kernels
    f_k, f_p = objective(L_k), objective(L_p)
    # optif9 returns its second-to-last iterate, so paths that differ by rounding
    # end ~1e-6 apart in the objective.
    assert f_k == pytest.approx(f_p, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("name", SMOOTH)
def test_fminsearch_rust_vs_python(name):
    """The Rust Nelder-Mead takes exactly the Python version's steps (MATLAB's)."""
    from pynlme.matlab_lb import fminsearch

    f, x0 = SMOOTH[name]
    for tol in (1e-2, 1e-4, 1e-8):
        x_k, f_k, ok_k = lb._kernels.fminsearch_py(f, list(x0), tol, tol, 200 * len(x0))
        x_p, f_p, ok_p = fminsearch(f, x0, tol, tol, 200 * len(x0), np.inf)
        np.testing.assert_array_equal(x_k, x_p)
        assert f_k == f_p and ok_k == ok_p


@pytest.mark.parametrize("seed", range(40))
def test_matlab_objective_kernels(seed):
    """MATLAB mode: the per-group QR with Delta and LMEnegloglikelihood."""
    from pynlme import matlab_lb

    rng = np.random.default_rng(seed)
    lme, L, _ = random_problem(rng, scale=rng.uniform(0.3, 3))
    lin = matlab_lb._Linearised.__new__(matlab_lb._Linearised)
    lin.model, lin.Z, lin.Xw = SimpleNamespace(n_obs=lme.n), lme.Z, lme.Xw
    Delta = np.linalg.cholesky(
        np.linalg.inv(L @ L.T)
    ).T  # upper triangular, as nlmefit's
    reml = bool(rng.integers(0, 2))
    kernels = lb._kernels
    try:
        R11_k, R10_k, R00_k = lin.decompose(Delta)
        v_k, c_k, _ = lin.negloglik(Delta, reml)
        lb._kernels = None
        R11_p, R10_p, R00_p = lin.decompose(Delta)
        v_p, c_p, _ = lin.negloglik(Delta, reml)
    finally:
        lb._kernels = kernels
    # QR factors are unique up to the signs of their rows.
    np.testing.assert_allclose(
        np.abs(R00_k), np.abs(R00_p), rtol=1e-9, atol=1e-12 * np.abs(R00_p).max()
    )
    np.testing.assert_allclose(
        np.abs(R11_k), np.abs(R11_p), rtol=1e-9, atol=1e-12 * np.abs(R11_p).max()
    )
    assert v_k == pytest.approx(v_p, rel=1e-10, abs=1e-8)
    assert c_k == pytest.approx(c_p, rel=1e-10)


def random_foce_problem(rng):
    """A model of the GroupedModel kind on simulated data of random size, with
    beta, L and starting modes; Vectorization "full" or per group."""
    from pynlme.data_types import NLMEOptions
    from pynlme.nlmefit import _build_model, _parse_options
    from pynlme.utils import validate_inputs

    m = int(rng.integers(2, 25))
    lengths = rng.integers(2, 12, size=m)
    group = np.repeat(np.arange(m), lengths)
    t = rng.uniform(0, 5, size=group.size)
    r = int(rng.integers(1, 4))
    beta = np.array([2.0, -0.5, 0.3]) + rng.normal(size=3) * 0.2
    b = rng.normal(size=(m, 3)) * 0.2
    b[:, r:] = 0
    phi = beta + b[group]
    y = phi[:, 0] * np.exp(-np.exp(phi[:, 1]) * t) + phi[:, 2]
    y = y + rng.normal(size=y.size) * 0.05

    def model(phi, t):
        phi = np.atleast_2d(phi)
        return phi[..., 0] * np.exp(-phi[..., 1] * t[:, 0]) + phi[..., 2]

    kwargs = {
        "ParamTransform": [0, 1, 0],
        "REParamsSelect": list(range(1, r + 1)),
        "Vectorization": "Full" if rng.integers(0, 2) else "SinglePhi",
    }
    parsed, matlab_style = _parse_options(kwargs)
    X, y, g, V, beta0 = validate_inputs(t, y, group, None, beta)
    built = _build_model(X, y, g, V, model, beta0, NLMEOptions(**parsed), matlab_style)
    L = np.diag(np.exp(rng.normal(size=r) * 0.5))
    if r > 1:
        L[np.tril_indices(r, -1)] = rng.normal(size=r * (r - 1) // 2) * 0.3
    weights = (
        None if rng.integers(0, 2) else [rng.uniform(0.5, 2, size=n) for n in lengths]
    )
    U0 = rng.normal(size=(m, r)) * 0.1
    return built, beta, L, U0, weights


@pytest.mark.parametrize("seed", range(20))
def test_foce_kernels_match_numpy(seed):
    """conditional_modes and linearized_marginal: the Rust kernels against the
    NumPy code (same steps; the modes to the solver's tolerance)."""
    from pynlme import first_order as fo

    rng = np.random.default_rng(seed)
    model, beta, L, U0, weights = random_foce_problem(rng)
    kernels = lb._kernels
    try:
        lb._kernels = None
        U_np = fo.conditional_modes(model, beta, L, U0, weights=weights)
        ll_np = fo.linearized_marginal(model, beta, L, U_np, weights=weights)[0]
    finally:
        lb._kernels = kernels
    U_rs = fo.conditional_modes(model, beta, L, U0, weights=weights)
    ll_rs = fo.linearized_marginal(
        model, beta, L, U_np, weights=weights, with_parts=False
    )[0]
    # Near the mode the objective is flat to rounding, so where Levenberg-Marquardt
    # stops can differ by ~sqrt(eps) between the two (summation order).
    np.testing.assert_allclose(U_rs, U_np, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(ll_rs, ll_np, rtol=1e-12)


@pytest.mark.parametrize(
    "error_model, chains",
    [("constant", 1), ("proportional", 2), ("combined", 1), ("exponential", 3)],
)
def test_saem_kernel_same_chain(error_model, chains):
    """SAEM with the Rust MCMC kernel takes the same chain as the NumPy code (the
    random numbers are drawn in the same order), so the estimates agree."""
    from pynlme import nlmefitsa
    from tests.validation.cases import theoph, vectorized

    case = vectorized(theoph())
    keep = (case.y > 0) & (case.X > 0)

    def run():
        return nlmefitsa(
            case.X[keep],
            case.y[keep],
            case.group[keep],
            case.V,
            case.modelfun,
            case.beta0,
            **case.kwargs,
            ErrorModel=error_model,
            NChains=chains,
            NIterations=[40, 20, 20],
            random_state=3,
        )

    kernels = lb._kernels
    try:
        lb._kernels = None
        beta_np, psi_np, stats_np, _ = run()
    finally:
        lb._kernels = kernels
    beta_rs, psi_rs, stats_rs, _ = run()
    np.testing.assert_allclose(beta_rs, beta_np, rtol=1e-9)
    np.testing.assert_allclose(psi_rs, psi_np, rtol=1e-9)
    np.testing.assert_allclose(stats_rs.errorparam, stats_np.errorparam, rtol=1e-9)
