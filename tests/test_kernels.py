"""
The optional Rust kernels (kernels/, package pynlme-kernels) against the NumPy
code they replace, on random problems of random sizes: they must do the same
arithmetic for any data, not just the validation cases.
"""

import numpy as np
import pytest

import pynlme.lindstrom_bates as lb

pytestmark = pytest.mark.skipif(lb._kernels is None, reason="pynlme-kernels not installed")


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
    J = rng.normal(size=(m, n_max, p)) * (rng.uniform(0.01, 100) if scale is None else scale)
    res = rng.normal(size=(m, n_max))
    lengths = rng.integers(1, n_max + 1, size=m)  # unbalanced groups: zero-padded rows
    mask = np.arange(n_max)[None, :] < lengths[:, None]
    J, res = J * mask[..., None], res * mask
    lme = lb.LinearizedLME(_Model(A, B, int(mask.sum())), rng.normal(size=f), rng.normal(size=(r, m)) * 0.1,
                           res, J, reml=bool(rng.integers(0, 2)))
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
    np.testing.assert_allclose(beta_k, beta_p, rtol=1e-8, atol=1e-10 * np.max(np.abs(beta_p)))
    np.testing.assert_allclose(s2_k, s2_p, rtol=1e-10)
    np.testing.assert_allclose(A_k, A_p, rtol=1e-9, atol=1e-12 * np.max(np.abs(A_p)))
    np.testing.assert_allclose(u_k, u_p, rtol=1e-8, atol=1e-10 * max(np.max(np.abs(u_p)), 1.0))
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
            worst = max(worst, float(np.max(np.abs(lb._em_factor(lme, L, full) - em_p))))
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
    "rosenbrock": (lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2, [-1.2, 1.0]),
    "quadratic": (lambda v: float(np.sum((np.asarray(v) - [1, -2, 3]) ** 2 * [1, 10, 100])), [0.0, 0.0, 0.0]),
    "exp-sum": (lambda v: float(np.exp(v[0]) + np.exp(-v[1]) + (v[0] - v[1]) ** 2), [2.0, -1.0]),
}


@pytest.mark.parametrize("name", SMOOTH)
def test_optif9_port(name):
    """The Rust optif9 takes the Python port's steps (up to rounding in the BFGS update)."""
    f, x0 = SMOOTH[name]
    for itnlim in (1, 2, 5, 20):
        a = np.asarray(lb._kernels.optif9_py(f, list(x0), itnlim))
        b = lb._optif9(f, np.asarray(x0, dtype=float), itnlim)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", range(40))
def test_nlm_kernel(seed):
    """nlme's nlm refinement of the covariance factor, on problems scaled like real
    linearisations: the same objective value as the Python version (to 1e-5). (On extreme
    random problems both can wander far, along different paths.)"""
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
