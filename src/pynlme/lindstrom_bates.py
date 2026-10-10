"""
Lindstrom-Bates estimation for nlmefit (ApproximationType "LME" and "RELME").

This is the algorithm behind MATLAB's nlmefit default and R's nlme::nlme. It
alternates two steps until the estimates stop changing:

1. PNLS: for a fixed relative covariance factor L (Psi = sigma^2 L L'), find
   beta and the spherical random effects u_i (b_i = L u_i) minimising
   sum_i ||y_i - f_i(A_i beta + B_i L u_i)||^2 + ||u_i||^2.
2. LME: linearise f_i around that solution and maximise the (restricted)
   likelihood of the resulting linear mixed model over L, with beta and
   sigma^2 profiled out.

Working with u_i instead of b_i keeps the problem well defined when a variance
component is estimated as zero (L singular).

Reference: Lindstrom & Bates (1990), Biometrics 46:673-687; Pinheiro & Bates
(2000), Mixed-Effects Models in S and S-PLUS, ch. 7.
"""

import os
from dataclasses import dataclass

import numpy as np
from scipy import optimize

# Optional compiled kernels (kernels/, package pynlme-kernels); the same
# arithmetic as the NumPy code below, without its per-call overhead.
# PYNLME_KERNELS=0 disables them.
try:
    if os.environ.get("PYNLME_KERNELS", "1") == "0":
        raise ImportError
    import pynlme_kernels as _kernels
except ImportError:
    _kernels = None

from .model import GroupedModel


@dataclass
class LBResult:
    beta: np.ndarray  # (f,)
    psi: np.ndarray  # (r, r)
    sigma2: float
    b: np.ndarray  # (r, m)
    logl: float
    covb: np.ndarray  # (f, f)
    iterations: int
    converged: bool
    error_param: np.ndarray | None = None  # (a, b) of the combined error model


class CovarianceFactor:
    """Lower-triangular L for a diagonal or full covariance pattern.

    Uses the log-Cholesky parameterisation (log of the diagonal, raw
    off-diagonal), as nlme does. With the diagonal itself as a parameter
    bounded at zero, the likelihood is stationary at every zero diagonal
    entry, so the optimizer can stall on the boundary (e.g. at a correlation
    of exactly +-1) when the optimum is inside.
    """

    LOG_MIN = -40.0  # exp(-40) ~ 4e-18: a variance component of effectively zero
    LOG_MAX = 50.0  # only guards against overflow; not a bound for the optimizer

    def __init__(self, r: int, full: bool):
        self.r = r
        self.full = full
        if full:
            self.rows, self.cols = np.tril_indices(r)
        else:
            self.rows = self.cols = np.arange(r)
        self.is_diag = self.rows == self.cols
        # Lower bounds only: when every variable has both bounds, L-BFGS-B
        # takes a full-gradient first step, which can jump to an extreme L.
        self.bounds = [(self.LOG_MIN, None) if d else (None, None) for d in self.is_diag]

    @property
    def n_params(self) -> int:
        return len(self.rows)

    def to_matrix(self, theta: np.ndarray) -> np.ndarray:
        values = np.where(self.is_diag, np.exp(np.minimum(theta, self.LOG_MAX)), theta)
        L = np.zeros((self.r, self.r))
        L[self.rows, self.cols] = values
        return L

    def from_matrix(self, L: np.ndarray) -> np.ndarray:
        values = L[self.rows, self.cols]
        safe = np.maximum(np.abs(values), np.exp(self.LOG_MIN))
        return np.where(self.is_diag, np.log(safe), values)


def _lm_step_solve(V, W, Udiag_blocks, g_beta, g_u):
    """Solve the block-arrow normal equations by eliminating the u blocks.

    [V      W_1 ... W_m] [d_beta]   [g_beta]
    [W_1'   U_1        ] [d_u_1 ] = [g_u_1 ]
    [ ...        ...   ] [ ...  ]   [ ...  ]
    """
    if g_u.shape[1] == 0:
        return np.linalg.solve(V, g_beta), g_u.copy()
    Uinv_Wt = np.linalg.solve(Udiag_blocks, np.swapaxes(W, 1, 2))  # (m, r, f)
    Uinv_g = np.linalg.solve(Udiag_blocks, g_u[..., None])[..., 0]  # (m, r)
    S = V - np.einsum("mfr,mrg->fg", W, Uinv_Wt)
    rhs = g_beta - np.einsum("mfr,mr->f", W, Uinv_g)
    d_beta = np.linalg.solve(S, rhs)
    d_u = Uinv_g - np.einsum("mrf,f->mr", Uinv_Wt, d_beta)
    return d_beta, d_u


def _weighted(w, J):
    return J if np.isscalar(w) else w[:, None] * J


def error_ab(ab, y, f):
    """MATLAB's objective for the combined error model parameters (a, b)."""
    g = abs(ab[0]) + abs(ab[1]) * np.abs(f)
    return float(np.sum(0.5 * ((y - f) / g) ** 2 + np.log(g)))


def update_weights(error_model, y, f, ab):
    """Observation weights 1/g(f) for the current fit, as nlmefit does.

    Returns the weights and (for "combined") the refitted (a, b).
    """
    if error_model == "proportional":
        w = 1.0 / np.abs(f)
    elif error_model == "combined":
        ab = np.abs(optimize.fmin(error_ab, ab, args=(y, f), disp=False, xtol=1e-10, ftol=1e-12, maxiter=5000, maxfun=10000))
        w = 1.0 / np.abs(ab[0] + ab[1] * np.abs(f))
    else:
        return np.ones_like(f), ab
    w[~np.isfinite(w)] = 1.0
    return w, ab


def pnls(
    model: GroupedModel,
    L: np.ndarray,
    beta: np.ndarray,
    U: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-12,
    weights: np.ndarray | None = None,
):
    """Penalised (weighted) nonlinear least squares by Levenberg-Marquardt.

    All groups are held in zero-padded (m, n_max) arrays. Returns beta, U
    (m, r), and the residuals (m, n_max) and Jacobians d f / d eta
    (m, n_max, p) at the solution, both multiplied by the observation
    weights if given.
    """
    r = model.n_random
    W = model.mask.astype(float) if weights is None else weights * model.mask
    A = model.A
    BL = np.einsum("mpr,rs->mps", model.B, L)
    eye_r = np.eye(r)

    def etas(beta, U):
        return A @ beta + np.einsum("mps,ms->mp", BL, U)

    def evaluate(beta, U):
        res = W * (model.y_pad - model.predict_padded(etas(beta, U)))
        return res, float(np.sum(res * res) + np.sum(U * U))

    res, S = evaluate(beta, U)
    lam = 1e-3
    for _ in range(max_iter):
        J = W[:, :, None] * model.jacobian_padded(etas(beta, U))
        G = np.einsum("mnp,mpf->mnf", J, A)
        H = np.einsum("mnp,mps->mns", J, BL)
        V = np.einsum("mnf,mng->fg", G, G)
        g_beta = np.einsum("mnf,mn->f", G, res)
        Wb = np.einsum("mnf,mns->mfs", G, H)
        Ublocks = np.einsum("mns,mnt->mst", H, H) + eye_r
        g_u = np.einsum("mns,mn->ms", H, res) - U

        improved = False
        while lam < 1e12:
            Vd = V + lam * np.diag(np.diag(V) + 1e-12)
            Ud = Ublocks + lam * eye_r[None] * np.diagonal(Ublocks, axis1=1, axis2=2)[:, None, :]
            d_beta, d_u = _lm_step_solve(Vd, Wb, Ud, g_beta, g_u)
            try:
                new_res, new_S = evaluate(beta + d_beta, U + d_u)
            except FloatingPointError:
                lam *= 10
                continue
            if new_S <= S:
                improved = True
                break
            lam *= 10
        if not improved:
            break
        beta, U = beta + d_beta, U + d_u
        decrease = S - new_S
        res, S = new_res, new_S
        lam = max(lam / 10, 1e-12)
        step = max(np.max(np.abs(d_beta) / (np.abs(beta) + 1e-8)), np.max(np.abs(d_u), initial=0.0))
        if decrease <= tol * (S + tol) and step < 1e-8:
            break

    J = W[:, :, None] * model.jacobian_padded(etas(beta, U))
    return beta, U, res, J


class LinearizedLME:
    """Profiled (restricted) log-likelihood of the linearised model.

    w_i = X_i beta + Z_i b_i + e_i with X_i = J_i A_i, Z_i = J_i B_i and the
    working response w_i = y_i - f_i + X_i beta_hat + Z_i b_hat_i.

    Evaluated with QR decompositions of the penalised least-squares system
    [Z_i L, X_i, w_i; I, 0, 0], as nlme and lme4 do. Forming cross-products
    instead (X'X - X'Z L M^-1 L'Z'X) cancels catastrophically when L is large,
    which can produce spuriously high likelihoods.
    """

    # log(L_kk) below which a variance component counts as collapsed to zero
    # (standard deviation < 3e-4 of the residual standard deviation).
    TRAPPED_LOG_SD = -8.0

    def __init__(self, model, beta, B_hat, res, J, reml: bool):
        """res (m, n_max) and J (m, n_max, p) as returned by pnls (padded)."""
        m, f, r = model.n_groups, model.n_fixed, model.n_random
        X = np.einsum("mnp,mpf->mnf", J, model.A)
        Z = np.einsum("mnp,mpr->mnr", J, model.B)
        w = res + np.einsum("mnf,f->mn", X, beta) + np.einsum("mnr,rm->mn", Z, B_hat)
        # Groups padded with zero rows to a common length (at least f + 1);
        # zero rows do not change the QR factors.
        extra = max(f + 1 - J.shape[1], 0)
        pad = ((0, 0), (0, extra), (0, 0))
        ZXw = np.pad(np.concatenate([Z, X, w[:, :, None]], axis=2), pad)
        # Every quantity below depends on [Z_i X_i w_i] only through inner
        # products, which an orthogonal transformation keeps: replace each
        # group's rows by its (r + f + 1)-square QR factor once, so that the
        # many solves (optimizer, EM) work on small blocks.
        if ZXw.shape[1] > ZXw.shape[2]:
            ZXw = np.linalg.qr(ZXw, mode="r")
        self.Z = ZXw[..., :r]
        self.Xw = ZXw[..., r:]
        self.penalty = np.concatenate(
            [np.broadcast_to(np.eye(r), (m, r, r)), np.zeros((m, r, f + 1))], axis=2
        )
        self.ZtZ = np.einsum("mnj,mnk->mjk", self.Z, self.Z)
        self.n = model.n_obs
        self.f = f
        self.r = r
        self.reml = reml

    def solve(self, L: np.ndarray):
        """Profiled log-likelihood and the GLS quantities at factor L."""
        f, r = self.f, self.r
        if _kernels is not None:
            logdet, beta, rss, R_xx, u, R_zz = _kernels.lme_solve(self.Z, self.Xw, np.ascontiguousarray(L))
            dof = self.n - f if self.reml else self.n
            sigma2 = rss / dof
            logl = -0.5 * dof * (1.0 + np.log(2.0 * np.pi * sigma2)) - 0.5 * logdet
            if self.reml:
                logl -= np.sum(np.log(np.abs(np.diag(R_xx))))
            self._last = (L, beta, u, rss, R_zz)
            return logl, beta, sigma2, R_xx.T @ R_xx, u
        top = np.concatenate([self.Z @ L, self.Xw], axis=2)
        R = np.linalg.qr(np.concatenate([top, self.penalty], axis=1), mode="r")
        R_zz, R_zx, r_zw = R[:, :r, :r], R[:, :r, r : r + f], R[:, :r, -1]
        logdet = 2.0 * np.sum(np.log(np.abs(np.diagonal(R_zz, axis1=1, axis2=2))))

        # Combine the groups' fixed-effects blocks [R_XX_i r_Xw_i; 0 s_i].
        blocks = R[:, r : r + f + 1, r:].reshape(-1, f + 1)
        R_fin = np.linalg.qr(blocks, mode="r")
        R_xx, r_xw, s = R_fin[:f, :f], R_fin[:f, f], R_fin[f, f]
        beta = np.linalg.solve(R_xx, r_xw)
        rss = max(s * s, 1e-300)

        dof = self.n - f if self.reml else self.n
        sigma2 = rss / dof
        logl = -0.5 * dof * (1.0 + np.log(2.0 * np.pi * sigma2)) - 0.5 * logdet
        if self.reml:
            logl -= np.sum(np.log(np.abs(np.diag(R_xx))))
        # Conditional modes of the spherical random effects (BLUPs in u-space).
        u = np.linalg.solve(R_zz, (r_zw - R_zx @ beta)[..., None])[..., 0]
        self._last = (L, beta, u, rss, R_zz)
        return logl, beta, sigma2, R_xx.T @ R_xx, u

    def gradient(self) -> np.ndarray:
        """d logl / d L at the last solve() (ML only).

        beta and u minimise the penalised RSS for the given L, so by the
        envelope theorem d RSS/dL = -2 sum_i Z_i' r_i u_i'. And
        d log|I + L'Z'ZL| / dL = 2 Z'Z L M^-1.
        """
        L, beta, u, rss, R_zz = self._last
        f = self.f
        resid = self.Xw[..., f] - self.Xw[..., :f] @ beta - np.einsum("mnr,mr->mn", self.Z @ L, u)
        d_rss = -2.0 * np.einsum("mnj,mn,mk->jk", self.Z, resid, u)
        R_inv = np.linalg.solve(R_zz, np.broadcast_to(np.eye(self.r), R_zz.shape))
        M_inv = R_inv @ np.swapaxes(R_inv, 1, 2)
        d_logdet = 2.0 * np.einsum("mjk,kl,mlp->jp", self.ZtZ, L, M_inv)
        return -0.5 * self.n / rss * d_rss - 0.5 * d_logdet

    def maximize(self, factor: CovarianceFactor, theta0: np.ndarray, tight: bool = True, restart: bool = True):
        """Maximise over the covariance factor.

        Returns the optimum and whether a restart of collapsed components
        found a better one. ``tight`` selects the final, strict tolerances;
        early outer iterations do not need them.
        """
        def nll(theta):
            try:
                value = -self.solve(factor.to_matrix(theta))[0]
            except np.linalg.LinAlgError:
                return np.inf
            return value if np.isfinite(value) else np.inf

        def fd_grad(theta):
            g = np.empty_like(theta)
            for k in range(theta.size):
                h = 1e-6 * max(abs(theta[k]), 1.0)
                up, down = theta.copy(), theta.copy()
                up[k] += h
                down[k] -= h
                lower = factor.bounds[k][0]
                if lower is not None and down[k] < lower:
                    down[k] = theta[k]
                    g[k] = (nll(up) - nll(down)) / h
                else:
                    g[k] = (nll(up) - nll(down)) / (2 * h)
            return g

        def fun_and_grad(theta):
            L = factor.to_matrix(theta)
            try:
                value = -self.solve(L)[0]
            except np.linalg.LinAlgError:
                return np.inf, np.zeros_like(theta)
            if not np.isfinite(value):
                return np.inf, np.zeros_like(theta)
            G = self.gradient()[factor.rows, factor.cols]
            # Chain rule for the log-diagonal parameters.
            G = np.where(factor.is_diag, G * L[factor.rows, factor.cols], G)
            return value, -G

        if self.reml:
            objective, jac = nll, fd_grad
        else:
            objective, jac = fun_and_grad, True

        # A component driven to (near) zero variance has a vanishing gradient
        # in log space and cannot recover by itself, even when a later
        # linearisation favours a positive variance. Also try from a moderate
        # value and keep the better optimum.
        starts = [theta0]
        trapped = factor.is_diag & (theta0 < self.TRAPPED_LOG_SD)
        if restart and np.any(trapped):
            start = theta0.copy()
            start[trapped] = 0.0
            starts.append(start)
        tol = {"ftol": 1e-15, "gtol": 1e-10} if tight else {"ftol": 1e-10, "gtol": 1e-6}
        results = [
            optimize.minimize(
                objective, start, jac=jac, method="L-BFGS-B", bounds=factor.bounds,
                options={**tol, "maxiter": 1000},
            )
            for start in starts
        ]
        best = min(range(len(results)), key=lambda k: results[k].fun)
        self.last_result = results[best]
        helped = best > 0 and results[best].fun < results[0].fun - 1e-9
        return results[best].x, helped


def _stagnated(history, tol, window: int = 8) -> bool:
    """True when the last iterations only jitter rather than drift.

    Jitter: the estimates stay within tol of each other, and the net change
    over the window is at most half of the range covered (a slow but steady
    drift, e.g. while error-model weights are still settling, moves about as
    far as its range).
    """
    if len(history) < window:
        return False
    betas = np.array([h[0] for h in history[-window:]])
    logls = np.array([h[1] for h in history[-window:]])
    scale = np.maximum(np.abs(betas).mean(axis=0), 1e-8)
    span = np.ptp(betas, axis=0)
    net = np.abs(betas[-1] - betas[0])
    jitter_only = np.all(net <= 0.5 * span + 1e-15) and abs(logls[-1] - logls[0]) <= 0.5 * np.ptp(logls) + 1e-15
    return bool(np.max(span / scale) < tol and jitter_only)


def initial_factor(model: GroupedModel, J: list[np.ndarray]) -> np.ndarray:
    """Starting relative covariance factor, as nlme's Initialize.reStruct.

    D = Psi / sigma^2 = diag(m / (0.375^2 * sum_ij Z_ijk^2)) with Z the
    derivatives with respect to the random effects at the start values, so
    each random effect starts at a scale matched to the model's sensitivity.
    """
    m = model.n_groups
    z2 = sum(np.sum((J[i] @ model.B[i]) ** 2, axis=0) for i in range(m))
    d = np.where(z2 > 0, m / (0.375**2 * np.maximum(z2, 1e-300)), 1.0)
    return np.diag(np.sqrt(d))


def refine_beta0(model: GroupedModel, beta0: np.ndarray) -> np.ndarray:
    """Pooled nonlinear least squares with random effects set to zero."""
    r = model.n_random
    beta, _, _, _ = pnls(model, np.zeros((r, r)), beta0, np.zeros((model.n_groups, r)))
    return beta


def fit_lindstrom_bates(
    model: GroupedModel,
    beta0: np.ndarray,
    full_cov: bool = False,
    reml: bool = False,
    refine: bool = True,
    max_iter: int = 200,
    tol_x: float = 1e-6,
    tol_fun: float = 1e-6,
    verbose: int = 0,
    error_model: str = "constant",
    error_param0=None,
) -> LBResult:
    """Lindstrom-Bates estimation.

    For "proportional" and "combined" error models the PNLS and LME steps are
    weighted least squares with weights 1/g(f) recomputed from the current fit
    after each LME step, and log(weights) is added to the log-likelihood, as
    MATLAB's nlmefit does. The first iteration uses constant weights.
    """
    m, r = model.n_groups, model.n_random
    factor = CovarianceFactor(r, full_cov)
    weights = model.mask.astype(float)
    ab = np.array([1.0, 1.0] if error_param0 is None else error_param0, dtype=float)
    beta = np.asarray(beta0, dtype=float).copy()
    if refine:
        beta = refine_beta0(model, beta)
    U = np.zeros((m, r))

    # As nlme, start with an LME step linearised at the start values (b = 0),
    # so the first PNLS step uses a covariance estimated from the data rather
    # than a guess. Starting PNLS from a poor guess can lead to a different
    # (worse) fixed point.
    eta0 = model.etas_all(beta)
    J0 = model.jacobian_padded(eta0)
    res0 = model.mask * (model.y_pad - model.predict_padded(eta0))
    lme0 = LinearizedLME(model, beta, np.zeros((r, m)), res0, J0, reml)
    theta, _ = lme0.maximize(factor, factor.from_matrix(initial_factor(model, J0)), tight=False)
    prev = None
    near = False  # estimates close to converged: use strict tolerances
    history = []
    skip_restart = 0  # iterations to skip the collapsed-variance restart
    best = None
    converged = False
    damping = 1.0

    for iteration in range(1, max_iter + 1):
        L = factor.to_matrix(theta)
        beta, U, res, J = pnls(model, L, beta, U, weights=weights)
        B_hat = L @ U.T
        lme = LinearizedLME(model, beta, B_hat, res, J, reml)
        # The LB iteration can cycle between states instead of converging.
        # Damp the covariance update after every 25 iterations without
        # convergence.
        if iteration % 25 == 0:
            damping /= 2
        theta_new, helped = lme.maximize(factor, theta, tight=near, restart=skip_restart == 0)
        # A restart that does not help is unlikely to help in the next few
        # iterations either; it is the most expensive part of the LME step.
        skip_restart = 0 if helped else (skip_restart - 1 if skip_restart else 4)
        theta = theta + damping * (theta_new - theta)
        L_new = factor.to_matrix(theta)
        logl, beta_lme, sigma2, A, U_new = lme.solve(L_new)
        logl += float(np.sum(np.log(weights[model.mask])))
        psi = sigma2 * L_new @ L_new.T
        state = LBResult(
            beta=beta.copy(), psi=psi, sigma2=sigma2, b=B_hat, logl=logl,
            covb=sigma2 * np.linalg.inv(A), iterations=iteration, converged=False,
            error_param=ab.copy(),
        )
        if error_model in ("proportional", "combined"):
            f = model.predict_padded(model.etas_all(beta, B_hat))[model.mask]
            w_flat, ab = update_weights(error_model, model.y_pad[model.mask], f, ab)
            weights = np.zeros(model.mask.shape)
            weights[model.mask] = w_flat
        if best is None or logl > best.logl:
            best = state
        if verbose:
            print(f"LB iteration {iteration}: logl={logl:.6f} beta={beta}")

        if prev is not None:
            beta_change = np.max(np.abs(beta - prev[0]) / np.maximum(np.abs(prev[0]), 1e-8))
            psi_scale = max(np.max(np.abs(np.diag(psi))), 1e-300)
            psi_change = np.max(np.abs(psi - prev[1])) / psi_scale
            logl_change = abs(logl - prev[2]) / (abs(prev[2]) + 1.0)
            if near and max(beta_change, psi_change) < tol_x and logl_change < tol_fun:
                converged = True
                break
            near = near or max(beta_change, psi_change) < 1e-3
            history.append((beta.copy(), logl))
            if _stagnated(history, 100 * tol_x):
                # Changes are at the model's numerical noise level (e.g. an
                # ODE solved to finite tolerance): the best state is converged.
                best.converged = True
                return best
        prev = (beta.copy(), psi.copy(), logl)
        # Start the next PNLS step from the LME step's GLS estimates and the
        # conditional modes under the new covariance.
        beta, U = beta_lme, U_new

    if converged:
        state.converged = True
        return state
    # Not converged (e.g. a cycle): return the best state seen; the caller warns.
    best.iterations = iteration
    return best


# --- R nlme's iteration (compat="r") ------------------------------------------
#
# nlme follows the same alternating scheme but its PNLS step differs from the
# one above in ways that change which fixed point is reached when there are
# several (nlme 3.1, src/nlme.c, nlme_iterate and nlme_increment):
#
# - Each PNLS iteration re-linearises the model at the current (beta, b),
#   updates the relative covariance factor by 20 EM iterations on that
#   linear model (ML), and steps towards its GLS/BLUP solution.
# - Steps are halved until the penalised objective decreases; the step factor
#   then grows by 4 (at most 1) for the next iteration.
# - It stops at nls's relative-offset criterion (pnlsTol) or after
#   pnlsMaxIter (7) iterations, i.e. usually before PNLS has converged.
# - The next LME step starts from the EM-updated factor, and the next PNLS
#   from the previous PNLS solution (not from the LME step's estimates).
#
# After EM, nlme_increment refines the factor with R's nlm() engine (optif9:
# line search, BFGS secant updates, forward-difference gradients, at most 20
# iterations) and then rebuilds it from optif9's *input* vector. optif9
# overwrites that vector with each accepted iterate except the last one, so
# the factor nlme uses is the second-to-last nlm iterate (the EM result when
# nlm stops after its first step). Reproducing R's iterates requires the same;
# _optif9 below implements that optimizer from Dennis & Schnabel (1983,
# Appendix A: algorithms A6.3.1, A9.4.2, A5.6.3/A5.6.4, A7.2.1) with optif9's
# settings in nlme.


def _inv_small(A: np.ndarray) -> np.ndarray:
    """Inverse of (batched) 1x1 or 2x2 matrices in closed form (LAPACK's call
    overhead dominates at this size); np.linalg.inv otherwise."""
    r = A.shape[-1]
    if r == 1:
        return 1.0 / A
    if r == 2:
        a, b, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
        det = a * d - b * c
        out = np.empty_like(A)
        out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = d / det, -b / det, -c / det, a / det
        return out
    return np.linalg.inv(A)


def _em_factor(lme: "LinearizedLME", L: np.ndarray, full: bool, n_iter: int = 20) -> np.ndarray:
    """nlme's EM update of the relative factor (internal_EM, ML).

    Each iteration: with precision P = (L L')^-1 and M_i = Z_i'Z_i + P, the
    BLUPs b_i, sigma^2 = penalised RSS / N and Var(b_i | y)/sigma^2 = M_i^-1
    give Psi/sigma^2 = mean_i(M_i^-1 + b_i b_i' / sigma^2). Written with the
    groups' cross-products (fixed during EM) for speed.
    """
    f = lme.f
    ZZ = lme.ZtZ
    ZXw = np.einsum("mnj,mnk->mjk", lme.Z, lme.Xw)
    XX = np.einsum("mnj,mnk->jk", lme.Xw, lme.Xw)
    if _kernels is not None:
        return _kernels.em_factor(np.ascontiguousarray(ZZ), np.ascontiguousarray(ZXw), np.ascontiguousarray(XX),
                                  np.ascontiguousarray(L), float(lme.n), full, n_iter)
    m = ZZ.shape[0]
    for _ in range(n_iter):
        M_inv = _inv_small(ZZ + _inv_small(L @ L.T))
        T = M_inv @ ZXw  # (m, r, f + 1)
        S = XX - np.einsum("mrj,mrk->jk", ZXw, T)
        beta = np.linalg.solve(S[:f, :f], S[:f, f])
        sigma2 = (S[f, f] - S[f, :f] @ beta) / lme.n
        b = T[:, :, f] - T[:, :, :f] @ beta
        psi = (M_inv.sum(axis=0) + b.T @ b / sigma2) / m
        L = np.linalg.cholesky(psi) if full else np.diag(np.sqrt(np.diag(psi)))
    return L


_DBL_MAX = np.finfo(float).max
_EPS = np.finfo(float).eps


def _optif9(fcn, x, itnlim: int = 20) -> np.ndarray:
    """R's optif9 as called by nlme_increment (method 1, iexp 1, no analytic
    derivatives, typsiz 1, fscale 1, ndigit 15, gradtl eps^(1/3), steptl
    sqrt(eps), default stepmx). Returns the vector optif9 leaves in its input
    argument: the last accepted iterate before the terminating one."""
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    rnf = max(1e-15, _EPS)
    gradtl, steptl = _EPS ** (1 / 3), np.sqrt(_EPS)
    stepmx = 1000.0 * max(np.linalg.norm(x), 1.0)

    def f_safe(v):
        value = fcn(v)
        return value if np.isfinite(value) else _DBL_MAX

    def forward_gradient(v, fv):
        g = np.empty(n)
        for j in range(n):
            h = np.sqrt(rnf) * max(abs(v[j]), 1.0)
            w = v.copy()
            w[j] += h
            g[j] = (f_safe(w) - fv) / h
        return g

    def central_gradient(v):
        g = np.empty(n)
        for j in range(n):
            h = rnf ** (1 / 3) * max(abs(v[j]), 1.0)
            up, down = v.copy(), v.copy()
            up[j] += h
            down[j] -= h
            g[j] = (f_safe(up) - f_safe(down)) / (2 * h)
        return g

    def converged_gradient(v, fv, g):
        return np.max(np.abs(g) * np.maximum(np.abs(v), 1.0)) / max(abs(fv), 1.0) <= gradtl

    def line_search(x, f, g, p):
        """A6.3.1: backtracking (quadratic, then cubic) with the Armijo test."""
        sln = np.linalg.norm(p)
        if sln > stepmx:
            p = p * (stepmx / sln)
            sln = stepmx
        slp = g @ p
        rln = np.max(np.abs(p) / np.maximum(np.abs(x), 1.0))
        rmnlmb = steptl / rln
        lam, first = 1.0, True
        plam = pf = 0.0
        while True:
            xp = x + lam * p
            fp = f_safe(xp)
            if fp <= f + slp * 1e-4 * lam:
                return xp, fp, 0, lam == 1.0 and sln > stepmx * 0.99
            if lam < rmnlmb:
                return xp, fp, 1, False
            if fp >= _DBL_MAX:
                lam *= 0.1
                first = True
                continue
            if first:
                tlam = -lam * slp / ((fp - f - slp) * 2.0)
                first = False
            else:
                t1 = fp - f - lam * slp
                t2 = pf - f - plam * slp
                t3 = 1.0 / (lam - plam)
                a3 = 3.0 * t3 * (t1 / lam**2 - t2 / plam**2)
                b = t3 * (t2 * lam / plam**2 - t1 * plam / lam**2)
                disc = b * b - a3 * slp
                if disc > b * b:
                    tlam = (-b + (-np.sqrt(disc) if a3 < 0 else np.sqrt(disc))) / a3
                else:
                    tlam = (-b + (np.sqrt(disc) if a3 < 0 else -np.sqrt(disc))) / a3
                tlam = min(tlam, lam * 0.5)
            plam, pf = lam, fp
            lam = lam * 0.1 if tlam < lam * 0.1 else tlam

    f = f_safe(x)
    g = forward_gradient(x, f)
    if converged_gradient(x, f, g):
        return x
    L = np.eye(n)  # Cholesky factor of the secant Hessian (hsnint)
    central = False
    consecutive_max = 0
    for iteration in range(1, itnlim + 1):
        while True:
            p = -np.linalg.solve(L.T, np.linalg.solve(L, g))
            xp, fp, iretcd, mxtake = line_search(x, f, g, p)
            if iretcd == 1 and not central:
                central = True  # retry with central differences
                g = central_gradient(x)
                continue
            break
        if iretcd == 1:
            return x
        gp = central_gradient(xp) if central else forward_gradient(xp, fp)
        # Stopping rules (opt_stop): relative gradient, relative step,
        # iteration limit, five consecutive maximal steps.
        if converged_gradient(xp, fp, gp):
            return x
        if np.max(np.abs(xp - x) / np.maximum(np.abs(xp), 1.0)) <= steptl or iteration >= itnlim:
            return x
        if mxtake:
            consecutive_max += 1
            if consecutive_max >= 5:
                return x
        else:
            consecutive_max = 0
        # BFGS update of the factor (secfac), with the first-update scaling.
        s, y = xp - x, gp - g
        den1 = s @ y
        if den1 >= np.sqrt(_EPS) * np.linalg.norm(s) * np.linalg.norm(y):
            u = L.T @ s
            den2 = u @ u
            alp = np.sqrt(den1 / den2)
            if iteration == 1:
                L = alp * L
                u = alp * u
                den2, alp = den1, 1.0
            w = L @ u
            reltol = rnf if central else np.sqrt(rnf)
            if not np.all(np.abs(y - w) < reltol * np.maximum(np.abs(g), np.abs(gp))):
                w = y - alp * w
                u = u * (alp / den1)
                R = np.linalg.qr(L.T + np.outer(u, w), mode="r")
                L = (R * np.sign(np.diag(R))[:, None]).T
        x, f, g = xp, fp, gp
    return x


def _nlm_factor(lme: "LinearizedLME", L: np.ndarray, full: bool) -> np.ndarray:
    """nlme_increment's optif9 refinement of the factor after EM."""
    r = L.shape[0]
    const = 0.5 * lme.n * (1.0 + np.log(2 * np.pi) - np.log(lme.n))
    cols = [(i, j) for j in range(r) for i in range(j)]  # column-major upper part

    def to_L(theta):
        if not full:
            return np.diag(np.exp(-theta))
        U = np.diag(np.exp(theta[:r]))
        for k, (i, j) in enumerate(cols):
            U[i, j] = theta[r + k]
        return np.linalg.cholesky(np.linalg.inv(U.T @ U))

    def from_L(L):
        if not full:
            return -np.log(np.diag(L))
        U = np.linalg.cholesky(np.linalg.inv(L @ L.T)).T
        return np.concatenate([np.log(np.diag(U)), [U[i, j] for i, j in cols]])

    def fcn(theta):
        try:
            return -(lme.solve(to_L(theta))[0] + const)
        except np.linalg.LinAlgError:
            return np.inf

    return to_L(_optif9(fcn, from_L(L)))


def _penalty(B: np.ndarray, L: np.ndarray) -> float:
    """sum_i ||L^-1 b_i||^2 for B (m, r)."""
    return float(np.sum(np.linalg.solve(L, B.T) ** 2))


def pnls_nlme(model: GroupedModel, L, beta, B, full: bool, max_iter: int = 7, min_scale: float = 1e-3,
              tol: float = 1e-3):
    """nlme's PNLS step. B is (m, r) random effects on the parameter scale.

    Returns beta, B, the EM-updated factor L, the residuals and Jacobian at the
    solution (padded), the penalised objective, the iterations, and whether the
    step halving failed.
    """
    m, r, f = model.n_groups, model.n_random, model.n_fixed
    n_par = m * r + f
    nrdof = model.n_obs - n_par

    def evaluate(beta, B):
        eta = model.A @ beta + np.einsum("mpr,mr->mp", model.B, B)
        return model.mask * (model.y_pad - model.predict_padded(eta)), eta

    res, eta = evaluate(beta, B)
    J = model.jacobian_padded(eta) * model.mask[..., None]
    step = 1.0
    for iteration in range(1, max_iter + 1):
        lme = LinearizedLME(model, beta, B.T, res, J, reml=False)
        L = _nlm_factor(lme, _em_factor(lme, L, full), full)
        objective = float(np.sum(res * res)) + _penalty(B, L)
        _, beta_gls, _, _, u = lme.solve(L)
        predicted = lme._last[3]
        d_beta, d_B = beta_gls - beta, u @ L.T - B
        criterion = np.sqrt(nrdof * max(objective - predicted, 0.0) / (n_par * predicted))
        if criterion < tol:
            return beta, B, L, res, J, objective, iteration, False
        while True:
            if step < min_scale:
                return beta, B, L, res, J, objective, iteration, True
            new_beta, new_B = beta + step * d_beta, B + step * d_B
            try:
                new_res, new_eta = evaluate(new_beta, new_B)
                new_objective = float(np.sum(new_res * new_res)) + _penalty(new_B, L)
            except FloatingPointError:
                new_objective = np.inf
            step /= 2
            if new_objective < objective:
                break
        step = min(4 * step, 1.0)
        beta, B, res, eta = new_beta, new_B, new_res, new_eta
        J = model.jacobian_padded(eta) * model.mask[..., None]
    return beta, B, L, res, J, float(np.sum(res * res)) + _penalty(B, L), max_iter, False


def _nlme_coef(L: np.ndarray, full: bool) -> np.ndarray:
    """nlme's unconstrained reStruct coefficients (pdDiag / pdLogChol of the
    precision factor), used only for its convergence criterion."""
    if not full:
        return -np.log(np.maximum(np.diag(L), 1e-300))
    precision = np.linalg.inv(L @ L.T)
    R = np.linalg.cholesky(precision).T
    return np.concatenate([np.log(np.diag(R)), R[np.triu_indices(L.shape[0], 1)]])


def fit_lindstrom_bates_nlme(
    model: GroupedModel,
    beta0: np.ndarray,
    full_cov: bool = False,
    reml: bool = False,
    max_iter: int = 50,
    tolerance: float = 1e-5,
    pnls_tol: float = 1e-3,
    pnls_max_iter: int = 7,
    min_scale: float = 1e-3,
    verbose: int = 0,
) -> LBResult:
    """Lindstrom-Bates estimation following R nlme's iteration step by step.

    Defaults are nlmeControl()'s (maxIter, tolerance, pnlsTol, pnlsMaxIter,
    minScale). Returns the fixed and random effects of the last PNLS step and
    the likelihood of the model linearised there, as nlme does.
    """
    m, r = model.n_groups, model.n_random
    factor = CovarianceFactor(r, full_cov)
    beta = np.asarray(beta0, dtype=float).copy()
    B = np.zeros((m, r))
    eta = model.etas_all(beta)
    J = model.jacobian_padded(eta) * model.mask[..., None]
    res = model.mask * (model.y_pad - model.predict_padded(eta))
    L = initial_factor(model, J)  # nlme's Initialize
    converged = False
    pnls_failed = False
    iteration = 0
    prev_logl = None
    for iteration in range(1, max_iter + 1):
        # LME step: maximise the likelihood of the model linearised at the
        # current PNLS solution, from the current factor.
        lme = LinearizedLME(model, beta, B.T, res, J, reml)
        theta_start = factor.from_matrix(L)
        logl_start = lme.solve(L)[0]
        # restart: L-BFGS-B can stall with a variance collapsed to zero where
        # nlminb does not (seed 1502491072, linear_2: 18.6 lower); the restart
        # is kept only if it finds a higher likelihood.
        theta, _ = lme.maximize(factor, theta_start, tight=True, restart=True)
        gain = lme.solve(factor.to_matrix(theta))[0] - logl_start
        old_coef, L_lme = _nlme_coef(L, full_cov), factor.to_matrix(theta)
        if verbose:
            # Laid out as nlme's verbose output (its reStruct coefficients).
            print(f"**Iteration {iteration}\nLME step: Loglik: {lme.solve(L_lme)[0]:.7g}\n"
                  f"reStruct parameters: {np.array2string(_nlme_coef(L_lme, full_cov), precision=7)}")
        old_beta = beta
        beta, B, L, res, J, objective, n_pnls, failed = pnls_nlme(
            model, L_lme, beta, B, full_cov, pnls_max_iter, min_scale, pnls_tol)
        pnls_failed = pnls_failed or failed
        # nlme overwrites the reStruct with PNLS's EM-updated factor before
        # comparing, so the criterion is the change between successive PNLS
        # factors (with REML these differ from the LME step's optimum).
        new_coef = _nlme_coef(L, full_cov)
        if verbose:
            print(f"PNLS step: RSS = {objective:.7g}\n fixed effects: {np.array2string(beta, precision=7)}\n"
                  f" iterations: {n_pnls}")
        conv_fixed = np.max(np.abs(old_beta - beta) / np.where(np.abs(beta) < tolerance, 1.0, np.abs(beta)))
        conv_re = np.max(np.abs(old_coef - new_coef) / np.where(np.abs(new_coef) < tolerance, 1.0, np.abs(new_coef)),
                         initial=0.0)
        # The reported log-likelihood: the model linearised at this PNLS
        # solution, with PNLS's factor.
        logl = LinearizedLME(model, beta, B.T, res, J, reml).solve(L)[0]
        if verbose:
            print(f"Convergence crit.: fixed {conv_fixed:.6g} reStruct {conv_re:.6g} "
                  f"(LME step: {lme.last_result.nit} iterations, gain {gain:.3g})")
        # nlme stops when all criteria are met, or when the fixed effects are
        # and its LME step (nlminb) took a single iteration. pynlme's LME step
        # uses another optimizer: where a variance has collapsed, nlminb stops
        # on the flat likelihood while L-BFGS-B moves on to the bound and PNLS
        # moves the factor back, so the factor never settles. Equivalent there:
        # the fixed effects and the reported likelihood have converged.
        logl_settled = prev_logl is not None and abs(logl - prev_logl) <= tolerance * (abs(logl) + 1.0)
        prev_logl = logl
        if max(conv_fixed, conv_re) <= tolerance or (
                conv_fixed <= tolerance and (lme.last_result.nit <= 1 or logl_settled)):
            converged = True
            break
    if pnls_failed:
        import warnings

        warnings.warn("step halving factor reduced below minimum in PNLS step (as nlme's warning)",
                      RuntimeWarning, stacklevel=3)
    lme = LinearizedLME(model, beta, B.T, res, J, reml)
    logl, _, sigma2, A, _ = lme.solve(L)
    return LBResult(
        beta=beta, psi=sigma2 * L @ L.T, sigma2=sigma2, b=B.T.copy(), logl=logl,
        covb=sigma2 * np.linalg.inv(A), iterations=iteration, converged=converged,
    )
