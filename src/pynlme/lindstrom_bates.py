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

from dataclasses import dataclass

import numpy as np
from scipy import optimize

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
        self.Z = np.pad(Z, pad)
        self.Xw = np.pad(np.concatenate([X, w[:, :, None]], axis=2), pad)
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
