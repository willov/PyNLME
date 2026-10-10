"""
MATLAB nlmefit's Lindstrom-Bates iteration (ApproximationType "LME"/"RELME"),
step by step, for compat="matlab".

Follows the algorithm of MATLAB R2023b's nlmefit (alternating algorithm,
Pinheiro & Bates 2000, p. 313), with its choices:

- start: RefineBeta0 by Levenberg-Marquardt on the model without random
  effects; Delta_0 = diag(0.375 sqrt(sum_obs Z^2 / M)) (Bates & Pinheiro 1998);
- PNLS: Gauss-Newton on beta and b for fixed Delta, with step halving, to the
  tolerances tolX * 10^tolFac, tolF * 10^tolFac; random effects whose precision
  exceeds 1e8 times the smallest are held fixed;
- LME step: fminsearch (Nelder-Mead) on the profiled (restricted) likelihood
  of the linearised model, with Delta'Delta parameterised by its matrix
  logarithm ("logm") or Delta by its entries ("chol"). Its tolerances stay at
  10^2 times TolX/TolFun (nlmefit tightens a differently named variable);
- tolFac goes from 2 to 0 when both the likelihood and the PNLS estimates
  change by less than the current tolerances;
- error models: weights 1/|f| (proportional) or 1/|a + b|f|| (combined, a and
  b by fminsearch) recomputed after each LME step;
- derivatives by forward differences, step eps^(1/3) * eta (statjacobian).

Returned beta and b are those of the last PNLS step; the covariance, sigma^2
(c_1^2 / N) and log-likelihood come from the last LME step, as in nlmefit.
"""

import logging
import warnings

import numpy as np
from scipy import linalg

from .lindstrom_bates import LBResult, error_ab
from .model import GroupedModel

_EPS = np.finfo(float).eps
_log = logging.getLogger("pynlme")


def _collapsed(Delta) -> bool:
    """A random-effects variance (an eigenvalue of Psi/sigma^2) at zero."""
    return Delta is None or bool(np.max(np.linalg.eigvalsh(Delta.T @ Delta)) > 1e10)


def _spacing(x: float) -> float:
    """MATLAB's eps(x)."""
    return float(np.spacing(abs(x)))


def fminsearch(fun, x0, tol_x: float = 1e-4, tol_fun: float = 1e-4, max_iter=None, max_fun_evals=None):
    """Nelder-Mead as MATLAB's fminsearch (Lagarias et al. 1998): initial
    simplex 5% (0.00025 for zero entries), reflection 1, expansion 2,
    contraction 0.5, do_shrink 0.5; stops when the function values and the
    simplex both lie within the tolerances."""
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    max_iter = 200 * n if max_iter is None else max_iter
    max_fun_evals = 200 * n if max_fun_evals is None else max_fun_evals
    REFLECT, EXPAND, CONTRACT, SHRINK = 1.0, 2.0, 0.5, 0.5  # Nelder-Mead coefficients
    simplex = np.empty((n + 1, n))
    values = np.empty(n + 1)
    simplex[0] = x0
    values[0] = fun(x0)
    for j in range(n):
        vertex = x0.copy()
        vertex[j] = (1 + 0.05) * vertex[j] if vertex[j] != 0 else 0.00025
        simplex[j + 1] = vertex
        values[j + 1] = fun(vertex)
    order = np.argsort(values, kind="stable")
    simplex, values = simplex[order], values[order]
    n_evals, iterations = n + 1, 1
    while n_evals < max_fun_evals and iterations < max_iter:
        if (np.max(np.abs(values[0] - values[1:])) <= max(tol_fun, 10 * _spacing(values[0]))
                and np.max(np.abs(simplex[1:] - simplex[0])) <= max(tol_x, 10 * _spacing(np.max(simplex[0])))):
            break
        centroid = simplex[:n].mean(axis=0)
        reflected = (1 + REFLECT) * centroid - REFLECT * simplex[n]
        f_reflected = fun(reflected)
        n_evals += 1
        do_shrink = False
        if f_reflected < values[0]:
            expanded = (1 + REFLECT * EXPAND) * centroid - REFLECT * EXPAND * simplex[n]
            f_expanded = fun(expanded)
            n_evals += 1
            simplex[n], values[n] = (expanded, f_expanded) if f_expanded < f_reflected else (reflected, f_reflected)
        elif f_reflected < values[n - 1]:
            simplex[n], values[n] = reflected, f_reflected
        elif f_reflected < values[n]:
            outside = (1 + CONTRACT * REFLECT) * centroid - CONTRACT * REFLECT * simplex[n]
            f_outside = fun(outside)
            n_evals += 1
            if f_outside <= f_reflected:
                simplex[n], values[n] = outside, f_outside
            else:
                do_shrink = True
        else:
            inside = (1 - CONTRACT) * centroid + CONTRACT * simplex[n]
            f_inside = fun(inside)
            n_evals += 1
            if f_inside < values[n]:
                simplex[n], values[n] = inside, f_inside
            else:
                do_shrink = True
        if do_shrink:
            for j in range(1, n + 1):
                simplex[j] = simplex[0] + SHRINK * (simplex[j] - simplex[0])
                values[j] = fun(simplex[j])
            n_evals += n
        order = np.argsort(values, kind="stable")
        simplex, values = simplex[order], values[order]
        iterations += 1
    converged = n_evals < max_fun_evals and iterations < max_iter
    return simplex[0], values[0], converged


class _Covariance:
    """theta <-> Delta for nlmefit's CovParameterization and CovPattern."""

    def __init__(self, r: int, full: bool, parametrization: str):
        if parametrization not in ("logm", "chol"):
            raise NotImplementedError(f"CovParameterization '{parametrization}' is not implemented")
        self.r = r
        self.kind = parametrization
        pattern = np.triu(np.ones((r, r), bool)) if full else np.eye(r, dtype=bool)
        # MATLAB's find(triu(pattern)): column-major positions.
        self.rows, self.cols = np.nonzero(pattern.T)[::-1]

    @property
    def n(self) -> int:
        return self.rows.size

    def to_theta(self, Delta: np.ndarray) -> np.ndarray:
        if self.kind == "chol":
            return Delta[self.rows, self.cols].copy()
        S = np.real(linalg.logm(Delta.T @ Delta))
        return S[self.rows, self.cols].copy()

    def to_delta(self, theta: np.ndarray) -> np.ndarray | None:
        M = np.zeros((self.r, self.r))
        M[self.rows, self.cols] = theta
        if self.kind == "chol":
            return M
        S = M + np.triu(M, 1).T
        try:
            return np.linalg.cholesky(linalg.expm(S)).T
        except np.linalg.LinAlgError:
            return None


class _Linearised:
    """The model linearised at (beta, b): weighted [Z | X w] per group, padded."""

    def __init__(self, model: GroupedModel, beta, b, weights):
        self.model = model
        eta = model.etas_all(beta, b.T)
        f = model.predict_padded(eta)
        J = model.jacobian_padded_matlab(eta, f)
        X = np.einsum("mnp,mpf->mnf", J, model.A)
        Z = np.einsum("mnp,mpr->mnr", J, model.B)
        w = model.y_pad - f + np.einsum("mnf,f->mn", X, beta) + np.einsum("mnr,mr->mn", Z, b)
        W = weights[..., None]
        self.f = f
        ZXw = np.concatenate([W * Z, W * np.concatenate([X, w[..., None]], axis=2)], axis=2)
        # The QR factors below depend on [Z_i X_i w_i] only through inner
        # products: replace each group's rows by its square QR factor once.
        if ZXw.shape[1] > ZXw.shape[2]:
            ZXw = np.linalg.qr(ZXw, mode="r")
        r = Z.shape[2]
        self.Z = np.ascontiguousarray(ZXw[..., :r])
        self.Xw = np.ascontiguousarray(ZXw[..., r:])

    def decompose(self, Delta):
        """Per-group QR of [Z_i X_i w_i; Delta 0]: R_b (m,r,r), R_bx (m,r,q+1),
        and the combined fixed-effects factor R_x (q+1, q+1)."""
        kernels = _kernels()
        if kernels is not None:
            return kernels.delta_qr(self.Z, self.Xw, np.ascontiguousarray(Delta))
        m, n, r = self.Z.shape
        q1 = self.Xw.shape[2]
        top = np.concatenate([self.Z, self.Xw], axis=2)
        bottom = np.concatenate([np.broadcast_to(Delta, (m, r, r)), np.zeros((m, r, q1))], axis=2)
        A = np.concatenate([top, bottom], axis=1)
        if A.shape[1] < r + q1:
            A = np.pad(A, ((0, 0), (0, r + q1 - A.shape[1]), (0, 0)))
        R = np.linalg.qr(A, mode="r")
        R_x = np.linalg.qr(R[:, r : r + q1, r:].reshape(-1, q1), mode="r")
        return R[:, :r, :r], R[:, :r, r:], R_x

    def negloglik(self, Delta, reml: bool):
        """nlmefit's LMEnegloglikelihood (varying part of P&B eq. 2.21/2.23)."""
        if Delta is None or not np.all(np.isfinite(Delta)):
            return 1 / _EPS, None, None
        q = self.Xw.shape[2] - 1
        N = self.model.n_obs
        kernels = _kernels()
        if kernels is not None:
            value, resid_norm, R_x = kernels.matlab_negloglik(self.Z, self.Xw, np.ascontiguousarray(Delta), float(N), reml)
            return value, resid_norm, np.triu(R_x[:q, :q])
        R_b, _, R_x = self.decompose(Delta)
        resid_norm = abs(R_x[q, q])
        det_rb = np.prod(np.diagonal(R_b, axis1=1, axis2=2), axis=1)
        log_ratio = np.sum(np.log(np.abs(np.linalg.det(Delta) / det_rb)))
        if reml:
            value = (N - q) * np.log(resid_norm) + np.log(abs(np.prod(np.diag(R_x[:q, :q])))) - log_ratio
        else:
            value = N * np.log(resid_norm) - log_ratio
        return value, resid_norm, np.triu(R_x[:q, :q])


def _kernels():
    """The Rust kernels if installed and enabled (shared switch with lindstrom_bates)."""
    from . import lindstrom_bates

    return lindstrom_bates._kernels


def _residual_sse(model, beta, b, weights, Delta):
    eta = model.etas_all(beta, b.T)
    res = weights * (model.y_pad - model.predict_padded(eta))
    return float(np.sum(res * res) + np.sum((b @ Delta.T) ** 2))


def _pnls(model, Delta, beta, b, weights, tol_fun, tol_x, max_iter):
    """nlmefit's GNfitPNLS. b is (m, r)."""
    m, r = b.shape
    q = beta.size
    frozen = np.diag(Delta) > np.min(np.diag(Delta)) * 1e8
    sse = _residual_sse(model, beta, b, weights, Delta)
    b_new = np.zeros((m, r))
    iteration = 0
    while iteration < max_iter:
        lin = _Linearised(model, beta, b, weights)
        R_b, R_bx, R_x = lin.decompose(Delta)
        beta_new = linalg.solve_triangular(R_x[:q, :q], R_x[:q, q])
        c = R_bx[:, :, q] - R_bx[:, :, :q] @ beta_new
        keep = ~frozen
        for i in range(m):
            b_new[i, keep] = linalg.solve_triangular(R_b[i][np.ix_(keep, keep)], c[i, keep])
        sse_new = _residual_sse(model, beta_new, b_new, weights, Delta)
        step = np.concatenate([beta_new - beta, (b_new - b).ravel()])
        while sse_new > sse:
            if np.max(np.abs(step)) < _EPS:
                warnings.warn("Unable to decrease the SSE in the PNLS step", RuntimeWarning, stacklevel=4)
                sse_new = sse
                break
            step = step / 2
            beta_new = beta + step[:q]
            b_new = b + step[q:].reshape(m, r)
            sse_new = _residual_sse(model, beta_new, b_new, weights, Delta)
        sse_diff, max_step = abs(sse_new - sse), np.max(np.abs(step))
        beta, b, sse = beta_new.copy(), b_new.copy(), sse_new
        if sse_diff <= tol_fun and max_step <= tol_x:
            break
        iteration += 1
    if iteration >= max_iter:
        warnings.warn("Iteration limit exceeded in the PNLS step", RuntimeWarning, stacklevel=4)
    return beta, b


def _lmfit(model: GroupedModel, beta, tol_fun, tol_x, max_iter):
    """nlmefit's LMfit: Levenberg-Marquardt for beta with b = 0."""
    zeros = np.zeros((model.n_groups, model.n_random))
    y = model.y_pad[model.mask]

    def fitted(beta):
        return model.predict_padded(model.etas_all(beta, zeros.T))

    lam = 0.01
    sqrteps = np.sqrt(_EPS)
    f = fitted(beta)
    res = y - f[model.mask]
    sse = res @ res
    step = np.zeros_like(beta)
    for _ in range(max_iter):
        beta_old, sse_old = beta.copy(), sse
        J = np.einsum("mnp,mpf->mnf", model.jacobian_padded_matlab(model.etas_all(beta, zeros.T), f), model.A)
        J = J[model.mask]
        d = np.sum(J * J, axis=0)

        def lm_step(lam, J=J, d=d, res=res, beta=beta):
            h = np.any(np.vstack([J, np.diag(np.sqrt(lam * d))]) != 0, axis=0)
            out = np.zeros_like(beta)
            if np.any(h):
                Jp = np.vstack([J[:, h], np.diag(np.sqrt(lam * d[h]))])
                out[h] = np.linalg.lstsq(Jp, np.concatenate([res, np.zeros(h.sum())]), rcond=None)[0]
            return out

        step = lm_step(lam)
        beta = beta_old + step
        f = fitted(beta)
        res_new = y - f[model.mask]
        sse = res_new @ res_new
        give_up = False
        if sse < sse_old:
            lam = max(0.1 * lam, _EPS)
        else:
            while sse >= sse_old:
                lam *= 10
                if lam > 1e16:
                    give_up = True
                    break
                step = lm_step(lam)
                beta = beta_old + step
                f = fitted(beta)
                res_new = y - f[model.mask]
                sse = res_new @ res_new
        res = res_new
        if np.linalg.norm(step) < tol_x * (sqrteps + np.linalg.norm(beta)) and abs(sse - sse_old) <= tol_fun * sse:
            return beta
        if give_up:
            warnings.warn("Unable to decrease the SSE in the RefineBeta0 step", RuntimeWarning, stacklevel=3)
            return beta
    warnings.warn("Iteration limit exceeded in the RefineBeta0 step", RuntimeWarning, stacklevel=3)
    return beta


def fit_lindstrom_bates_matlab(
    model: GroupedModel,
    beta0: np.ndarray,
    full_cov: bool = False,
    reml: bool = False,
    refine: bool = True,
    max_iter: int = 200,
    tol_x: float = 1e-4,
    tol_fun: float = 1e-4,
    error_model: str = "constant",
    error_param0=None,
    parametrization: str = "logm",
    x_first: float = 1.0,
    verbose: int = 0,
) -> LBResult:
    """nlmefit's alternating algorithm. ``x_first`` is X(1), which MATLAB uses
    in TolFun = max(TolFun, sqrt(eps(X(1))))."""
    m, r = model.n_groups, model.n_random
    beta = np.asarray(beta0, dtype=float).copy()
    q = beta.size
    N = model.n_obs
    tol_f = max(tol_fun, np.sqrt(_spacing(x_first)))
    tol_x = max(tol_x, np.sqrt(_spacing(beta[0])))
    weights = model.mask.astype(float)
    ab = None if error_param0 is None else np.atleast_1d(np.asarray(error_param0, dtype=float))
    if error_model == "combined" and ab is None:
        ab = np.array([1.0, 1.0])

    if refine:
        beta = _lmfit(model, beta, tol_f, tol_x, max_iter * q)

    b = np.zeros((m, r))
    eta = model.etas_all(beta, b.T)
    J = model.jacobian_padded_matlab(eta)
    Z = np.einsum("mnp,mpr->mnr", J, model.B)
    Delta = np.diag(0.375 * np.sqrt(np.sum(Z * Z, axis=(0, 1)) / m))
    cov = _Covariance(r, full_cov, parametrization)
    theta = cov.to_theta(Delta)
    Delta = cov.to_delta(theta)

    tol_exp = 2
    lme_tol = {"tol_fun": tol_f * 100.0, "tol_x": tol_x * 100.0}  # never tightened (see module docstring)
    lme_max_iter = (cov.n + q) * max_iter
    pnls_max_iter = (r * m + q) * max_iter
    prev_objective = np.inf
    converged = False
    collapsed = False
    iteration = 0
    lin = None
    while iteration < max_iter:
        iteration += 1
        beta_new, b_new = _pnls(model, Delta, beta, b, weights, tol_f * 10.0**tol_exp, tol_x * 10.0**tol_exp,
                                pnls_max_iter)
        largest_change = max(np.max(np.abs(beta_new - beta)), np.max(np.abs(b_new - b), initial=0.0))
        beta, b = beta_new, b_new

        lin = _Linearised(model, beta, b, weights)

        def objective(theta, lin=lin):
            return lin.negloglik(cov.to_delta(theta), reml)[0]

        kernels = _kernels()
        if kernels is not None and (cov.kind == "chol" or not full_cov):
            theta_hat, objective_value, ok = kernels.matlab_lme_step(
                lin.Z, lin.Xw, [float(v) for v in theta], [int(i) for i in cov.rows], [int(j) for j in cov.cols],
                cov.kind == "chol", float(N), reml, lme_tol["tol_x"], lme_tol["tol_fun"], int(lme_max_iter))
            theta_hat = np.asarray(theta_hat)
        else:
            theta_hat, objective_value, ok = fminsearch(objective, theta, max_iter=lme_max_iter, max_fun_evals=np.inf,
                                               **lme_tol)
        if not ok:
            warnings.warn("Iteration limit exceeded in the LME step", RuntimeWarning, stacklevel=3)

        f = lin.f
        if error_model == "proportional":
            with np.errstate(divide="ignore"):
                w = 1.0 / np.abs(f)
            weights = np.where(np.isinf(w), 1.0, w) * model.mask
        elif error_model == "combined":
            y, fm = model.y_pad[model.mask], f[model.mask]
            ab = np.abs(fminsearch(lambda v, y=y, fm=fm: error_ab(v, y, fm), ab)[0])
            with np.errstate(divide="ignore"):
                w = 1.0 / np.abs(ab[0] + ab[1] * np.abs(f))
            weights = np.where(np.isinf(w), 1.0, w) * model.mask

        theta = theta_hat
        Delta = cov.to_delta(theta)
        # A random-effects variance at zero: the likelihood is flat along it,
        # and where fminsearch stops there is decided by rounding (in MATLAB
        # too: its result changes with the number of BLAS threads).
        collapsed = collapsed or _collapsed(Delta)
        objective_change = abs(prev_objective - objective_value)
        prev_objective = objective_value
        if verbose:
            print(f"ALT {iteration - 1}: beta={np.array2string(beta, precision=10)} theta={np.array2string(theta, precision=10)} "
                  f"-logl(varying part)={objective_value:.10g} tolFac={tol_exp}")
        if objective_change <= tol_f * 10.0**tol_exp and largest_change <= tol_x * 10.0**tol_exp:
            if tol_exp <= 0:
                converged = True
                break
            tol_exp -= 1

    if _collapsed(Delta):
        warnings.warn(
            "A random-effects variance was estimated as zero. The likelihood is flat along it, so the "
            "estimates are not well determined: where MATLAB's (and this) iteration stops depends on "
            "rounding. Consider removing that random effect (REParamsSelect).",
            RuntimeWarning, stacklevel=3)
    elif collapsed:
        _log.info("nlmefit: a random-effects variance was zero during the iterations (flat likelihood); "
                  "the path there depends on rounding, in MATLAB too")
    objective_value, resid_norm, R_x = lin.negloglik(Delta, reml)
    dof = N - q if reml else N
    const = (np.log(dof) - np.log(2 * np.pi) - 1) * dof / 2
    logl = const - objective_value + float(np.sum(np.log(weights[model.mask])))
    # MATLAB reports sigma^2 = c_1^2 / N for both LME and RELME. nlmefit()
    # rescales RELME results to that convention itself, so return the RSS/(N-q)
    # form it expects.
    sigma2 = resid_norm**2 / dof
    Delta_inv = np.linalg.inv(Delta)
    R00_inv = np.linalg.inv(R_x)
    error_param = None
    if error_model == "combined":
        error_param = ab
    return LBResult(
        beta=beta, psi=sigma2 * Delta_inv @ Delta_inv.T, sigma2=sigma2, b=b.T.copy(), logl=logl,
        covb=sigma2 * R00_inv @ R00_inv.T, iterations=iteration, converged=converged, error_param=error_param,
    )
