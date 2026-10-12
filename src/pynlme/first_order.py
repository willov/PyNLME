"""
First-order approximations for nlmefit: ApproximationType "FO" and "FOCE".

The model is linearised in the random effects around b_hat_i, and the
resulting Gaussian marginal likelihood is maximised directly over beta, the
random-effects covariance and the error variance:

    y_i ~ N(f_i(beta, b_hat_i) - Z_i b_hat_i, Z_i Psi Z_i' + sigma^2 I)

with Z_i = d f_i / d b at b_hat_i. FO uses b_hat_i = 0; FOCE uses the
conditional modes of b_i given the current parameters. Unlike the
Lindstrom-Bates "LME" approximation, beta is optimised in the nonlinear
model rather than by GLS in the linearised one.

As elsewhere, Psi = sigma^2 L L' and b_i = L u_i, so the conditional modes
depend only on (beta, L), and sigma^2 is profiled out.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from . import lindstrom_bates
from .lindstrom_bates import CovarianceFactor, refine_beta0
from .model import GroupedModel


@dataclass
class FirstOrderResult:
    beta: np.ndarray
    psi: np.ndarray
    sigma2: float
    b: np.ndarray  # (r, m)
    logl: float
    covb: np.ndarray
    iterations: int
    converged: bool
    error_param: np.ndarray | None = None


def conditional_modes(
    model: GroupedModel,
    beta,
    L,
    U0,
    max_iter: int = 100,
    tol: float = 1e-12,
    weights=None,
):
    """u_i minimising ||w_i (y_i - f_i(beta, L u_i))||^2 + ||u_i||^2 for each group.

    Levenberg-Marquardt per group. The groups are independent, so they advance
    in lock-step: each round evaluates the model once for all groups that need a
    Jacobian and once for all trial steps (one call each with
    Vectorization="Full"). Every group takes the same steps as when solved alone.
    """
    m, r = model.n_groups, model.n_random
    if lindstrom_bates._kernels is not None:
        return _conditional_modes_kernel(model, beta, L, U0, max_iter, tol, weights)
    U = U0.copy()
    w = [1.0 if weights is None else weights[i] for i in range(m)]
    etas = model.etas_all(beta, (U @ L.T).T)
    fs = model.predict_many(np.arange(m), etas)
    res = [w[i] * (model.y[i] - fs[i]) for i in range(m)]
    S = np.array([res[i] @ res[i] + U[i] @ U[i] for i in range(m)])
    lam = np.full(m, 1e-3)
    outer = np.zeros(m, dtype=int)  # Levenberg-Marquardt iterations started
    need_jacobian = np.ones(m, dtype=bool)
    active = np.ones(m, dtype=bool)
    M = [None] * m
    g = [None] * m
    while active.any():
        # Groups starting an iteration: Jacobian, gradient and Gauss-Newton matrix.
        starting = np.flatnonzero(active & need_jacobian)
        if starting.size:
            Js = model.jacobian_many(starting, etas[starting])
            for k, i in enumerate(starting):
                H = _rows(w[i], Js[k]) @ model.B[i] @ L
                g[i] = H.T @ res[i] - U[i]
                M[i] = H.T @ H + np.eye(r)
                outer[i] += 1
                need_jacobian[i] = False
                if lam[i] >= 1e10:  # no damping left to try
                    active[i] = False
        trying = np.flatnonzero(active)
        if not trying.size:
            break
        steps = {
            i: np.linalg.solve(M[i] + lam[i] * np.diag(np.diag(M[i])), g[i])
            for i in trying
        }
        U_new = np.array([U[i] + steps[i] for i in trying])
        etas_new = np.array(
            [model.eta(i, beta, L @ u) for i, u in zip(trying, U_new, strict=True)]
        )
        fs_new = model.predict_many(trying, etas_new, allow_invalid=True)
        for k, i in enumerate(trying):
            if fs_new[k] is None:
                S_new = np.inf
            else:
                res_new = w[i] * (model.y[i] - fs_new[k])
                S_new = res_new @ res_new + U_new[k] @ U_new[k]
            if S_new > S[i]:
                lam[i] *= 10
                if lam[i] >= 1e10:  # no improving step: done
                    active[i] = False
                continue
            U[i], etas[i], res[i], S[i] = U_new[k], etas_new[k], res_new, S_new
            lam[i] = max(lam[i] / 10, 1e-12)
            # The marginal likelihood is differentiated numerically, so the
            # modes must be solved to near machine precision.
            step = steps[i]
            if (
                np.max(np.abs(step)) <= tol * (1 + np.max(np.abs(U[i])))
                or outer[i] >= max_iter
            ):
                active[i] = False
            else:
                need_jacobian[i] = True
    return U


def _conditional_modes_kernel(model, beta, L, U0, max_iter, tol, weights):
    """conditional_modes by the Rust kernel, with the model as Python callbacks."""
    lengths = [len(y) for y in model.y]
    w = (
        np.ones(sum(lengths))
        if weights is None
        else np.concatenate(
            [np.broadcast_to(wt, (n,)) for wt, n in zip(weights, lengths, strict=True)]
        )
    )
    return lindstrom_bates._kernels.conditional_modes(
        lambda groups, etas: model.predict_flat(groups, etas),
        lambda groups, etas: model.jacobian_rows(groups, etas),
        np.ascontiguousarray(model.A, dtype=float),
        np.ascontiguousarray(model.B, dtype=float),
        np.concatenate(model.y).astype(float),
        np.ascontiguousarray(w, dtype=float),
        lengths,
        np.ascontiguousarray(beta, dtype=float),
        np.ascontiguousarray(L, dtype=float),
        np.ascontiguousarray(U0, dtype=float),
        max_iter,
        tol,
    )


def _rows(w, J):
    return J if np.isscalar(w) else w[:, None] * J


def linearized_marginal(
    model: GroupedModel, beta, L, U, sigma2=None, weights=None, with_parts=True
):
    """Log-likelihood of the model linearised at b_i = L u_i.

    With observation weights w (error SD proportional to 1/w), residuals and
    Jacobians are weighted and sum(log w) is added, as MATLAB's nlmefit does.
    With sigma2=None the error variance is profiled out. Returns the
    log-likelihood, sigma^2, and per-group (X_i, H_i) for standard errors
    (empty unless with_parts).
    """
    n = model.n_obs
    logdet, quad = 0.0, 0.0
    parts = []
    groups = np.arange(model.n_groups)
    etas = np.array([model.eta(i, beta, L @ U[i]) for i in groups])
    kernels = lindstrom_bates._kernels
    if kernels is not None and not with_parts:
        values, valid = model.predict_flat(groups, etas)
        if not valid.all():
            raise FloatingPointError("modelfun returned non-finite values")
        lengths = [len(y) for y in model.y]
        w = (
            np.ones(n)
            if weights is None
            else np.concatenate(
                [
                    np.broadcast_to(wt, (k,))
                    for wt, k in zip(weights, lengths, strict=True)
                ]
            )
        )
        logdet, quad = kernels.linearized_marginal(
            np.ascontiguousarray(model.jacobian_rows(groups, etas)),
            values,
            np.concatenate(model.y).astype(float),
            np.ascontiguousarray(w, dtype=float),
            lengths,
            np.ascontiguousarray(model.B, dtype=float),
            np.ascontiguousarray(L, dtype=float),
            np.ascontiguousarray(U, dtype=float),
        )
        groups = []  # the per-group loop below is done
    else:
        Js = model.jacobian_many(groups, etas)
        fs = model.predict_many(groups, etas)
    for i in groups:
        wt = 1.0 if weights is None else weights[i]
        J = _rows(wt, Js[i])
        H = J @ model.B[i] @ L
        w = wt * (model.y[i] - fs[i]) + H @ U[i]
        M = np.eye(model.n_random) + H.T @ H
        C = np.linalg.cholesky(M)
        logdet += 2 * np.sum(np.log(np.diag(C)))
        Hw = np.linalg.solve(C, H.T @ w)
        quad += w @ w - Hw @ Hw
        parts.append((J @ model.A[i], H))
    if sigma2 is None:
        sigma2 = quad / n
        logl = -0.5 * n * (1 + np.log(2 * np.pi * sigma2)) - 0.5 * logdet
    else:
        logl = -0.5 * (n * np.log(2 * np.pi * sigma2) + logdet + quad / sigma2)
    if weights is not None:
        logl += float(sum(np.sum(np.log(wt)) for wt in weights))
    return logl, sigma2, parts


def fit_first_order(
    model: GroupedModel,
    beta0,
    conditional: bool,
    full_cov: bool = False,
    refine: bool = True,
    max_iter: int = 200,
    tol_x: float = 1e-6,
    tol_fun: float = 1e-6,
    verbose: int = 0,
    error_model: str = "constant",
    error_param0=None,
) -> FirstOrderResult:
    """FO/FOCE estimation.

    For "proportional" and "combined" error models, as MATLAB: optimise with
    fixed observation weights 1/g(f), refresh the weights from the fit
    (at b = 0 for FO, at the conditional modes for FOCE), and repeat until
    they settle.
    """
    from .lindstrom_bates import LinearizedLME, initial_factor, update_weights

    m, r, f = model.n_groups, model.n_random, model.n_fixed
    factor = CovarianceFactor(r, full_cov)
    beta = np.asarray(beta0, dtype=float).copy()
    if refine:
        beta = refine_beta0(model, beta)

    # Starting covariance: an LME step at the start values, as nlmefit's LME.
    eta0 = model.etas_all(beta)
    J0 = model.jacobian_padded(eta0)
    res0 = model.mask * (model.y_pad - model.predict_padded(eta0))
    lme0 = LinearizedLME(model, beta, np.zeros((r, m)), res0, J0, reml=False)
    theta, _ = lme0.maximize(
        factor, factor.from_matrix(initial_factor(model, J0)), tight=False
    )

    state = {"U": np.zeros((m, r)), "weights": None}

    def modes(beta_x, L):
        if not conditional:
            return np.zeros((m, r))
        return conditional_modes(model, beta_x, L, state["U"], weights=state["weights"])

    def objective(x):
        beta_x, L = x[:f], factor.to_matrix(x[f:])
        try:
            U = modes(beta_x, L)
            logl, _, _ = linearized_marginal(
                model, beta_x, L, U, weights=state["weights"], with_parts=False
            )
        except (np.linalg.LinAlgError, FloatingPointError):
            return np.inf
        if not np.isfinite(logl):
            return np.inf
        state["U"] = U
        return -logl

    def gradient(x):
        g = np.empty_like(x)
        for k in range(x.size):
            h = 1e-5 * max(abs(x[k]), 1.0)
            up, down = x.copy(), x.copy()
            up[k] += h
            down[k] -= h
            g[k] = (objective(up) - objective(down)) / (2 * h)
        return g

    bounds = [(None, None)] * f + factor.bounds

    def optimise(x0):
        result = optimize.minimize(
            objective,
            x0,
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-13, "gtol": 1e-7},
        )
        converged = bool(result.success)
        if not converged and "ABNORMAL" in str(result.message):
            # The line search could not improve further: usually the optimum
            # at the precision of the numerical gradient. Confirm with a restart.
            retry = optimize.minimize(
                objective,
                result.x,
                jac=gradient,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": 1e-13, "gtol": 1e-7},
            )
            converged = retry.fun > result.fun - 1e-6 * (1 + abs(result.fun))
            if retry.fun < result.fun:
                result = retry
        return result, converged

    x = np.concatenate([beta, theta])
    ab = np.array([1.0, 1.0] if error_param0 is None else error_param0, dtype=float)
    reweight = error_model in ("proportional", "combined")
    for _ in range(max_iter if reweight else 1):
        result, converged = optimise(x)
        if verbose:
            print(
                f"{'FOCE' if conditional else 'FO'}: {result.message} after "
                f"{result.nit} iterations"
            )
        step = np.max(np.abs(result.x - x) / np.maximum(np.abs(x), 1e-8))
        x = result.x
        if not reweight:
            break
        beta, L = x[:f], factor.to_matrix(x[f:])
        U = modes(beta, L)
        fb = np.concatenate(
            model.predict_many(
                np.arange(m), [model.eta(i, beta, L @ U[i]) for i in range(m)]
            )
        )
        w_flat, ab = update_weights(error_model, np.concatenate(model.y), fb, ab)
        state["weights"] = np.split(w_flat, np.cumsum([len(yi) for yi in model.y])[:-1])
        if step < tol_x:
            break

    beta, L = x[:f], factor.to_matrix(x[f:])
    U = modes(beta, L)
    logl, sigma2, parts = linearized_marginal(
        model, beta, L, U, weights=state["weights"]
    )
    # Standard errors from the GLS information of the linearised model.
    fim = (
        sum(X.T @ np.linalg.solve(np.eye(len(X)) + H @ H.T, X) for X, H in parts)
        / sigma2
    )
    return FirstOrderResult(
        beta=beta,
        psi=sigma2 * L @ L.T,
        sigma2=sigma2,
        b=L @ U.T,
        logl=logl,
        covb=np.linalg.inv(fim),
        iterations=result.nit,
        converged=converged,
        error_param=ab,
    )
