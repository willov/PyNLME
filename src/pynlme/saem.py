"""
Stochastic Approximation EM (SAEM) for nlmefitsa.

The algorithm of Delyon, Lavielle & Moulines (1999) and Kuhn & Lavielle (2005),
as implemented in MATLAB's nlmefitsa and R's saemix. The latent variables are
the individual parameters phi_i = eta_i[R] (transformed scale) of the
parameters R that have random effects:

    phi_i ~ N(A_i[R] beta, Omega),   y_ij = f(g^{-1}(eta_i), x_ij) + g_ij e_ij

with error standard deviation g = a (constant), b|f| (proportional) or
a + b|f| (combined). Each iteration
1. simulates phi_i with Metropolis-Hastings (three kernels: draws from the
   prior, single-coordinate random walks, full-vector random walks),
2. moves the sufficient statistics towards their values at the new samples
   (step size 1 in the first two phases, 1/k in the third),
3. maximises the complete-data likelihood given the statistics: closed form
   for beta (GLS), Omega and the constant/proportional error parameter;
   (a, b) of the combined model are refitted to the current samples, as
   MATLAB does; fixed effects of parameters without random effects get a
   Gauss-Newton minimisation followed by a stochastic-approximation step.

During the first phase (simulated annealing) variances may shrink by at most a
factor TAU per iteration, which keeps the sampler exploring.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy import stats as sps

from .model import GroupedModel

TAU = 0.97  # simulated-annealing decay per iteration, as saemix's alpha.sa
TARGET_ACCEPT = 0.4
_EPS = np.finfo(float).eps


@dataclass
class SAEMResult:
    beta: np.ndarray
    psi: np.ndarray
    errorparam: np.ndarray  # (a,), (b,) or (a, b) as MATLAB's stats.errorparam
    b: np.ndarray  # (r, m) conditional means of the random effects
    logl: float | None
    covb: np.ndarray | None
    iterations: int


class ErrorModel:
    """Error standard deviation g(f) and per-row negative log-likelihoods.

    Rows are segments of flat observation arrays starting at ``starts``.
    """

    def __init__(self, kind: str, params):
        self.kind = kind
        self.a, self.b = params

    @property
    def errorparam(self) -> np.ndarray:
        if self.kind == "combined":
            return np.array([self.a, self.b])
        return np.array([self.b if self.kind == "proportional" else self.a])

    def sd(self, f: np.ndarray) -> np.ndarray:
        if self.kind == "proportional":
            return np.maximum(self.b * np.abs(f), _EPS)
        if self.kind == "combined":
            return np.maximum(self.a + self.b * np.abs(f), _EPS)
        return np.full_like(f, self.a)

    def nll(self, y, f, starts) -> np.ndarray:
        """0.5 sum(((y - f)/g)^2) + sum(log g) per row (no 2*pi constant)."""
        g = self.sd(f)
        terms = 0.5 * ((y - f) / g) ** 2 + np.log(g)
        out = np.add.reduceat(terms, starts)
        return np.where(np.isfinite(out), out, np.inf)

    def statistic(self, y, f) -> float:
        """Sufficient statistic for the constant/proportional parameter."""
        if self.kind == "proportional":
            return float(np.sum((y - f) ** 2 / np.maximum(_EPS, f**2)))
        return float(np.sum((y - f) ** 2))


def _error_ab(ab, y, f):
    g = abs(ab[0]) + abs(ab[1]) * np.abs(f)
    return float(np.sum(0.5 * ((y - f) / g) ** 2 + np.log(g)))


def _random_effect_rows(model: GroupedModel) -> np.ndarray:
    """Indices of the parameters with random effects (selection designs only)."""
    B = model.B
    if not np.allclose(B, B[0]):
        raise NotImplementedError("nlmefitsa does not support REGroupDesign yet")
    B0 = B[0]
    if not (np.all(np.isin(B0, (0.0, 1.0))) and np.all(B0.sum(axis=0) == 1) and np.all(B0.sum(axis=1) <= 1)):
        raise NotImplementedError("nlmefitsa supports only REParamsSelect-style random-effects designs")
    return np.argmax(B0, axis=0)


class _SAEM:
    def __init__(self, model: GroupedModel, beta0, full_cov: bool, rng, n_chains: int, err: ErrorModel):
        self.model = model
        self.rng = rng
        self.full_cov = full_cov
        self.err = err
        self.m = model.n_groups
        # Several chains per subject are handled as copies of the subjects
        # ("rows"); sufficient statistics are averaged over the chains. All
        # rows' observations are kept in flat arrays.
        self.n_chains = n_chains
        self.sub = np.tile(np.arange(self.m), n_chains)
        self.M = self.m * n_chains
        lengths = np.array([len(model.y[i]) for i in self.sub])
        self.starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self.row_of_obs = np.repeat(np.arange(self.M), lengths)
        self.y = np.concatenate([model.y[i] for i in self.sub])
        self.n = model.n_obs
        self.R = _random_effect_rows(model)
        self.F = np.setdiff1d(np.arange(model.A.shape[1]), self.R)
        self.r = len(self.R)
        A = model.A
        in_R = np.any(A[:, self.R, :] != 0, axis=(0, 1))
        in_F = np.any(A[:, self.F, :] != 0, axis=(0, 1)) if self.F.size else np.zeros(A.shape[2], bool)
        if np.any(in_R & in_F):
            raise NotImplementedError(
                "nlmefitsa does not support a fixed effect shared by parameters with and without random effects"
            )
        self.cols_R = np.flatnonzero(~in_F)  # updated in closed form
        self.cols_F = np.flatnonzero(in_F)  # updated by Gauss-Newton
        self.A_R = A[:, self.R, :]  # (m, r, f)
        self.A_rows = A[self.sub]  # (M, p, f)
        self.beta = np.asarray(beta0, dtype=float).copy()
        # Parameters without random effects but with fixed effects. During the
        # annealing phase they are sampled too, with a pseudo-variance that
        # shrinks to ~0 (as MATLAB and saemix do), so the sampler explores
        # them globally; afterwards they are fixed and optimised.
        self.Fx = self.F[np.any(A[:, self.F, :] != 0, axis=(0, 2))] if self.F.size else self.F
        self.set_extended(False)

    def set_extended(self, extended: bool) -> None:
        self.extended = extended and self.Fx.size > 0
        self.latent = np.concatenate([self.R, self.Fx]) if self.extended else self.R
        self.d = len(self.latent)
        self.A_latent = self.model.A[:, self.latent, :]

    # -- model evaluation ---------------------------------------------------
    def eta(self, i, phi):
        eta = self.model.A[i] @ self.beta
        eta[self.R] = phi
        return eta

    def etas(self, Phi) -> np.ndarray:
        """Transformed parameters for every row, shape (M, p)."""
        etas = self.A_rows @ self.beta
        etas[:, self.latent] = Phi
        return etas

    def predict(self, Phi):
        """Flat predictions for all rows and a validity flag per row."""
        return self.model.predict_flat(self.sub, self.etas(Phi))

    def nll(self, F, valid=None) -> np.ndarray:
        U = self.err.nll(self.y, F, self.starts)
        return U if valid is None else np.where(valid, U, np.inf)

    def statistic(self, F) -> float:
        return self.err.statistic(self.y, F)

    def mu(self) -> np.ndarray:
        return self.A_latent @ self.beta  # (m, d)

    def log_prior(self, Phi, mu, omega_inv) -> np.ndarray:
        d = Phi - mu
        return -0.5 * np.einsum("mi,ij,mj->m", d, omega_inv, d)

    # -- E-step: Metropolis-Hastings ----------------------------------------
    def mcmc(self, Phi, F, omega, n_kernel, scales):
        """Update Phi and the flat predictions F in place."""
        m, r = self.M, self.d
        mu = self.mu()[self.sub]
        omega_inv = np.linalg.inv(omega)
        chol = np.linalg.cholesky(omega)
        U = self.nll(F)
        lp = self.log_prior(Phi, mu, omega_inv)

        def propose(prop, with_prior):
            F_new, valid = self.predict(prop)
            U_new = self.nll(F_new, valid)
            lp_new = self.log_prior(prop, mu, omega_inv)
            log_ratio = U - U_new + ((lp_new - lp) if with_prior else 0.0)
            ok = np.log(self.rng.random(m)) < log_ratio
            Phi[ok], U[ok], lp[ok] = prop[ok], U_new[ok], lp_new[ok]
            obs = ok[self.row_of_obs]
            F[obs] = F_new[obs]
            return ok

        # Kernel 1: independent proposals from the prior.
        for _ in range(n_kernel[0]):
            propose(mu + self.rng.standard_normal((m, r)) @ chol.T, with_prior=False)

        # Kernel 2: single-coordinate random walks.
        for _ in range(n_kernel[1]):
            for j in range(r):
                prop = Phi.copy()
                prop[:, j] += scales[0][j] * self.rng.standard_normal(m)
                ok = propose(prop, with_prior=True)
                scales[0][j] *= 1 + 0.4 * (ok.mean() - TARGET_ACCEPT)

        # Kernel 3: full-vector random walks.
        for _ in range(n_kernel[2]):
            ok = propose(Phi + scales[1] * self.rng.standard_normal((m, r)), with_prior=True)
            scales[1] *= 1 + 0.4 * (ok.mean() - TARGET_ACCEPT)

    # -- M-step ---------------------------------------------------------------
    def update_beta_R(self, s1, omega):
        if not self.cols_R.size:
            return
        omega_inv = np.linalg.inv(omega)
        X = self.A_R[:, :, self.cols_R]  # (m, r, fR)
        lhs = np.einsum("mif,ij,mjg->fg", X, omega_inv, X)
        rhs = np.einsum("mif,ij,mj->f", X, omega_inv, s1)
        self.beta[self.cols_R] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def update_omega(self, s1, S2):
        mu = self.A_R @ self.beta
        omega = (S2 - mu.T @ s1 - s1.T @ mu + mu.T @ mu) / self.m
        omega = 0.5 * (omega + omega.T)
        if not self.full_cov:
            omega = np.diag(np.diag(omega))
        # Keep it positive definite. Relative floor, plus an absolute one so
        # that an all-zero Omega (e.g. a single group) stays invertible.
        w, V = np.linalg.eigh(omega)
        floor = max(1e-10 * np.max(np.abs(w)), 1e-12)
        return (V * np.maximum(w, floor)) @ V.T

    def update_beta_F(self, Phi, gamma, n_steps: int = 1):
        """Fixed effects of parameters without random effects.

        Approximately minimise the error-model negative log-likelihood over
        these effects given the current samples (Gauss-Newton on the weighted
        residuals, with step halving; one step per iteration by default, as
        MATLAB limits its fminsearch to 10 iterations), then move towards the
        result with the stochastic-approximation step size.
        """
        cols = self.cols_F
        start = self.beta[cols].copy()

        def objective(values):
            self.beta[cols] = values
            F, valid = self.predict(Phi)
            return float(self.nll(F, valid).sum())

        target = start.copy()
        current = objective(target)
        for _ in range(n_steps):
            self.beta[cols] = target
            # Directional derivatives along the design columns of these
            # effects only, all rows and directions in one batch.
            etas = self.etas(Phi)
            h = 1e-6 * np.maximum(np.max(np.abs(etas), axis=1), 1.0)  # (M,)
            batch = [etas]
            for k in cols:
                step = h[:, None] * self.A_rows[:, :, k]
                batch += [etas + step, etas - step]
            values, _ = self.model.predict_flat(np.tile(self.sub, 1 + 2 * len(cols)), np.vstack(batch))
            n_obs = len(self.y)
            f = values[:n_obs]
            hh = h[self.row_of_obs]
            G = np.column_stack([
                (values[(1 + 2 * k) * n_obs:(2 + 2 * k) * n_obs] - values[(2 + 2 * k) * n_obs:(3 + 2 * k) * n_obs]) / (2 * hh)
                for k in range(len(cols))
            ])
            w = 1.0 / self.err.sd(f)
            ok = np.isfinite(f) & np.all(np.isfinite(G), axis=1)
            step = np.linalg.lstsq(w[ok, None] * G[ok], (w * (self.y - f))[ok], rcond=None)[0]
            for _ in range(30):
                candidate = objective(target + step)
                if candidate <= current:
                    break
                step = step / 2
            else:
                break
            target, current = target + step, candidate
            if np.all(np.abs(step) <= 1e-6 * (np.abs(target) + 1e-6)):
                break
        self.beta[cols] = start + gamma * (target - start)


def fit_saem(
    model: GroupedModel,
    beta0,
    full_cov: bool = False,
    n_iterations=(150, 150, 100),
    n_mcmc=(2, 2, 2),
    n_burn_in: int = 5,
    random_state=None,
    loglik_method: str = "none",
    compute_std_errors: bool = False,
    n_importance: int = 5000,
    n_chains: int = 1,
    error_model: str = "constant",
    error_params=None,
    verbose: int = 0,
) -> SAEMResult:
    rng = np.random.default_rng(random_state)
    params = np.ones(2) if error_params is None else np.asarray(error_params, dtype=float).ravel()
    if error_model == "proportional" and params.size == 1:
        params = np.array([0.0, params[0]])
    elif params.size == 1:
        params = np.array([params[0], 0.0])
    err = ErrorModel(error_model, params)
    s = _SAEM(model, beta0, full_cov, rng, n_chains, err)
    m, r, C = s.m, s.r, n_chains
    K1, K2, K3 = (int(k) for k in n_iterations)
    alpha0 = 10 ** (-3 / max(K1, 1))  # pseudo-variance decay, as MATLAB/saemix
    s.set_extended(K1 > 0)
    Phi = s.mu()[s.sub]
    F, _ = s.predict(Phi)
    if error_model in ("constant", "exponential", "proportional"):
        # Start the error parameter at the residual scale, not at 1.
        value = np.sqrt(s.statistic(F) / (C * s.n))
        err.a, err.b = (0.0, value) if error_model == "proportional" else (value, 0.0)
    omega = np.eye(r)  # saemix's default starting covariance
    omega_F = np.ones(len(s.Fx))  # pseudo-variances of the fixed-only parameters
    d = s.d
    scales = [0.5 * np.ones(d), 0.5 * np.ones(d)]

    def latent_omega():
        if not s.extended:
            return omega
        out = np.zeros((d, d))
        out[:r, :r] = omega
        out[r:, r:] = np.diag(omega_F)
        return out

    for _ in range(n_burn_in):
        s.mcmc(Phi, F, latent_omega(), n_mcmc, scales)

    def chain_stats():
        P = Phi.reshape(C, m, d)
        rese = s.statistic(F) / C
        return P.mean(axis=0), np.einsum("cmi,cmj->ij", P, P) / C, rese, P

    s1, S2, S3, _ = chain_stats()
    cm1 = np.zeros((m, r))
    cm2 = np.zeros((m, r, r))
    total = K1 + K2 + K3
    for k in range(1, total + 1):
        if s.extended and k == K1 + 1:
            # End of annealing: fix the parameters without random effects at
            # their current means and drop them from the sampler.
            s.set_extended(False)
            d = r
            Phi = Phi[:, :r].copy()
            F, _ = s.predict(Phi)
            s1, S2 = s1[:, :r].copy(), S2[:r, :r].copy()
            scales = [scales[0][:r], scales[1][:r]]
        s.mcmc(Phi, F, latent_omega(), n_mcmc, scales)

        gamma = 1.0 if k <= K1 + K2 else 1.0 / (k - K1 - K2)
        t1, t2, t3, P = chain_stats()
        s1 += gamma * (t1 - s1)
        S2 += gamma * (t2 - S2)
        S3 += gamma * (t3 - S3)
        if k > K1 + K2:  # running conditional moments in the last phase
            cm1 += gamma * (t1 - cm1)
            cm2 += gamma * (np.einsum("cmi,cmj->mij", P, P) / C - cm2)

        if s.extended:
            # Means of the sampled fixed-only parameters (GLS with their
            # diagonal pseudo-covariance), then shrink the pseudo-variances.
            X = s.A_latent[:, r:, :][:, :, s.cols_F]
            w = 1.0 / omega_F
            lhs = np.einsum("mif,i,mig->fg", X, w, X)
            rhs = np.einsum("mif,i,mi->f", X, w, s1[:, r:])
            s.beta[s.cols_F] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            omega_F = omega_F * alpha0
        elif s.cols_F.size:
            s.update_beta_F(Phi, gamma)
            F, _ = s.predict(Phi)
        s.update_beta_R(s1[:, :r], omega)
        omega_new = s.update_omega(s1[:, :r], S2[:r, :r])
        if k <= K1:
            # Simulated annealing: variances may shrink by at most TAU per
            # iteration. Only the diagonal is raised (as saemix does), which
            # also pulls correlations towards zero while exploring.
            dg = np.maximum(np.diag(omega_new), TAU * np.diag(omega))
            omega_new = omega_new - np.diag(np.diag(omega_new)) + np.diag(dg)
        elif k <= K1 + K2:
            omega_new = _limit_eigenvalue_decrease(omega_new, omega)
        omega = omega_new
        _update_error(err, S3 / s.n, s.y, F, k <= K1, gamma)
        if verbose > 1:
            print(f"SAEM iteration {k}: beta={s.beta} diag(psi)={np.diag(omega)} error={err.errorparam}")

    s.set_extended(False)
    mu = s.mu()
    b = (cm1 - mu).T
    cond_cov = cm2 - np.einsum("mi,mj->mij", cm1, cm1)

    logl = None
    if loglik_method == "is":
        logl = _loglik_importance(s, cm1, cond_cov, omega, n_importance, rng)
    elif loglik_method == "lin":
        logl = _loglik_linearized(s, cm1, omega)
    elif loglik_method != "none":
        raise NotImplementedError(f"LogLikMethod '{loglik_method}' is not implemented yet")

    covb = _covb_linearized(s, cm1, omega) if compute_std_errors else None
    return SAEMResult(
        beta=s.beta.copy(), psi=omega, errorparam=err.errorparam, b=b, logl=logl, covb=covb, iterations=total
    )


def _update_error(err: ErrorModel, sig2, y, f, annealing: bool, gamma):
    """M-step for the error parameters.

    constant/exponential: a^2 = mean squared residual; proportional:
    b^2 = mean squared relative residual (both annealed in the first phase);
    combined: (a, b) refitted to the current samples by Nelder-Mead and then
    annealed (first phase) or moved with the step size, as MATLAB does.
    """
    if err.kind == "combined":
        ab = np.abs(optimize.fmin(_error_ab, [err.a, err.b], args=(y, f), disp=False))
        if annealing:
            err.a, err.b = max(TAU * err.a, ab[0]), max(TAU * err.b, ab[1])
        else:
            err.a += gamma * (ab[0] - err.a)
            err.b += gamma * (ab[1] - err.b)
        return
    current = err.b if err.kind == "proportional" else err.a
    value = np.sqrt(max(sig2, 1e-300))
    if annealing:
        value = max(value, np.sqrt(TAU) * current)
    if err.kind == "proportional":
        err.b = value
    else:
        err.a = value


def _limit_eigenvalue_decrease(omega_new, omega_old):
    """Let each eigenvalue (matched by rank) shrink by at most TAU per iteration.

    With step size 1, each covariance update rests on a single draw per
    subject. That noise can push Omega towards singularity, and a singular
    Omega is absorbing: the sampler then only proposes values on the
    degenerate subspace. With 20 subjects this happened in most runs
    (correlation -> 1). Limiting the rate keeps the exploration phase away
    from the boundary; genuinely small variances still shrink by TAU^K2 and
    then freely in the averaging phase.
    """
    w, V = np.linalg.eigh(omega_new)
    w_old = np.linalg.eigvalsh(omega_old)
    w = np.maximum(w, TAU * w_old)
    return (V * w) @ V.T


def _linearized(s: _SAEM, Phi, omega):
    """Per-group design X_i, residual r_i and marginal covariance V_i, linearised at Phi."""
    mu = s.mu()
    for i in range(s.m):
        eta = s.eta(i, Phi[i])
        J = s.model.jacobian(i, eta)
        Z = J[:, s.R]
        f = s.model.predict(i, eta)
        res = s.model.y[i] - f + Z @ (Phi[i] - mu[i])
        V = Z @ omega @ Z.T + np.diag(s.err.sd(f) ** 2)
        yield J @ s.model.A[i], res, V


def _loglik_linearized(s, Phi, omega) -> float:
    total = 0.0
    for _, res, V in _linearized(s, Phi, omega):
        total += sps.multivariate_normal(np.zeros(len(res)), V).logpdf(res)
    return float(total)


def _covb_linearized(s, Phi, omega) -> np.ndarray:
    fim = sum(X.T @ np.linalg.solve(V, X) for X, _, V in _linearized(s, Phi, omega))
    return np.linalg.inv(fim)


def _loglik_importance(s, cm1, cond_cov, omega, n_samples, rng, df=10) -> float:
    """Importance sampling with Student-t proposals at the conditional moments.

    10 degrees of freedom: heavier tails than the posterior for robustness,
    but not so heavy that the estimate gets noisy (with 5 df the run-to-run
    SD on a linear model was ~3x larger). With 5000 draws the SD is ~0.01.
    """
    mu = s.mu()
    omega_inv = np.linalg.inv(omega)
    _, logdet_omega = np.linalg.slogdet(omega)
    total = 0.0
    for i in range(s.m):
        n_i = len(s.model.y[i])
        cov = cond_cov[i] + 1e-10 * np.eye(s.r) * max(np.max(np.diag(cond_cov[i])), 1e-12)
        proposal = sps.multivariate_t(loc=cm1[i], shape=cov, df=df, seed=rng)
        draws = np.atleast_2d(proposal.rvs(size=n_samples)).reshape(n_samples, s.r)
        etas = np.tile(s.model.A[i] @ s.beta, (n_samples, 1))
        etas[:, s.R] = draws
        values, valid = s.model.predict_flat(np.full(n_samples, i), etas)
        starts = np.arange(n_samples) * n_i
        nll = np.where(valid, s.err.nll(np.tile(s.model.y[i], n_samples), values, starts), np.inf)
        d = draws - mu[i]
        log_w = (
            -nll - 0.5 * n_i * np.log(2 * np.pi)
            - 0.5 * (s.r * np.log(2 * np.pi) + logdet_omega + np.einsum("ki,ij,kj->k", d, omega_inv, d))
        )
        log_w -= proposal.logpdf(draws)
        top = np.max(log_w)
        total += top + np.log(np.mean(np.exp(log_w - top)))
    return float(total)
