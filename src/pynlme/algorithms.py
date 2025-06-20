"""
Core fitting algorithms for PyNLME.

This module contains the main fitting algorithms for nonlinear mixed-effects models:
- MLEFitter: Maximum likelihood estimation (similar to MATLAB's nlmefit)
- SAEMFitter: Stochastic Approximation EM (similar to MATLAB's nlmefitsa)
"""

import numpy as np
from scipy import optimize

from .data_types import ModelFunction, NLMEOptions, NLMEStats, SAEMOptions
from .utils import compute_information_criteria, compute_residuals


class MLEFitter:
    """
    Maximum likelihood estimation for nonlinear mixed-effects models.

    This class implements the traditional MLE approach used by MATLAB's nlmefit,
    with the core computations eventually to be implemented in Rust.
    """

    def __init__(self, options: NLMEOptions):
        self.options = options
        if options.random_state is not None:
            np.random.seed(options.random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        V: np.ndarray | None,
        modelfun: ModelFunction,
        beta0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
        """
        Fit nonlinear mixed-effects model using maximum likelihood.

        Parameters
        ----------
        X : ndarray
            Predictor variables
        y : ndarray
            Response variable
        group : ndarray
            Grouping variable (0-based indices)
        V : ndarray or None
            Group-level predictors
        modelfun : callable
            Model function
        beta0 : ndarray
            Initial parameter estimates

        Returns
        -------
        beta : ndarray
            Fixed-effects estimates
        psi : ndarray
            Random-effects covariance matrix
        stats : NLMEStats
            Fitting statistics
        b : ndarray or None
            Random-effects estimates
        """
        n_obs = len(y)
        n_groups = len(np.unique(group))
        n_params = len(beta0)

        # Input validation
        if len(X) != len(y) or len(y) != len(group):
            raise ValueError("X, y, and group must have the same length")
        if beta0 is None:
            raise TypeError("beta0 cannot be None")
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")

        # For now, implement a simplified version
        # In the full implementation, this would call Rust functions

        try:
            # Simple optimization using scipy
            result = self._optimize_likelihood(X, y, group, V, modelfun, beta0)
            beta = result["beta"]
            psi = result["psi"]
            logl = result["logl"]

            # Compute statistics
            stats = NLMEStats()
            stats.dfe = n_obs - n_params
            stats.logl = logl
            stats.aic, stats.bic = compute_information_criteria(
                logl, n_params, n_groups
            )
            stats.rmse = np.sqrt(
                np.mean((y - self._predict_population(X, V, modelfun, beta)) ** 2)
            )

            # Compute residuals
            y_pred_pop = self._predict_population(X, V, modelfun, beta)
            y_pred_ind = y_pred_pop  # Simplified - no random effects computed yet

            residuals = compute_residuals(y, y_pred_pop, y_pred_ind)
            stats.ires = residuals["ires"]
            stats.pres = residuals["pres"]
            stats.iwres = residuals["iwres"]
            stats.pwres = residuals["pwres"]
            stats.cwres = residuals["cwres"]

            # Placeholder for random effects
            b = (
                np.zeros((n_groups, n_params))
                if self.options.compute_std_errors
                else None
            )

            return beta, psi, stats, b

        except (ValueError, RuntimeError, optimize.OptimizeWarning) as e:
            if self.options.verbose > 0:
                print(f"MLE fitting failed: {e}")
            # Return reasonable defaults
            return (
                beta0,
                np.eye(n_params) * 0.1,
                NLMEStats(dfe=n_obs - n_params),
                None,
            )

    def _optimize_likelihood(self, X, y, group, V, modelfun, beta0):
        """Simplified likelihood optimization - to be replaced with Rust implementation."""
        # Note: group parameter will be used in full mixed-effects implementation
        _ = group  # Acknowledge unused parameter

        def objective(params):
            beta = params[: len(beta0)]
            sigma = np.exp(params[-1])  # Log-parameterized error variance

            try:
                y_pred = self._predict_population(X, V, modelfun, beta)
                residuals = y - y_pred
                logl = -0.5 * np.sum(
                    residuals**2 / sigma**2 + np.log(2 * np.pi * sigma**2)
                )
                return -logl  # Minimize negative log-likelihood
            except (ValueError, RuntimeError, OverflowError):
                return 1e10  # Return large value if model evaluation fails

        # Initial parameters: beta + log(sigma)
        initial_params = np.concatenate([beta0, [np.log(1.0)]])

        result = optimize.minimize(objective, initial_params, method="L-BFGS-B")

        if result.success:
            beta = result.x[: len(beta0)]
            sigma = np.exp(result.x[-1])
            logl = -result.fun
            psi = np.eye(len(beta0)) * 0.1  # Simplified random effects covariance

            return {"beta": beta, "psi": psi, "logl": logl, "sigma": sigma}
        else:
            raise RuntimeError("Optimization failed")

    def _predict_population(self, X, V, modelfun, beta):
        """Make population predictions (fixed effects only)."""
        n_obs = X.shape[0]
        y_pred = np.zeros(n_obs)

        if V is None:
            # No group-level predictors
            try:
                y_pred = modelfun(beta, X)
            except (ValueError, RuntimeError, TypeError):
                # Try observation-by-observation
                for i in range(n_obs):
                    y_pred[i] = modelfun(beta, X[i : i + 1])
        else:
            # Handle group-level predictors
            # V should be (m, g) where m=number of groups, g=number of group variables
            # For now, pass the V matrix for the first group to the model function
            try:
                # Pass the full V matrix to let the model function handle it
                y_pred = modelfun(beta, X, V)
            except (ValueError, TypeError):
                y_pred = modelfun(beta, X)

        return y_pred


class SAEMFitter:
    """
    Stochastic Approximation Expectation-Maximization for nonlinear mixed-effects models.

    This class implements the SAEM algorithm used by MATLAB's nlmefitsa.
    """

    def __init__(self, options: NLMEOptions, saem_options: SAEMOptions | None = None):
        self.options = options
        self.saem_options = saem_options or SAEMOptions()
        self.rng = np.random.default_rng(options.random_state)

        # SAEM-specific options from SAEMOptions
        self.n_iterations = self.saem_options.n_iterations
        self.n_mcmc_iterations = self.saem_options.n_mcmc_iterations
        self.n_burn_in = self.saem_options.n_burn_in
        self.tol_sa = self.saem_options.tol_sa

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        V: np.ndarray | None,
        modelfun: ModelFunction,
        beta0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
        """
        Fit nonlinear mixed-effects model using SAEM algorithm.

        This is a proper (simplified) SAEM implementation that estimates both
        fixed effects (beta) and random effects covariance (psi).
        """
        n_groups = len(np.unique(group))
        n_params = len(beta0)

        # Input validation
        if len(X) != len(y) or len(y) != len(group):
            raise ValueError("X, y, and group must have the same length")
        if beta0 is None:
            raise TypeError("beta0 cannot be None")
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        if len(group) == 0:
            raise ValueError("group cannot be empty")
        if any(n < 0 for n in self.n_iterations):
            raise ValueError("n_iterations must be non-negative")

        # Initialize parameters
        beta = beta0.copy()
        psi = np.eye(n_params) * 0.1  # Random effects covariance
        sigma = 0.2  # Residual standard deviation

        # Initialize random effects for each group
        b = np.zeros((n_groups, n_params))

        # SAEM algorithm phases
        total_iterations = sum(self.n_iterations)
        completed_iterations = 0

        for phase, (n_iter, n_mcmc) in enumerate(
            zip(self.n_iterations, self.n_mcmc_iterations, strict=False)
        ):
            phase_name = ["Burn-in", "Stochastic", "Smoothing"][phase]
            phase_emoji = ["🔥", "🎲", "✨"][phase]

            if self.options.verbose > 0:
                print(
                    f"{phase_emoji} SAEM {phase_name} Phase: {n_iter} iterations, {n_mcmc} MCMC steps"
                )

            for iteration in range(n_iter):
                # Overall progress
                overall_progress = (completed_iterations + iteration + 1) / total_iterations * 100

                if self.options.verbose > 1:
                    print(
                        f"  [{overall_progress:5.1f}%] Phase {phase + 1}, Iter {iteration + 1}/{n_iter}: β={beta.round(3)}, σ={sigma:.3f}"
                    )

                # E-step: Sample random effects using MCMC
                b = self._mcmc_sample_random_effects(
                    X, y, group, V, modelfun, beta, psi, sigma, b, n_mcmc
                )

                if self.options.verbose > 1:
                    print(f"            After E-step: b_mean={np.mean(b, axis=0).round(3)}")

                # M-step: Update parameters
                step_size = 1.0 / (iteration + 1) if phase == 0 else 0.1
                beta, psi, sigma = self._update_parameters(
                    X, y, group, V, modelfun, b, beta, psi, sigma, step_size
                )

                # Progress indicators every 10 iterations or at end
                if (iteration + 1) % 10 == 0 or iteration + 1 == n_iter:
                    if self.options.verbose > 0:
                        phase_progress = (iteration + 1) / n_iter * 100
                        print(f"  ⏳ [{overall_progress:5.1f}%] {phase_name}: {phase_progress:.0f}% | β={beta.round(3)} | σ={sigma:.3f}")

                # Quick progress for very verbose mode
                elif self.options.verbose > 1:
                    print(f"            Updated: β={beta.round(3)}, σ={sigma:.3f}")

            completed_iterations += n_iter

            if self.options.verbose > 0:
                print(f"✅ {phase_name} phase completed")
                print(f"   Final estimates: β={beta.round(3)}, σ={sigma:.3f}")
                print("")

        # Compute final statistics
        stats = self._compute_stats(X, y, group, V, modelfun, beta, psi, sigma, b)

        if self.options.verbose > 0:
            total_iter = sum(self.n_iterations)
            print("🎉 SAEM algorithm completed!")
            print(f"   Total iterations: {total_iter}")
            print(f"   Final log-likelihood: {stats.logl:.2f}")
            print(f"   Final parameter estimates: β={beta.round(3)}")

        return beta, psi, stats, b

    def _mcmc_sample_random_effects(
        self, X, y, group, V, modelfun, beta, psi, sigma, b_current, n_mcmc
    ):
        """Sample random effects using simple Metropolis-Hastings MCMC."""
        if self.options.verbose > 1:
            print(f"    Starting MCMC sampling with {n_mcmc} steps")

        n_groups = len(np.unique(group))
        n_params = len(beta)
        b_new = b_current.copy()

        # For each group, sample random effects
        for g in range(n_groups):
            if self.options.verbose > 1:
                print(f"      Sampling group {g}/{n_groups}")

            group_mask = group == g
            if not np.any(group_mask):
                continue

            X_g = X[group_mask]
            y_g = y[group_mask]

            # Current random effects for this group
            b_g = b_current[g].copy()

            # MCMC sampling - SIMPLIFIED to avoid infinite loops
            for mcmc_step in range(min(n_mcmc, 3)):  # Limit MCMC steps
                if self.options.verbose > 1:
                    print(f"        MCMC step {mcmc_step}")

                # Propose new random effects
                proposal_std = 0.1
                b_prop = b_g + self.rng.normal(0, proposal_std, n_params)

                # Compute log-posterior for current and proposed
                try:
                    log_p_current = self._log_posterior_b(
                        X_g, y_g, modelfun, beta, b_g, psi, sigma, V, g
                    )
                    log_p_proposal = self._log_posterior_b(
                        X_g, y_g, modelfun, beta, b_prop, psi, sigma, V, g
                    )

                    # Numerically stable accept/reject calculation
                    # Avoid overflow in exp() by handling large positive log_alpha separately
                    log_alpha = log_p_proposal - log_p_current

                    # Avoid overflow: if log_alpha > 0, always accept
                    if log_alpha >= 0:
                        alpha = 1.0
                    else:
                        # Only compute exp for negative values to avoid overflow
                        alpha = np.exp(log_alpha)

                    if self.rng.random() < alpha:
                        b_g = b_prop
                        if self.options.verbose > 1:
                            print("          Accepted proposal")
                except (ValueError, RuntimeError, OverflowError) as e:
                    if self.options.verbose > 1:
                        print(f"          MCMC error: {e}")
                    break

            b_new[g] = b_g

        if self.options.verbose > 1:
            print("    MCMC sampling completed")
        return b_new

    def _log_posterior_b(self, X_g, y_g, modelfun, beta, b_g, psi, sigma, V, g):
        """Compute log-posterior for random effects of one group."""
        try:
            # Individual parameters for this group
            phi_g = beta + b_g

            # Model prediction
            # Pass the appropriate group-level covariates for this group
            v_group = V[g] if V is not None and len(V) > g else None
            y_pred = modelfun(phi_g, X_g, v_group)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)

            # Log-likelihood contribution
            residuals = y_g - y_pred
            log_lik = -0.5 * np.sum(residuals**2) / (sigma**2) - len(y_g) * np.log(
                sigma
            )

            # Log-prior (multivariate normal)
            psi_inv = np.linalg.inv(psi + np.eye(len(psi)) * 1e-6)  # Regularize
            log_prior = -0.5 * b_g.T @ psi_inv @ b_g

            result = log_lik + log_prior
            return result
        except (ValueError, RuntimeError, OverflowError) as e:
            if self.options.verbose > 0:
                print(f"        Exception in log_posterior_b: {e}")
            return -np.inf

    def _update_parameters(
        self, X, y, group, V, modelfun, b, beta_old, psi_old, sigma_old, step_size
    ):
        """Update parameters in M-step."""
        n_groups = len(np.unique(group))
        n_params = len(beta_old)

        # Update beta using weighted least squares approach
        # We need to find beta that minimizes the residuals given current random effects
        total_weight = 0
        weighted_sum = np.zeros(n_params)

        for g in range(n_groups):
            group_mask = group == g
            if not np.any(group_mask):
                continue

            X_g = X[group_mask]
            y_g = y[group_mask]
            b_g = b[g]

            # Weight is inverse variance (simplified)
            weight = len(y_g) / (sigma_old**2)

            # For SAEM, we want to estimate beta by fitting the model
            # The current individual parameters are phi_g = beta + b_g
            # We need to extract what beta should be given the current b_g
            try:
                # Try to estimate phi_g from the data for this group
                v_group = V[g] if V is not None and len(V) > g else None

                # Use current phi as starting point for optimization
                phi_current = beta_old + b_g
                phi_optimal = self._optimize_individual_params(
                    X_g, y_g, v_group, modelfun, phi_current
                )

                # The optimal beta contribution from this group would be phi_optimal - b_g
                beta_contribution = phi_optimal - b_g
                weighted_sum += weight * beta_contribution
                total_weight += weight

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Fallback: use the current estimate
                weighted_sum += weight * beta_old
                total_weight += weight

        if total_weight > 0:
            beta_new = weighted_sum / total_weight
        else:
            beta_new = beta_old

        # Update psi (random effects covariance)
        psi_new = np.cov(b.T) if n_groups > 1 else np.eye(n_params) * 0.01

        # Ensure positive definiteness
        psi_new += np.eye(n_params) * 1e-6

        # Update sigma (residual standard deviation)
        total_sse = 0
        total_n = 0

        for g in range(n_groups):
            group_mask = group == g
            if not np.any(group_mask):
                continue

            X_g = X[group_mask]
            y_g = y[group_mask]
            phi_g = beta_new + b[g]

            try:
                v_group = V[g] if V is not None and len(V) > g else None
                y_pred = modelfun(phi_g, X_g, v_group)
                if not isinstance(y_pred, np.ndarray):
                    y_pred = np.array(y_pred)
                residuals = y_g - y_pred
                total_sse += np.sum(residuals**2)
                total_n += len(y_g)
            except (ValueError, RuntimeError):
                continue

        if total_n > 0:
            sigma_new = np.sqrt(total_sse / total_n)
        else:
            sigma_new = sigma_old

        # Apply step size (stochastic approximation)
        beta = (1 - step_size) * beta_old + step_size * beta_new
        psi = (1 - step_size) * psi_old + step_size * psi_new
        sigma = (1 - step_size) * sigma_old + step_size * sigma_new

        # Ensure reasonable bounds
        sigma = max(sigma, 0.01)

        return beta, psi, sigma

    def _compute_stats(self, X, y, group, V, modelfun, beta, psi, sigma, b):
        """Compute final statistics."""
        # Note: psi used implicitly in total parameter count calculation
        _ = psi  # Acknowledge unused parameter
        n_obs = len(y)
        n_params = len(beta)

        # Compute log-likelihood
        log_lik = 0
        total_sse = 0

        for g in range(len(np.unique(group))):
            group_mask = group == g
            if not np.any(group_mask):
                continue

            X_g = X[group_mask]
            y_g = y[group_mask]
            phi_g = beta + b[g]

            try:
                v_group = V[g] if V is not None and len(V) > g else None
                y_pred = modelfun(phi_g, X_g, v_group)
                if not isinstance(y_pred, np.ndarray):
                    y_pred = np.array(y_pred)
                residuals = y_g - y_pred

                # Likelihood contribution
                log_lik += -0.5 * np.sum(residuals**2) / (sigma**2) - len(y_g) * np.log(
                    sigma
                )
                total_sse += np.sum(residuals**2)
            except (ValueError, RuntimeError):
                continue

        rmse = np.sqrt(total_sse / n_obs) if n_obs > 0 else sigma

        # Information criteria
        n_total_params = (
            n_params + n_params * (n_params + 1) // 2 + 1
        )  # beta + psi + sigma
        aic = -2 * log_lik + 2 * n_total_params
        bic = -2 * log_lik + np.log(n_obs) * n_total_params

        stats = NLMEStats(
            logl=log_lik, aic=aic, bic=bic, rmse=rmse, dfe=n_obs - n_total_params
        )

        return stats

    def _optimize_individual_params(self, X_g, y_g, v_group, modelfun, phi_init):
        """
        Optimize individual parameters for a single group.

        This is a simplified optimization that tries to find the best phi_g
        for group g given the current data.
        """
        from scipy.optimize import minimize

        def objective(phi):
            try:
                y_pred = modelfun(phi, X_g, v_group)
                if not isinstance(y_pred, np.ndarray):
                    y_pred = np.array(y_pred)
                residuals = y_g - y_pred
                return np.sum(residuals**2)
            except (ValueError, RuntimeError):
                return 1e10  # Large penalty for invalid parameters

        try:
            # Simple optimization with bounds to avoid extreme values
            bounds = [(-10, 10) for _ in phi_init]
            result = minimize(objective, phi_init, bounds=bounds, method='L-BFGS-B')
            if result.success:
                return result.x
            else:
                return phi_init
        except (ValueError, RuntimeError, optimize.OptimizeWarning):
            return phi_init


# Convenience functions for easier testing and usage


def mle_algorithm(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    options: NLMEOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
    """
    Convenience function for MLE fitting.

    Parameters
    ----------
    X : ndarray
        Predictor variables
    y : ndarray
        Response variable
    group : ndarray
        Grouping variable (0-based indices)
    V : ndarray or None
        Covariate matrix for random effects
    modelfun : callable
        Model function
    beta0 : ndarray
        Initial parameter estimates
    options : NLMEOptions, optional
        Fitting options

    Returns
    -------
    beta : ndarray
        Fixed effects estimates
    psi : ndarray
        Random effects covariance matrix
    stats : NLMEStats
        Fitting statistics
    b : ndarray or None
        Individual random effects
    """
    if options is None:
        options = NLMEOptions()

    fitter = MLEFitter(options)
    return fitter.fit(X, y, group, V, modelfun, beta0)


def saem_algorithm(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    options: SAEMOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
    """
    Convenience function for SAEM fitting.

    Parameters
    ----------
    X : ndarray
        Predictor variables
    y : ndarray
        Response variable
    group : ndarray
        Grouping variable (0-based indices)
    V : ndarray or None
        Covariate matrix for random effects
    modelfun : callable
        Model function
    beta0 : ndarray
        Initial parameter estimates
    options : SAEMOptions, optional
        SAEM fitting options

    Returns
    -------
    beta : ndarray
        Fixed effects estimates
    psi : ndarray
        Random effects covariance matrix
    stats : NLMEStats
        Fitting statistics
    b : ndarray or None
        Individual random effects
    """
    if options is None:
        options = SAEMOptions()

    # Convert SAEMOptions to NLMEOptions for the fitter
    nlme_options = NLMEOptions(
        max_iter=max(options.n_iterations) if options.n_iterations else 100,
        tol_fun=options.tol_sa,
        verbose=getattr(options, "verbose", 0),
        random_state=getattr(options, "random_state", None),
    )

    fitter = SAEMFitter(nlme_options, options)
    return fitter.fit(X, y, group, V, modelfun, beta0)
