"""
Main API functions for PyNLME - similar to MATLAB's nlmefit and nlmefitsa.
"""

import warnings

import numpy as np

from .data_types import ModelFunction, NLMEOptions, NLMEStats, SAEMOptions
from .utils import validate_inputs

# Try to import Rust backend, fall back to Python implementation
try:
    from . import _core as rust_backend  # Rust backend via maturin

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    rust_backend = None
    warnings.warn("Rust backend not available, using Python implementation")

# Always import Python implementations as fallback
from .algorithms import MLEFitter, SAEMFitter


def nlmefit(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
    """
    Fit nonlinear mixed-effects model using maximum likelihood estimation.

    This function provides an interface similar to MATLAB's nlmefit function.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Predictor variables matrix where n is the number of observations
        and p is the number of predictors.
    y : array_like, shape (n,)
        Response variable vector.
    group : array_like, shape (n,)
        Grouping variable indicating which group each observation belongs to.
    V : array_like, shape (m, q) or None
        Group-level predictor variables where m is the number of groups
        and q is the number of group predictors. Can be None if no group
        predictors are used.
    modelfun : callable
        Model function with signature f(phi, x, v) -> y_pred where:
        - phi: parameter vector
        - x: predictor data
        - v: group predictor data (can be None)
    beta0 : array_like, shape (k,)
        Initial values for fixed-effects parameters.
    **kwargs : keyword arguments
        Additional options passed to NLMEOptions. Common options include:
        - approximation_type : str, default "LME"
        - error_model : str, default "constant"
        - param_transform : array_like, optional
        - max_iter : int, default 200
        - verbose : int, default 0

    Returns
    -------
    beta : ndarray, shape (k,)
        Fixed-effects parameter estimates.
    psi : ndarray, shape (r, r)
        Random-effects covariance matrix where r is the number of random effects.
    stats : NLMEStats
        Statistics and diagnostics from the fit including log-likelihood, AIC, BIC,
        standard errors, and residuals.
    b : ndarray, shape (r, m) or None
        Random-effects estimates for each group. None if not computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pynlme import nlmefit
    >>>
    >>> # Define exponential decay model
    >>> def model(phi, x, v=None):
    ...     return phi[0] * np.exp(-phi[1] * x)
    >>>
    >>> # Sample data
    >>> X = np.array([[0, 1, 2, 3, 4]]).T
    >>> y = np.array([10, 7.3, 5.4, 4.0, 2.9])
    >>> group = np.array([1, 1, 1, 1, 1])
    >>> beta0 = np.array([10.0, 0.5])
    >>>
    >>> # Fit model
    >>> beta, psi, stats, b = nlmefit(X, y, group, None, model, beta0)
    >>> print(f"Fixed effects: {beta}")

    Notes
    -----
    This implementation uses the same algorithmic approach as MATLAB's nlmefit,
    with the core computations implemented in Rust for performance.
    """
    # Create options object from kwargs
    options = NLMEOptions(**kwargs)

    # Validate and prepare inputs
    X, y, group, V, beta0 = validate_inputs(X, y, group, V, beta0)

    # Create fitter and run optimization
    if RUST_AVAILABLE:
        # Use Rust backend
        try:
            # Convert options to Rust-compatible format
            rust_options = rust_backend.MLEOptions()
            rust_options.max_iter = options.max_iter
            rust_options.tol_fun = options.tol_fun
            rust_options.tol_x = options.tol_x
            rust_options.approximation_type = options.approximation_type
            rust_options.compute_std_errors = options.compute_std_errors
            rust_options.verbose = options.verbose

            # Call Rust implementation
            # Convert group array to uint64 for Rust compatibility
            group_uint = group.astype(np.uint64)
            result = rust_backend.fit_nlme_mle(X, y, group_uint, beta0, rust_options, V)

            # Convert result to expected format
            # Create stats object from Rust result
            stats = NLMEStats()
            stats.logl = result.logl
            stats.aic = result.aic
            stats.bic = result.bic
            stats.rmse = result.rmse

            return result.beta, result.psi, stats, result.random_effects

        except Exception as e:
            if options.verbose > 0:
                print(
                    f"Rust backend failed: {e}, falling back to Python implementation"
                )
            # Fall back to Python implementation
            fitter = MLEFitter(options)
    else:
        # Use Python implementation
        fitter = MLEFitter(options)

    try:
        beta, psi, stats, b = fitter.fit(X, y, group, V, modelfun, beta0)
        return beta, psi, stats, b
    except Exception as e:
        warnings.warn(f"Fitting failed: {str(e)}")
        # Return reasonable defaults
        k = len(beta0)
        return (beta0, np.eye(k) * 0.1, NLMEStats(dfe=len(y) - k), None)


def nlmefitsa(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
    """
    Fit nonlinear mixed-effects model using Stochastic Approximation EM (SAEM).

    This function provides an interface similar to MATLAB's nlmefitsa function,
    which uses the SAEM algorithm for fitting nonlinear mixed-effects models.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Predictor variables matrix.
    y : array_like, shape (n,)
        Response variable vector.
    group : array_like, shape (n,)
        Grouping variable.
    V : array_like, shape (m, q) or None
        Group-level predictor variables.
    modelfun : callable
        Model function with signature f(phi, x, v) -> y_pred.
    beta0 : array_like, shape (k,)
        Initial values for fixed-effects parameters.
    **kwargs : keyword arguments
        Additional SAEM-specific options:
        - n_iterations : tuple, default (150, 150, 100)
        - n_mcmc_iterations : tuple, default (2, 2, 2)
        - n_burn_in : int, default 5
        - max_iter : int, default 200
        - verbose : int, default 0

    Returns
    -------
    beta : ndarray, shape (k,)
        Fixed-effects parameter estimates.
    psi : ndarray, shape (r, r)
        Random-effects covariance matrix.
    stats : NLMEStats
        Statistics from the fit.
    b : ndarray, shape (r, m) or None
        Random-effects estimates.

    Examples
    --------
    >>> # Using SAEM algorithm instead of MLE
    >>> beta, psi, stats, b = nlmefitsa(X, y, group, None, model, beta0,
    ...                                 n_iterations=(50, 50, 25))

    Notes
    -----
    SAEM is particularly useful for complex models where traditional MLE
    approaches may have convergence difficulties. The algorithm uses
    Monte Carlo methods in the E-step and stochastic approximation
    in the M-step.
    """
    # Set SAEM-specific defaults
    saem_defaults = {
        "n_iterations": (150, 150, 100),
        "n_mcmc_iterations": (2, 2, 2),
        "n_burn_in": 5,
        "tol_sa": 1e-4,
    }

    # Separate SAEM-specific options from general NLME options
    saem_kwargs = {}
    nlme_kwargs = {}

    for key, value in kwargs.items():
        if key in [
            "n_iterations",
            "n_mcmc_iterations",
            "n_burn_in",
            "step_size_sequence",
            "tol_ll",
            "tol_sa",
        ]:
            saem_kwargs[key] = value
        else:
            nlme_kwargs[key] = value

    # Apply SAEM defaults for missing parameters
    for key, default_val in saem_defaults.items():
        if key not in saem_kwargs:
            saem_kwargs[key] = default_val

    # Create options objects
    options = NLMEOptions(**nlme_kwargs)
    saem_options = SAEMOptions(**saem_kwargs)

    # Validate inputs
    X, y, group, V, beta0 = validate_inputs(X, y, group, V, beta0)

    # Create SAEM fitter
    if RUST_AVAILABLE:
        # Use Rust backend
        try:
            # Convert options to Rust-compatible format
            rust_saem_options = rust_backend.SAEMOptions()
            # Set SAEM-specific options (adjust based on actual Rust struct fields)

            # Call Rust implementation
            # Convert group array to uint64 for Rust compatibility
            group_uint = group.astype(np.uint64)
            result = rust_backend.fit_nlme_saem(
                X, y, group_uint, beta0, rust_saem_options, V
            )

            # Convert result to expected format
            # Create stats object from Rust result
            stats = NLMEStats()
            stats.logl = result.logl
            stats.aic = result.aic
            stats.bic = result.bic
            stats.rmse = result.rmse

            return result.beta, result.psi, stats, result.random_effects

        except Exception as e:
            if options.verbose > 0:
                print(
                    f"Rust SAEM backend failed: {e}, falling back to Python implementation"
                )
            # Fall back to Python implementation
            fitter = SAEMFitter(options, saem_options)
    else:
        # Use Python implementation
        fitter = SAEMFitter(options, saem_options)

    try:
        beta, psi, stats, b = fitter.fit(X, y, group, V, modelfun, beta0)
        return beta, psi, stats, b
    except Exception as e:
        warnings.warn(f"SAEM fitting failed: {str(e)}")
        # Return reasonable defaults
        k = len(beta0)
        return (beta0, np.eye(k) * 0.1, NLMEStats(dfe=len(y) - k), None)


# ========================================================================================
# Pythonic interface
# ========================================================================================


def fit_nlme(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    method: str = "ML",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
    """
    Fit nonlinear mixed-effects model with unified Pythonic interface.

    This is the main Python-style function that provides a unified interface
    to both MLE and SAEM algorithms, similar to scikit-learn conventions.
    Use this function if you prefer Python-style APIs over MATLAB-style.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Predictor variables matrix.
    y : array_like, shape (n,)
        Response variable vector.
    group : array_like, shape (n,)
        Grouping variable.
    V : array_like, shape (m, q) or None
        Group-level predictor variables.
    modelfun : callable
        Model function with signature f(phi, x, v) -> y_pred.
    beta0 : array_like, shape (k,)
        Initial parameter estimates.
    method : str, default "ML"
        Fitting method: "ML" for Maximum Likelihood or "SAEM" for
        Stochastic Approximation EM.
    **kwargs : keyword arguments
        Additional fitting options.

    Returns
    -------
    beta : ndarray
        Fixed-effects parameter estimates.
    psi : ndarray
        Random-effects covariance matrix.
    stats : NLMEStats
        Model statistics and diagnostics.
    b : ndarray or None
        Random-effects estimates for each group.

    Examples
    --------
    >>> # Using ML estimation (default)
    >>> beta, psi, stats, b = fit_nlme(X, y, group, None, model, beta0)
    >>>
    >>> # Using SAEM estimation
    >>> beta, psi, stats, b = fit_nlme(X, y, group, None, model, beta0, method="SAEM")

    Notes
    -----
    This is the recommended function for Python users as it provides a unified
    interface to both algorithms. MATLAB users can use nlmefit()/nlmefitsa() directly.
    """
    if method.upper() == "ML":
        return nlmefit(X, y, group, V, modelfun, beta0, **kwargs)
    elif method.upper() == "SAEM":
        return nlmefitsa(X, y, group, V, modelfun, beta0, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'ML' or 'SAEM'.")
