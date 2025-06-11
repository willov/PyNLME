"""
Type definitions for PyNLME package.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class NLMEStats:
    """Statistics and diagnostics from NLME model fitting.

    This class contains the same fields as MATLAB's nlmefit stats output.
    """

    dfe: int | None = None  # Error degrees of freedom
    logl: float | None = None  # Maximized log-likelihood
    aic: float | None = None  # Akaike Information Criterion
    bic: float | None = None  # Bayesian Information Criterion
    rmse: float | None = None  # Root mean squared error
    mse: float | None = None  # Mean squared error
    sse: float | None = None  # Sum of squared errors
    errorparam: np.ndarray | None = None  # Error model parameters
    sebeta: np.ndarray | None = None  # Standard errors of fixed effects
    covb: np.ndarray | None = None  # Covariance matrix of fixed effects

    # Residuals
    ires: np.ndarray | None = None  # Individual residuals
    pres: np.ndarray | None = None  # Population residuals
    iwres: np.ndarray | None = None  # Individual weighted residuals
    pwres: np.ndarray | None = None  # Population weighted residuals
    cwres: np.ndarray | None = None  # Conditional weighted residuals
    residuals: dict[str, np.ndarray] | None = None  # All residuals in a dict


@dataclass
class NLMEOptions:
    """Options for NLME fitting algorithms.

    This class provides options similar to MATLAB's nlmefit name-value arguments.
    """

    # Algorithm selection
    approximation_type: str = "LME"  # "LME", "RELME", "FO", "FOCE"
    optim_fun: str = "lbfgs"  # Optimization algorithm

    # Parameter transformations
    param_transform: np.ndarray | None = None  # 0=identity, 1=log, 2=probit, 3=logit

    # Error model
    error_model: str = (
        "constant"  # "constant", "proportional", "combined", "exponential"
    )

    # Covariance structure
    cov_pattern: np.ndarray | None = None  # Pattern for random effects covariance
    cov_parametrization: str = "logm"  # "logm" or "chol"

    # Design matrices
    fe_params_select: np.ndarray | None = None  # Fixed effects selection
    re_params_select: np.ndarray | None = None  # Random effects selection
    fe_const_design: np.ndarray | None = None  # Fixed effects design matrix
    re_const_design: np.ndarray | None = None  # Random effects design matrix

    # Algorithmic options
    vectorization: str = "SinglePhi"  # "SinglePhi", "SingleGroup", "Full"
    compute_std_errors: bool = True
    refine_beta0: bool = True

    # Convergence
    max_iter: int = 200
    tol_fun: float = 1e-6
    tol_x: float = 1e-6

    # Output control
    verbose: int = 0
    random_state: int | None = None


@dataclass
class SAEMOptions:
    """Options specific to SAEM algorithm.

    This class contains SAEM-specific options that extend the base NLMEOptions.
    """

    # SAEM-specific algorithm parameters
    n_iterations: tuple[int, int, int] = (
        150,
        150,
        100,
    )  # (burn-in, stochastic, smooth)
    n_mcmc_iterations: tuple[int, int, int] = (2, 2, 2)  # MCMC iterations per phase
    n_burn_in: int = 5  # Additional burn-in
    step_size_sequence: str = "auto"  # "auto" or custom sequence

    # SAEM convergence criteria
    tol_ll: float = 1e-6  # Log-likelihood tolerance
    tol_sa: float = 1e-4  # Stochastic approximation tolerance

    # Additional options
    verbose: int = 0  # Verbosity level
    random_state: int | None = None  # Random seed
    algorithm: str = "SAEM"  # Algorithm name


@dataclass
class ErrorModel:
    """Error model specification."""

    model_type: str  # "constant", "proportional", "combined", "exponential"
    parameters: np.ndarray

    def evaluate(self, y_pred: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Evaluate error model variance."""
        if self.model_type == "constant":
            return np.full_like(y_pred, theta[0] ** 2)
        elif self.model_type == "proportional":
            return (theta[0] * np.abs(y_pred)) ** 2
        elif self.model_type == "combined":
            return (theta[0] + theta[1] * np.abs(y_pred)) ** 2
        elif self.model_type == "exponential":
            return np.exp(2 * theta[0] * np.log(np.abs(y_pred)))
        else:
            raise ValueError(f"Unknown error model: {self.model_type}")


# Type aliases for better readability
ModelFunction = Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]
DesignMatrix = np.ndarray | None
GroupVariable = np.ndarray | list
