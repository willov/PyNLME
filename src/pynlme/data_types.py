"""
Type definitions for pynlme package.
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
    mse: float | None = (
        None  # Estimated error variance (MATLAB's undocumented stats.mse)
    )
    errorparam: np.ndarray | None = None  # Error model parameters
    sebeta: np.ndarray | None = None  # Standard errors of fixed effects
    covb: np.ndarray | None = None  # Covariance matrix of fixed effects

    # Residuals
    ires: np.ndarray | None = None  # Individual residuals
    pres: np.ndarray | None = None  # Population residuals
    iwres: np.ndarray | None = None  # Individual weighted residuals
    pwres: np.ndarray | None = None  # Population weighted residuals
    cwres: np.ndarray | None = None  # Conditional weighted residuals


@dataclass
class NLMEOptions:
    """Options for NLME fitting algorithms.

    Each field corresponds to a MATLAB nlmefit/nlmefitsa name-value argument;
    the fitting functions also accept the MATLAB names (e.g. ``ParamTransform``).
    """

    # Algorithm selection
    approximation_type: str = "LME"  # "LME", "RELME" ("FO", "FOCE" not yet)
    optim_fun: str = (
        "fminsearch"  # LME step optimizer (compat="matlab"); only "fminsearch"
    )

    # Parameter transformations
    param_transform: np.ndarray | None = None  # 0=identity, 1=log, 2=probit, 3=logit

    # Error model
    error_model: str = (
        "constant"  # "constant", "proportional", "combined", "exponential"
    )
    error_parameters: np.ndarray | None = None  # Starting (a,) or (a, b) for "combined"

    # Covariance structure
    cov_pattern: np.ndarray | None = None  # None/identity = diagonal, ones = full
    cov_parametrization: str = "logm"  # "logm" or "chol" (compat="matlab" LME/RELME)

    # Design matrices
    fe_params_select: np.ndarray | None = None  # Fixed effects selection
    re_params_select: np.ndarray | None = None  # Random effects selection
    fe_const_design: np.ndarray | None = None  # p-by-f fixed effects design
    re_const_design: np.ndarray | None = None  # p-by-r random effects design
    fe_group_design: np.ndarray | None = None  # p-by-f-by-m, one per group
    re_group_design: np.ndarray | None = None  # p-by-r-by-m, one per group
    fe_obs_design: np.ndarray | None = None  # Not yet supported
    re_obs_design: np.ndarray | None = None  # Not yet supported

    # Algorithmic options
    vectorization: str = "SinglePhi"  # Only "SinglePhi" is implemented so far
    compute_std_errors: bool = True
    refine_beta0: bool = True

    # Convergence. LME/RELME use the reference's defaults instead (MATLAB's
    # statset 1e-4 for compat="matlab", nlmeControl's for compat="r").
    max_iter: int = 200
    tol_fun: float = 1e-6
    tol_x: float = 1e-6

    # Output control
    verbose: int = 0
    random_state: int | None = None

    # Input layout: "stacked" (y a vector, group per observation) or
    # "grouped" (y one row per group). None: inferred from the shape of y.
    data_format: str | None = None

    # Conventions and defaults: "matlab" (MATLAB nlmefit/nlmefitsa) or "r"
    # (R nlme for nlmefit, saemix for nlmefitsa). Explicit options override.
    compat: str = "matlab"

    # nlmeControl's PNLS settings (compat="r", LME/RELME): pnlsTol,
    # pnlsMaxIter, minScale. None: nlme's defaults (1e-3, 7, 1e-3).
    pnls_tol: float | None = None
    pnls_max_iter: int | None = None
    min_scale: float | None = None


@dataclass
class SAEMOptions:
    """Options specific to nlmefitsa (MATLAB names in comments)."""

    n_iterations: tuple[int, int, int] = (150, 150, 100)  # NIterations
    n_mcmc_iterations: tuple[int, int, int] = (2, 2, 2)  # NMCMCIterations, per kernel
    n_burn_in: int = 5  # NBurnIn
    loglik_method: str = "none"  # LogLikMethod: "none", "is", "lin"
    n_chains: int | None = None  # NChains; None: max(1, round(100 / m)) as MATLAB


# Type aliases for better readability
ModelFunction = Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]
DesignMatrix = np.ndarray | None
GroupVariable = np.ndarray | list
