"""
Utility functions for PyNLME package.
"""

import numpy as np


def stack_grouped_data(
    X_grouped: np.ndarray,
    y_grouped: np.ndarray,
    group_ids: np.ndarray | list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert multi-dimensional grouped data to stacked format.

    This function converts data where each row represents a subject/group
    to the stacked format expected by the NLME algorithms.

    Parameters
    ----------
    X_grouped : array_like, shape (n_groups, n_obs_per_group, n_features) or (n_groups, n_obs_per_group)
        Predictor variables in grouped format where:
        - n_groups: number of groups/subjects
        - n_obs_per_group: number of observations per group
        - n_features: number of predictor variables (optional dimension)
    y_grouped : array_like, shape (n_groups, n_obs_per_group)
        Response variables in grouped format
    group_ids : array_like, optional
        Custom group identifiers. If None, uses consecutive integers starting from 0.

    Returns
    -------
    X_stacked : ndarray, shape (n_groups * n_obs_per_group, n_features)
        Stacked predictor variables
    y_stacked : ndarray, shape (n_groups * n_obs_per_group,)
        Stacked response variables
    group_stacked : ndarray, shape (n_groups * n_obs_per_group,)
        Group identifiers for each observation

    Examples
    --------
    >>> # 3 subjects, 4 measurements each, 1 predictor variable
    >>> X_grouped = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    >>> y_grouped = np.array([[10, 7, 5, 3], [12, 8, 6, 4], [11, 8, 6, 3]])
    >>> X_stacked, y_stacked, group_stacked = stack_grouped_data(X_grouped, y_grouped)
    >>> print(X_stacked.shape)  # (12, 1)
    >>> print(y_stacked.shape)  # (12,)
    >>> print(group_stacked.shape)  # (12,)
    """
    X_grouped = np.asarray(X_grouped)
    y_grouped = np.asarray(y_grouped)

    # Ensure both arrays are at least 2D
    if X_grouped.ndim == 1:
        X_grouped = X_grouped.reshape(1, -1)
    if y_grouped.ndim == 1:
        y_grouped = y_grouped.reshape(1, -1)

    # Check dimensions match
    if X_grouped.shape[:2] != y_grouped.shape[:2]:
        raise ValueError(
            f"X_grouped and y_grouped must have same dimensions for groups and observations. "
            f"Got X_grouped: {X_grouped.shape[:2]}, y_grouped: {y_grouped.shape[:2]}"
        )

    n_groups, n_obs_per_group = y_grouped.shape[:2]

    # Handle X_grouped dimensions
    if X_grouped.ndim == 2:
        # Shape: (n_groups, n_obs_per_group) -> add feature dimension
        X_grouped = X_grouped.reshape(n_groups, n_obs_per_group, 1)
    elif X_grouped.ndim == 3:
        # Shape: (n_groups, n_obs_per_group, n_features) - already correct
        pass
    else:
        raise ValueError(f"X_grouped must be 2D or 3D array, got {X_grouped.ndim}D")

    n_features = X_grouped.shape[2]

    # Stack the data
    X_stacked = X_grouped.reshape(-1, n_features)
    y_stacked = y_grouped.reshape(-1)

    # Create group identifiers
    if group_ids is None:
        group_ids = np.arange(n_groups)
    else:
        group_ids = np.asarray(group_ids)
        if len(group_ids) != n_groups:
            raise ValueError(f"group_ids must have length {n_groups}, got {len(group_ids)}")

    # Repeat group IDs for each observation
    group_stacked = np.repeat(group_ids, n_obs_per_group)

    return X_stacked, y_stacked, group_stacked


def detect_data_format(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list | None = None,
) -> str:
    """
    Detect whether data is in stacked or grouped format.

    Parameters
    ----------
    X : array_like
        Predictor variables
    y : array_like
        Response variables
    group : array_like, optional
        Group identifiers (only used for stacked format)

    Returns
    -------
    format : str
        Either 'stacked' or 'grouped'
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # If y is 2D, likely grouped format
    if y.ndim == 2:
        return 'grouped'

    # If X is 3D, likely grouped format
    if X.ndim == 3:
        return 'grouped'

    # If group is None and data could be grouped, check dimensions
    if group is None:
        # If X is 2D and y is 2D, grouped
        if X.ndim == 2 and y.ndim == 2:
            return 'grouped'
        # If X is 2D and y is 1D, but both have small first dimension, might be grouped
        if X.ndim == 2 and y.ndim == 1 and X.shape[0] <= 10 and X.shape[0] == len(y):
            # Could be either, but if no group provided, assume grouped
            return 'grouped'

    # If group is provided and has same length as y, likely stacked format
    if group is not None:
        group = np.asarray(group)
        if len(group) == len(y):
            return 'stacked'

    # If X and y have same first dimension and it's large, likely stacked
    if X.shape[0] == y.shape[0] and X.shape[0] > 10:
        return 'stacked'

    # Default to stacked format
    return 'stacked'


def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list | None,
    V: np.ndarray | None,
    beta0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Validate and standardize inputs for NLME fitting.

    This function now supports both stacked and grouped data formats.
    If grouped format is detected, it automatically converts to stacked format.

    Parameters
    ----------
    X : array_like
        Predictor variables. Can be:
        - Stacked: shape (n_obs, n_features)
        - Grouped: shape (n_groups, n_obs_per_group, n_features) or (n_groups, n_obs_per_group)
    y : array_like
        Response variable. Can be:
        - Stacked: shape (n_obs,)
        - Grouped: shape (n_groups, n_obs_per_group)
    group : array_like or None
        Grouping variable. Required for stacked format, ignored for grouped format.
    V : array_like or None
        Group-level predictors
    beta0 : array_like
        Initial parameter values

    Returns
    -------
    X, y, group, V, beta0 : validated and converted arrays
        All arrays are returned in stacked format regardless of input format.
    """
    # Convert to numpy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta0 = np.asarray(beta0, dtype=float)

    if V is not None:
        V = np.asarray(V, dtype=float)

    # Detect data format and convert if necessary
    data_format = detect_data_format(X, y, group)

    if data_format == 'grouped':
        # Convert grouped data to stacked format
        if group is not None:
            # Use provided group IDs
            group_ids = np.asarray(group)
        else:
            # Generate sequential group IDs
            group_ids = None

        X, y, group = stack_grouped_data(X, y, group_ids)
    else:
        # Already in stacked format
        if group is None:
            raise ValueError("group parameter is required for stacked data format")
        group = np.asarray(group)
        y = y.ravel()

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Check dimensions
    n_obs = len(y)
    if X.shape[0] != n_obs:
        raise ValueError(
            f"X and y must have same number of observations. Got {X.shape[0]} and {n_obs}"
        )

    if len(group) != n_obs:
        raise ValueError(
            f"group must have same length as y. Got {len(group)} and {n_obs}"
        )

    # Check for missing values
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y cannot contain NaN values")

    # Convert group to integer indices
    group_indices, unique_groups = grp2idx(group)
    n_groups = len(unique_groups)

    # Validate V if provided
    if V is not None:
        if V.ndim == 1:
            V = V.reshape(1, -1)
        if V.shape[0] != n_groups:
            raise ValueError(f"V must have {n_groups} rows (one per group)")

    return X, y, group_indices, V, beta0


def grp2idx(group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert group labels to consecutive integer indices starting from 0.

    Parameters
    ----------
    group : array_like
        Group labels

    Returns
    -------
    indices : ndarray
        Integer indices (0-based)
    unique_groups : ndarray
        Unique group labels in order
    """
    unique_groups = np.unique(group)
    group_map = {g: i for i, g in enumerate(unique_groups)}
    indices = np.array([group_map[g] for g in group])
    return indices, unique_groups


def transform_parameters(
    phi: np.ndarray, transform_codes: np.ndarray | None
) -> np.ndarray:
    """
    Apply parameter transformations similar to MATLAB's nlmefit.

    This function provides parameter transformations commonly used in
    nonlinear mixed-effects modeling to ensure parameter constraints
    (e.g., positive parameters, bounded parameters).

    Parameters
    ----------
    phi : ndarray
        Parameter values to transform
    transform_codes : ndarray or None
        Transformation codes:
        - 0: identity (no transformation)
        - 1: exponential (log-normal) - for positive parameters
        - 2: probit (normal CDF) - for parameters in (0,1)
        - 3: logit (inverse logit) - for parameters in (0,1)

    Returns
    -------
    phi_transformed : ndarray
        Transformed parameters

    Notes
    -----
    This function is currently a placeholder for future implementation.
    Parameter transformations are useful for:
    - Ensuring positive parameters (e.g., clearance, volume)
    - Constraining parameters to intervals (e.g., bioavailability 0-1)
    - Improving numerical stability during optimization

    Examples
    --------
    >>> import numpy as np
    >>> phi = np.array([2.0, -1.0])  # log-space parameters
    >>> codes = np.array([1, 1])     # exponential transform
    >>> phi_trans = transform_parameters(phi, codes)
    >>> # phi_trans will be [exp(2.0), exp(-1.0)] = [7.39, 0.37]
    """
    if transform_codes is None:
        return phi.copy()

    phi_transformed = phi.copy()
    transform_codes = np.asarray(transform_codes)

    # Exponential transformation (for positive parameters)
    exp_mask = transform_codes == 1
    if np.any(exp_mask):
        phi_transformed[..., exp_mask] = np.exp(phi[..., exp_mask])

    # Probit transformation (normal CDF approximation)
    probit_mask = transform_codes == 2
    if np.any(probit_mask):
        # Use tanh approximation to normal CDF: Φ(x) ≈ 0.5(1 + tanh(0.7978x))
        phi_transformed[..., probit_mask] = 0.5 * (
            1 + np.tanh(phi[..., probit_mask] * 0.7978)
        )

    # Logit transformation (inverse logit)
    logit_mask = transform_codes == 3
    if np.any(logit_mask):
        phi_transformed[..., logit_mask] = 1 / (1 + np.exp(-phi[..., logit_mask]))

    return phi_transformed


def generate_design_matrix(
    n_params: int, pattern: np.ndarray | None = None
) -> np.ndarray:
    """
    Generate design matrix for fixed or random effects.

    Parameters
    ----------
    n_params : int
        Number of parameters
    pattern : ndarray, optional
        Pattern specification. If None, returns identity matrix.

    Returns
    -------
    design_matrix : ndarray
        Design matrix
    """
    if pattern is None:
        return np.eye(n_params)

    pattern = np.asarray(pattern)
    if pattern.ndim == 1:
        # Diagonal pattern
        return np.diag(pattern)
    else:
        # Full pattern matrix
        return pattern.copy()


def compute_residuals(
    y: np.ndarray,
    y_pred_pop: np.ndarray,
    y_pred_ind: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """
    Compute various types of residuals for model diagnostics.

    Parameters
    ----------
    y : ndarray
        Observed responses
    y_pred_pop : ndarray
        Population predictions (fixed effects only)
    y_pred_ind : ndarray
        Individual predictions (fixed + random effects)
    weights : ndarray, optional
        Weights for weighted residuals

    Returns
    -------
    residuals : dict
        Dictionary containing different types of residuals
    """
    residuals = {}

    # Population and individual residuals
    residuals["pres"] = y - y_pred_pop
    residuals["ires"] = y - y_pred_ind

    # Weighted residuals if weights provided
    if weights is not None:
        sqrt_weights = np.sqrt(weights)
        residuals["pwres"] = residuals["pres"] * sqrt_weights
        residuals["iwres"] = residuals["ires"] * sqrt_weights
        # Conditional weighted residuals (simplified)
        residuals["cwres"] = residuals["iwres"]
    else:
        residuals["pwres"] = residuals["pres"]
        residuals["iwres"] = residuals["ires"]
        residuals["cwres"] = residuals["ires"]

    return residuals


def compute_information_criteria(
    logl: float, n_params: int, n_groups: int
) -> tuple[float, float]:
    """
    Compute AIC and BIC for model selection.

    Parameters
    ----------
    logl : float
        Log-likelihood
    n_params : int
        Number of parameters
    n_groups : int
        Number of groups

    Returns
    -------
    aic, bic : float, float
        Akaike and Bayesian information criteria
    """
    aic = -2 * logl + 2 * n_params
    bic = -2 * logl + np.log(n_groups) * n_params
    return aic, bic
