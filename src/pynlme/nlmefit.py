"""
Main API functions for PyNLME - similar to MATLAB's nlmefit and nlmefitsa.
"""

import warnings

import numpy as np

from .data_types import ModelFunction, NLMEOptions, NLMEStats, SAEMOptions
from .first_order import fit_first_order
from .lindstrom_bates import fit_lindstrom_bates
from .model import GroupedModel, build_design
from .saem import fit_saem as _run_saem
from .utils import validate_inputs

# MATLAB name-value argument names and the NLMEOptions/SAEMOptions fields they set.
_MATLAB_NAMES = {
    "ApproximationType": "approximation_type",
    "CovParameterization": "cov_parametrization",
    "CovPattern": "cov_pattern",
    "ErrorModel": "error_model",
    "ErrorParameters": "error_parameters",
    "FEConstDesign": "fe_const_design",
    "FEGroupDesign": "fe_group_design",
    "FEObsDesign": "fe_obs_design",
    "FEParamsSelect": "fe_params_select",
    "OptimFun": "optim_fun",
    "Options": "options",
    "ParamTransform": "param_transform",
    "REConstDesign": "re_const_design",
    "REGroupDesign": "re_group_design",
    "REObsDesign": "re_obs_design",
    "REParamsSelect": "re_params_select",
    "RefineBeta0": "refine_beta0",
    "Vectorization": "vectorization",
    "DataFormat": "data_format",
    "NIterations": "n_iterations",
    "NMCMCIterations": "n_mcmc_iterations",
    "NBurnIn": "n_burn_in",
    "ComputeStdErrors": "compute_std_errors",
    "LogLikMethod": "loglik_method",
    "NChains": "n_chains",
}

# statset fields understood in the Options argument.
_STATSET_NAMES = {"MaxIter": "max_iter", "TolFun": "tol_fun", "TolX": "tol_x"}
_DISPLAY_LEVELS = {"off": 0, "final": 1, "iter": 2}

_SAEM_FIELDS = ("n_iterations", "n_mcmc_iterations", "n_burn_in", "loglik_method", "n_chains")


def _parse_options(kwargs: dict) -> tuple[dict, set[str]]:
    """Normalise MATLAB-style and Python-style options to NLMEOptions fields.

    Returns the options and the set of fields that were given by their MATLAB
    name. Parameter selections given by MATLAB name use 1-based indices, as in
    MATLAB; the Python names use 0-based indices.
    """
    parsed: dict = {}
    matlab_style: set[str] = set()
    for key, value in kwargs.items():
        field = _MATLAB_NAMES.get(key, key)
        if field in parsed:
            raise TypeError(f"Option '{key}' given more than once")
        if field != key:
            matlab_style.add(field)
        parsed[field] = value

    statset = parsed.pop("options", None)
    for key, value in (statset or {}).items():
        if key == "Display":
            parsed["verbose"] = _DISPLAY_LEVELS[str(value).lower()]
        elif key in _STATSET_NAMES or key in _STATSET_NAMES.values():
            parsed[_STATSET_NAMES.get(key, key)] = value
        else:
            raise NotImplementedError(f"Options field '{key}' is not supported")

    if isinstance(parsed.get("refine_beta0"), str):
        parsed["refine_beta0"] = parsed["refine_beta0"].lower() == "on"
    return parsed, matlab_style


def _compat(parsed: dict) -> str:
    """The convention set ("matlab" or "r"); stays in parsed for NLMEOptions."""
    compat = str(parsed.get("compat", "matlab")).lower()
    if compat not in ("matlab", "r"):
        raise ValueError(f"compat must be 'matlab' or 'r', not '{compat}'")
    parsed["compat"] = compat
    return compat


def _n_model_params(options: NLMEOptions, n_beta: int) -> int:
    """Number of model parameters p (length of phi)."""
    for design in (
        options.fe_const_design,
        options.fe_group_design,
        options.re_const_design,
        options.re_group_design,
    ):
        if design is not None:
            return np.asarray(design).shape[0]
    if options.param_transform is not None:
        return len(np.asarray(options.param_transform).reshape(-1))
    for select in (options.fe_params_select, options.re_params_select):
        if select is not None and np.asarray(select).dtype == bool:
            return len(np.asarray(select).reshape(-1))
    if options.fe_params_select is None:
        return n_beta
    raise ValueError(
        "Cannot infer the number of model parameters from an index-based "
        "FEParamsSelect; also pass ParamTransform or a logical selection"
    )


def _build_model(X, y, group, V, modelfun, beta0, options, matlab_style) -> GroupedModel:
    m = int(group.max()) + 1
    rows = [np.flatnonzero(group == i) for i in range(m)]
    p = _n_model_params(options, len(beta0))
    A = build_design(
        p, m, options.fe_params_select, options.fe_const_design,
        options.fe_group_design, "fe_params_select" in matlab_style, "FE",
    )
    B = build_design(
        p, m, options.re_params_select, options.re_const_design,
        options.re_group_design, "re_params_select" in matlab_style, "RE",
    )
    if A.shape[1] != p or B.shape[1] != p:
        raise ValueError("Fixed- and random-effects designs must have the same number of rows")
    if A.shape[2] != len(beta0):
        raise ValueError(f"beta0 has {len(beta0)} elements but the fixed-effects design has {A.shape[2]} columns")

    transform = None
    if options.param_transform is not None:
        transform = np.asarray(options.param_transform, dtype=int).reshape(-1)
        if transform.shape[0] != p or not np.all(np.isin(transform, [0, 1, 2, 3])):
            raise ValueError(f"ParamTransform must have {p} elements with values 0, 1, 2 or 3")

    # "SingleGroup" models also accept a parameter vector, so they are called
    # as "SinglePhi".
    vectorization = "full" if options.vectorization.lower() == "full" else "singlephi"
    return GroupedModel(
        vectorization=vectorization,
        X=[X[r] for r in rows],
        y=[y[r] for r in rows],
        V=V,
        A=A,
        B=B,
        transform=transform,
        modelfun=modelfun,
        rows=rows,
    )


def _full_covariance(cov_pattern, r: int) -> bool:
    """True for a full covariance pattern, False for diagonal."""
    if cov_pattern is None:
        return False
    pattern = np.asarray(cov_pattern) != 0
    if pattern.shape != (r, r):
        raise ValueError(f"CovPattern must be {r}-by-{r}")
    if np.array_equal(pattern, np.eye(r, dtype=bool)):
        return False
    if pattern.all():
        return True
    raise NotImplementedError("Only diagonal and full CovPattern are supported so far")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(message)


def nlmefit(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray]:
    """
    Fit a nonlinear mixed-effects model by maximum likelihood (MATLAB nlmefit).

    The model for group i is ``y_i = modelfun(phi_i, X_i[, V_i]) + e`` with
    ``phi_i = g^{-1}(A_i beta + B_i b_i)``, ``b_i ~ N(0, PSI)`` and
    ``e ~ N(0, sigma^2)``. Estimation uses the Lindstrom-Bates algorithm, as
    MATLAB's nlmefit and R's nlme::nlme do.

    Parameters
    ----------
    X : array_like, shape (n, h)
        Predictor variables, one row per observation.
    y : array_like, shape (n,)
        Responses.
    group : array_like, shape (n,)
        Group of each observation. Groups are ordered by sorted label.
    V : array_like, shape (m, g) or None
        Group-level predictors, one row per group. Row i is passed to
        ``modelfun`` for group i.
    modelfun : callable
        ``modelfun(phi, x)`` when V is None, otherwise ``modelfun(phi, x, v)``.
        ``phi`` is a vector of the p model parameters for one group, ``x`` is
        that group's rows of X (n_i, h) and ``v`` its row of V. Must return
        n_i predictions.
    beta0 : array_like, shape (f,)
        Initial fixed effects.
    **kwargs
        MATLAB name-value arguments (e.g. ``ParamTransform=[0, 1, 0, 1]``,
        ``REParamsSelect=[2, 3]``) or the equivalent snake_case NLMEOptions
        fields. Parameter selections use 1-based indices under the MATLAB names
        and 0-based indices under the Python names. ``compat="r"`` switches
        defaults and reporting conventions to R nlme's (no RefineBeta0, BIC
        with log(N), ML standard errors scaled by sqrt(N/(N - f)), REML
        error variance RSS/(N - f), and the full likelihood for the
        exponential error model with FO/FOCE, which MATLAB reports without
        the -sum(log y) term); explicit options still override.
        Implemented so far:
        ApproximationType "LME"/"RELME"/"FO"/"FOCE", ParamTransform, CovPattern (diagonal or
        full), FE/RE ParamsSelect, ConstDesign and GroupDesign, RefineBeta0 and
        Options (MaxIter, TolFun, TolX, Display). Other options raise
        NotImplementedError.

    Returns
    -------
    beta : ndarray, shape (f,)
        Fixed-effects estimates.
    psi : ndarray, shape (r, r)
        Random-effects covariance.
    stats : NLMEStats
        dfe, logl, aic, bic, rmse, errorparam, sebeta, covb, ires, pres, iwres.
        As in MATLAB, ``errorparam`` holds the estimated error standard
        deviation and ``rmse`` is sqrt(sum(ires^2) / dfe).
    b : ndarray, shape (r, m)
        Random-effects estimates, one column per group (MATLAB layout).
    """
    parsed, matlab_style = _parse_options(kwargs)
    compat = _compat(parsed)
    if compat == "r":
        parsed.setdefault("refine_beta0", False)  # nlme has no pre-fit
    options = NLMEOptions(**parsed)
    X, y, group, V, beta0 = validate_inputs(X, y, group, V, beta0, options.data_format)

    approx = options.approximation_type.upper()
    error_model = options.error_model.lower()
    _require(approx in ("LME", "RELME", "FO", "FOCE"), f"Unknown ApproximationType '{options.approximation_type}'")
    _require(error_model in _ERROR_MODELS, f"Unknown ErrorModel '{options.error_model}'")
    _require(options.vectorization.lower() in _VECTORIZATIONS, f"Unknown Vectorization '{options.vectorization}'")
    _require(options.fe_obs_design is None and options.re_obs_design is None, "FEObsDesign/REObsDesign are not implemented yet")

    y, modelfun, logl_offset = _apply_error_transform(error_model, y, modelfun, V)
    model = _build_model(X, y, group, V, modelfun, beta0, options, matlab_style)
    full_cov = _full_covariance(options.cov_pattern, model.n_random)
    fit_args = {
        "full_cov": full_cov,
        "refine": options.refine_beta0,
        "max_iter": options.max_iter,
        "tol_x": options.tol_x,
        "tol_fun": options.tol_fun,
        "verbose": options.verbose,
    }
    if approx in ("FO", "FOCE"):
        result = fit_first_order(
            model, beta0, conditional=approx == "FOCE", error_model=error_model,
            error_param0=options.error_parameters, **fit_args,
        )
    else:
        result = fit_lindstrom_bates(
            model, beta0, reml=approx == "RELME", error_model=error_model,
            error_param0=options.error_parameters, **fit_args,
        )
    if not result.converged:
        warnings.warn(
            f"nlmefit did not converge in {options.max_iter} iterations",
            RuntimeWarning,
            stacklevel=2,
        )

    psi, sigma2, covb = result.psi, result.sigma2, result.covb
    n, f = model.n_obs, model.n_fixed
    if compat == "r":
        # R nlme/lme conventions: BIC uses the number of observations (N - f
        # for REML); summary() inflates ML standard errors by sqrt(N/(N - f)).
        bic_groups = n - f if approx == "RELME" else n
        if approx != "RELME":
            covb = covb * n / (n - f)
    elif approx == "RELME":
        # MATLAB conventions for RELME (checked against R2023b): the error
        # variance is reported as RSS/N rather than RSS/(N - f), which
        # rescales psi and covb too, and BIC uses log(m - f).
        scale = (n - f) / n
        psi, sigma2, covb = psi * scale, sigma2 * scale, covb * scale
        bic_groups = model.n_groups - f
    else:
        bic_groups = model.n_groups
    if compat == "matlab" and approx in ("FO", "FOCE"):
        # MATLAB's FO/FOCE log-likelihood for the exponential error model
        # leaves out the -sum(log y) term that its LME/RELME include.
        logl_offset = 0.0
    stats = _build_stats(model, y, result.beta, psi, result.b, result.logl + logl_offset,
                         full_cov, error_model, sigma2, result.error_param, bic_groups)
    stats.sebeta = np.sqrt(np.diag(covb))
    stats.covb = covb
    return result.beta, psi, stats, result.b


_ERROR_MODELS = ("constant", "proportional", "combined", "exponential")
_VECTORIZATIONS = ("singlephi", "singlegroup", "full")


def _apply_error_transform(error_model, y, modelfun, V):
    """For the exponential error model, fit log(y) = log(f) + a e, as MATLAB does.

    Returns the response, model function and the log-likelihood correction
    back to the original scale (the Jacobian term -sum(log y)).
    """
    if error_model != "exponential":
        return y, modelfun, 0.0
    if np.any(y <= 0):
        raise ValueError("The exponential error model requires positive responses")
    from .model import _accepts_third_argument

    pass_none = V is None and _accepts_third_argument(modelfun)
    tiny = np.finfo(float).tiny

    def log_model(phi, x, *v):
        args = (phi, x, None) if pass_none else (phi, x, *v)
        return np.log(np.maximum(np.asarray(modelfun(*args), dtype=float), tiny))

    log_y = np.log(y)
    return log_y, log_model, -float(np.sum(log_y))


def _build_stats(model, y, beta, psi, b, logl, full_cov, error_model, sigma2, ab, bic_groups, errorparam=None):
    """MATLAB's stats structure. Residuals are on the fitted scale (log scale
    for the exponential error model), as in MATLAB."""
    from .lindstrom_bates import error_ab
    from .residuals import compute_residuals

    r = model.n_random
    n_error = 2 if error_model == "combined" else 1
    num_param = model.n_fixed + (r * (r + 1) // 2 if full_cov else r) + n_error
    fb = model.predictions(beta, b)
    if errorparam is not None:
        pass
    elif error_model == "combined":
        # As MATLAB: refit (a, b) to the final individual predictions.
        from scipy import optimize

        errorparam = np.abs(optimize.fmin(error_ab, ab, args=(y, fb), disp=False,
                                          xtol=1e-10, ftol=1e-12, maxiter=5000, maxfun=10000))
    else:
        errorparam = np.array([np.sqrt(sigma2)])
    res = compute_residuals(model, y, beta, b, psi, error_model, errorparam)
    dfe = model.n_obs - num_param
    return NLMEStats(
        dfe=dfe,
        logl=logl,
        aic=None if logl is None else -2 * logl + 2 * num_param,
        bic=None if logl is None else -2 * logl + np.log(bic_groups) * num_param,
        rmse=float(np.sqrt(res["ires"] @ res["ires"] / dfe)),
        mse=None if sigma2 is None else float(sigma2),
        errorparam=errorparam,
        ires=res["ires"],
        pres=res["pres"],
        iwres=res["iwres"],
        pwres=res["pwres"],
        cwres=res["cwres"],
    )


def nlmefitsa(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray]:
    """
    Fit a nonlinear mixed-effects model with SAEM (MATLAB nlmefitsa).

    Same model, arguments and outputs as :func:`nlmefit`, estimated by
    stochastic approximation EM, which maximises the exact likelihood rather
    than a linearised one. Results depend on the random seed (``random_state``).

    SAEM-specific name-value arguments (MATLAB defaults): NIterations
    ([150, 150, 100]), NMCMCIterations ([2, 2, 2]), NBurnIn (5), LogLikMethod
    ("none"; also "is" or "lin"), ComputeStdErrors (False). ``n_chains`` (a
    pynlme extension; default ceil(50 / m)) sets the number of MCMC chains per
    subject: more chains give less run-to-run variation on small datasets;
    ``n_chains=1`` behaves like MATLAB. Also supported:
    ParamTransform, CovPattern (diagonal or full), FEParamsSelect,
    FEConstDesign, FEGroupDesign and REParamsSelect. ``stats.logl``, ``aic`` and
    ``bic`` are None unless LogLikMethod is given; ``b`` holds the conditional
    means of the random effects.
    """
    parsed, matlab_style = _parse_options(kwargs)
    compat = _compat(parsed)
    if compat == "r":
        parsed.setdefault("n_chains", 1)  # saemix's default
    saem_kwargs = {k: parsed.pop(k) for k in _SAEM_FIELDS if k in parsed}
    compute_se = bool(parsed.pop("compute_std_errors", False))
    options = NLMEOptions(**parsed)
    saem_options = SAEMOptions(**saem_kwargs)
    error_model = options.error_model.lower()
    _require(error_model in _ERROR_MODELS, f"Unknown ErrorModel '{options.error_model}'")
    _require(options.vectorization.lower() in _VECTORIZATIONS, f"Unknown Vectorization '{options.vectorization}'")
    _require(options.fe_obs_design is None and options.re_obs_design is None, "FEObsDesign/REObsDesign are not implemented yet")

    X, y, group, V, beta0 = validate_inputs(X, y, group, V, beta0, options.data_format)
    y, modelfun, logl_offset = _apply_error_transform(error_model, y, modelfun, V)
    model = _build_model(X, y, group, V, modelfun, beta0, options, matlab_style)
    full_cov = _full_covariance(options.cov_pattern, model.n_random)
    n_iter = np.broadcast_to(np.asarray(saem_options.n_iterations, dtype=int), 3)
    n_mcmc = np.broadcast_to(np.asarray(saem_options.n_mcmc_iterations, dtype=int), 3)
    result = _run_saem(
        model,
        beta0,
        full_cov=full_cov,
        n_iterations=n_iter,
        n_mcmc=n_mcmc,
        n_burn_in=saem_options.n_burn_in,
        random_state=options.random_state,
        loglik_method=saem_options.loglik_method.lower(),
        compute_std_errors=compute_se,
        # MATLAB's default (undocumented NChains option): max(1, round(100/m))
        # chains, which stabilises SAEM on small datasets.
        n_chains=saem_options.n_chains or max(1, round(100 / model.n_groups)),
        error_model=error_model,
        error_params=options.error_parameters,
        verbose=options.verbose,
    )

    logl = None if result.logl is None else result.logl + logl_offset
    stats = _build_stats(model, y, result.beta, result.psi, result.b, logl, full_cov,
                         error_model, None, None, model.n_groups, errorparam=result.errorparam)
    stats.sebeta = None if result.covb is None else np.sqrt(np.diag(result.covb))
    stats.covb = result.covb
    return result.beta, result.psi, stats, result.b


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
    method: str = "MLE",
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
    method : str, default "MLE"
        Fitting method: "MLE" (or "ML") for Maximum Likelihood or "SAEM" for
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
    >>> # Using MLE estimation (default)
    >>> beta, psi, stats, b = fit_nlme(X, y, group, None, model, beta0)
    >>>
    >>> # Using SAEM estimation
    >>> beta, psi, stats, b = fit_nlme(X, y, group, None, model, beta0, method="SAEM")

    Notes
    -----
    This is the recommended function for Python users as it provides a unified
    interface to both algorithms. MATLAB users can use nlmefit()/nlmefitsa() directly.
    """
    if method.upper() in ["MLE", "ML"]:
        return nlmefit(X, y, group, V, modelfun, beta0, **kwargs)
    elif method.upper() == "SAEM":
        return nlmefitsa(X, y, group, V, modelfun, beta0, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'MLE' (or 'ML') or 'SAEM'.")


def fit_mle(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
    """
    Fit nonlinear mixed-effects model using Maximum Likelihood Estimation.

    Direct interface to MLE algorithm. This is equivalent to calling
    fit_nlme(..., method='ML') or nlmefit().

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
    **kwargs : keyword arguments
        Additional MLE options.

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
    >>> beta, psi, stats, b = fit_mle(X, y, group, None, model, beta0)

    Notes
    -----
    This is a convenience function that directly calls the MLE algorithm.
    For MATLAB compatibility, use nlmefit().
    """
    return nlmefit(X, y, group, V, modelfun, beta0, **kwargs)


def fit_saem(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray | list,
    V: np.ndarray | None,
    modelfun: ModelFunction,
    beta0: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, NLMEStats, np.ndarray | None]:
    """
    Fit nonlinear mixed-effects model using Stochastic Approximation EM.

    Direct interface to SAEM algorithm. This is equivalent to calling
    fit_nlme(..., method='SAEM') or nlmefitsa().

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
    **kwargs : keyword arguments
        Additional SAEM options.

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
    >>> beta, psi, stats, b = fit_saem(X, y, group, None, model, beta0)

    Notes
    -----
    This is a convenience function that directly calls the SAEM algorithm.
    For MATLAB compatibility, use nlmefitsa().
    """
    return nlmefitsa(X, y, group, V, modelfun, beta0, **kwargs)
