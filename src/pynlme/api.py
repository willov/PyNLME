"""
A Pythonic interface: named parameters, dict (or Polars) data, dict results.

    result = fit(model, data, response="conc", group="subject",
                 predictors=["time"], group_predictors=["dose"],
                 start={"ke": 0.08, "ka": 1.6, "cl": 0.04},
                 transform={"ke": "log", "ka": "log", "cl": "log"},
                 random=["ka", "cl"])

The model receives dicts of arrays, ``model(p, x)`` or ``model(p, x, v)``:
``p["ka"]``, ``x["time"]``, ``v["dose"]``. Each value has one entry per
observation (or is a scalar), so ordinary NumPy expressions work and the model
is evaluated for many groups in one call (Vectorization="Full").

Runs on nlmefit/nlmefitsa; every option of those (e.g. ``error_model``,
``cov_pattern``, ``compat``, ``options``) can be passed as a keyword.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.special import expit, logit, ndtr, ndtri

from .nlmefit import nlmefit, nlmefitsa

_TRANSFORMS = {None: 0, "none": 0, "identity": 0, "log": 1, "probit": 2, "logit": 3}
_FORWARD = {0: lambda v: v, 1: np.log, 2: ndtri, 3: logit}
_INVERSE = {0: lambda v: v, 1: np.exp, 2: ndtr, 3: expit}
_NAMES = {0: "identity", 1: "log", 2: "probit", 3: "logit"}
_METHODS = {"LME", "RELME", "FO", "FOCE", "SAEM"}


def _column(data, name: str) -> np.ndarray:
    """A column of a dict of arrays or of a dataframe (Polars, pandas, ...)."""
    try:
        values = data[name]
    except (KeyError, IndexError, TypeError) as err:
        raise KeyError(f"data has no column '{name}'") from err
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values)


def fit(
    model: Callable,
    data,
    response: str,
    group: str,
    start: Mapping[str, float],
    predictors: Sequence[str] = (),
    group_predictors: Sequence[str] = (),
    transform: Mapping[str, str] | None = None,
    random: Sequence[str] | None = None,
    covariates: Mapping[str, Sequence[str]] | None = None,
    method: str = "LME",
    **options,
) -> dict:
    """Fit a nonlinear mixed-effects model with named parameters.

    Parameters
    ----------
    model : callable
        ``model(p, x)`` or, with ``group_predictors``, ``model(p, x, v)``;
        ``p``, ``x`` and ``v`` are dicts of arrays keyed by parameter,
        predictor and group-predictor names. Returns the predictions.
    data : dict of arrays or dataframe
        One row per observation. Anything indexable by column name works
        (a dict, a Polars or pandas DataFrame).
    response, group : str
        Columns with the observations and the group (subject) labels.
    start : dict
        Starting values per parameter, on the natural scale. The order of
        the keys is the parameter order.
    predictors : list of str
        Observation-level columns passed to the model as ``x``.
    group_predictors : list of str
        Columns that are constant within a group (e.g. the dose), passed as
        ``v``. Taken from each group's first row.
    transform : dict, optional
        Per parameter: "log", "logit", "probit" or "identity" (default).
        Estimation is on the transformed scale.
    random : list of str, optional
        Parameters with random effects (default: all).
    covariates : dict, optional
        Group-level covariate models: ``{"cl": ["wt"]}`` makes the
        transformed ``cl`` equal ``beta_cl + beta_cl.wt * wt`` (plus the random
        effect). The coefficients are named ``"cl.wt"``; they start at 0 unless
        given in ``start`` (on the transformed scale).
    method : str
        "LME" (default), "RELME", "FO", "FOCE" or "SAEM".
    **options
        Further nlmefit/nlmefitsa options by their Python names, e.g.
        ``error_model="proportional"``, ``cov_pattern=np.ones((2, 2))``,
        ``random_state=1``, ``compat="r"``.

    Returns
    -------
    dict
        ``fixed`` (estimates on the transformed scale, covariate coefficients
        included), ``typical`` (parameters on the natural scale, at covariate
        values 0), ``se`` (standard errors of ``fixed``),
        ``psi`` (random-effects covariance: ``names``, ``matrix``,
        ``variances``), ``random_effects`` ({group: {parameter: b}}),
        ``error`` (error-model parameters), ``stats`` (all of nlmefit's
        statistics as a dict), ``method``.
    """
    covariates = {n: list(c) for n, c in (covariates or {}).items()}
    effects = {f"{n}.{c}" for n, cs in covariates.items() for c in cs}
    names = [n for n in start if n not in effects]
    if set(covariates) - set(names):
        raise ValueError(
            "covariates names unknown parameters: "
            f"{sorted(set(covariates) - set(names))}"
        )
    method = method.upper()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, not '{method}'")
    transform = dict(transform or {})
    unknown = set(transform) - set(names)
    if unknown:
        raise ValueError(f"transform names unknown parameters: {sorted(unknown)}")
    codes = []
    for name in names:
        kind = transform.get(name)
        if kind not in _TRANSFORMS:
            raise ValueError(f"unknown transform '{kind}' for '{name}'")
        codes.append(_TRANSFORMS[kind])
    fixed_names, beta0 = [], []
    for n, c in zip(names, codes, strict=True):
        fixed_names.append(n)
        beta0.append(_FORWARD[c](float(start[n])))
        for cov in covariates.get(n, ()):
            fixed_names.append(f"{n}.{cov}")
            beta0.append(float(start.get(f"{n}.{cov}", 0.0)))
    beta0 = np.array(beta0)

    random = names if random is None else list(random)
    missing = set(random) - set(names)
    if missing:
        raise ValueError(f"random names unknown parameters: {sorted(missing)}")
    select = np.array([n in random for n in names])

    y = _column(data, response).astype(float)
    groups = _column(data, group)
    X = (
        np.column_stack([_column(data, c).astype(float) for c in predictors])
        if predictors
        else np.zeros((len(y), 0))
    )
    V = None
    if group_predictors:
        V = np.column_stack([_per_group(data, groups, c) for c in group_predictors])
    if covariates:
        m = len(np.unique(groups))
        A = np.zeros((len(names), len(fixed_names), m))
        for k, n in enumerate(names):
            A[k, fixed_names.index(n)] = 1
            for cov in covariates.get(n, ()):
                A[k, fixed_names.index(f"{n}.{cov}")] = _per_group(data, groups, cov)
        options = {"fe_group_design": A, **options}

    def wrapped(phi, x, v=None):
        phi = np.asarray(phi)
        p = {n: phi[..., k] for k, n in enumerate(names)}
        xd = {c: x[:, k] for k, c in enumerate(predictors)}
        if group_predictors:
            vd = {c: np.asarray(v)[..., k] for k, c in enumerate(group_predictors)}
            return model(p, xd, vd)
        return model(p, xd)

    if hasattr(model, "jacobian"):
        # d f / d phi: pass the jacobian through with the same dict arguments.
        def jacobian(phi, x, v=None):
            phi = np.asarray(phi)
            p = {n: phi[..., k] for k, n in enumerate(names)}
            xd = {c: x[:, k] for k, c in enumerate(predictors)}
            if group_predictors:
                return model.jacobian(
                    p,
                    xd,
                    {c: np.asarray(v)[..., k] for k, c in enumerate(group_predictors)},
                )
            return model.jacobian(p, xd)

        wrapped.jacobian = jacobian

    kwargs = {
        "param_transform": codes,
        "re_params_select": select,
        "vectorization": "full",
        **options,
    }
    if method == "SAEM":
        beta, psi, stats, b = nlmefitsa(X, y, groups, V, wrapped, beta0, **kwargs)
    else:
        beta, psi, stats, b = nlmefit(
            X, y, groups, V, wrapped, beta0, approximation_type=method, **kwargs
        )

    labels = np.unique(groups)
    se = stats.sebeta
    return {
        "method": method,
        "fixed": dict(zip(fixed_names, map(float, beta), strict=True)),
        "typical": {
            n: float(_INVERSE[c](beta[fixed_names.index(n)]))
            for n, c in zip(names, codes, strict=True)
        },
        "se": None
        if se is None
        else dict(zip(fixed_names, map(float, se), strict=True)),
        "transform": {n: _NAMES[c] for n, c in zip(names, codes, strict=True)},
        "psi": {
            "names": random,
            "matrix": psi,
            "variances": {n: float(psi[k, k]) for k, n in enumerate(random)},
        },
        "random_effects": {
            _label(g): {n: float(b[k, j]) for k, n in enumerate(random)}
            for j, g in enumerate(labels)
        },
        "error": {
            "model": options.get("error_model", "constant"),
            "parameters": stats.errorparam,
        },
        "stats": dict(vars(stats)),
        # What predict() needs; not part of the documented result.
        "_spec": {
            "model": model,
            "data": data,
            "names": names,
            "codes": codes,
            "covariates": covariates,
            "predictors": list(predictors),
            "group_predictors": list(group_predictors),
            "group": group,
        },
    }


def predict(result: dict, data=None, level: int = 1) -> np.ndarray:
    """Predictions of a ``fit`` result.

    Parameters
    ----------
    result : dict
        The output of ``fit``.
    data : dict of arrays or dataframe, optional
        Rows to predict, with the predictor, group-predictor and covariate
        columns (and the group column for ``level=1``). Default: the fitted data.
    level : int
        0: population predictions (random effects 0). 1: individual
        predictions with each group's estimated random effects; NaN for groups
        that were not in the fit.
    """
    spec = result["_spec"]
    data = spec["data"] if data is None else data
    if level not in (0, 1):
        raise ValueError("level must be 0 or 1")
    n = len(
        _column(data, spec["predictors"][0] if spec["predictors"] else spec["group"])
    )
    b = None
    if level == 1:
        groups = [_label(g) for g in _column(data, spec["group"])]
        re = result["random_effects"]
        missing = dict.fromkeys(result["psi"]["names"], np.nan)
        b = {
            n: np.array([re.get(g, missing)[n] for g in groups])
            for n in result["psi"]["names"]
        }
    return _evaluate(result, data, n, b)


def _evaluate(
    result: dict, data, n: int, b: Mapping[str, np.ndarray] | None, fixed=None
) -> np.ndarray:
    """Model predictions for the rows of ``data`` with per-row random effects ``b``."""
    spec = result["_spec"]
    fixed = result["fixed"] if fixed is None else fixed
    p = {}
    for name, code in zip(spec["names"], spec["codes"], strict=True):
        eta = np.full(n, fixed[name])
        for cov in spec["covariates"].get(name, ()):
            eta = eta + fixed[f"{name}.{cov}"] * _column(data, cov).astype(float)
        if b is not None and name in b:
            eta = eta + b[name]
        p[name] = _INVERSE[code](eta)
    x = {c: _column(data, c).astype(float) for c in spec["predictors"]}
    if spec["group_predictors"]:
        out = spec["model"](
            p, x, {c: _column(data, c).astype(float) for c in spec["group_predictors"]}
        )
    else:
        out = spec["model"](p, x)
    return np.broadcast_to(np.asarray(out, dtype=float), (n,)).copy()


def _per_group(data, groups: np.ndarray, column: str) -> np.ndarray:
    """A group-level column, one value per group (groups in sorted order)."""
    labels, first = np.unique(groups, return_index=True)
    values = _column(data, column).astype(float)
    per_group = values[first]
    if not np.allclose(
        values, per_group[np.searchsorted(labels, groups)], equal_nan=True
    ):
        raise ValueError(f"'{column}' is not constant within groups")
    return per_group


def _label(value):
    """Group labels as plain Python values (dict keys)."""
    return value.item() if hasattr(value, "item") else value
