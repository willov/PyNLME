"""
R nlme-style interface: formulas as strings, R's names and printouts.

    from pynlme.r import nlme, pdDiag, summary, fixef, ranef, VarCorr

    fit = nlme("conc ~ dose * exp(lKe + lKa - lCl) * (exp(-exp(lKe) * time)"
               " - exp(-exp(lKa) * time)) / (exp(lKa) - exp(lKe))",
               data=theo, fixed="lKe + lKa + lCl ~ 1",
               random=pdDiag("lKa + lCl ~ 1"), groups="~subject",
               start={"lKe": -2.5, "lKa": 0.5, "lCl": -3.2})
    print(summary(fit))

Runs on ``pynlme.fit`` with ``compat="r"`` (R's standard errors and BIC). The
result is a dict; the functions ``fixef``, ``ranef``, ``VarCorr``, ``vcov``,
``logLik``, ``AIC``, ``BIC``, ``fitted``, ``resid``, ``predict`` and ``summary``
read it as their R namesakes do.

saemix: ``saemixData``, ``saemixModel``, ``saemix`` and ``coef``, ``psi``,
``phi``, ``eta``, ``predict``, ``summary`` (see ``pynlme.r_saemix``).
"""

import ast
import inspect
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .api import _column, fit
from .api import predict as _predict
from .r_saemix import (
    coef,
    eta,
    phi,
    predict_saemix,
    psi,
    saemix,
    saemixData,
    saemixModel,
    summary_saemix,
)

__all__ = [
    "AIC", "BIC", "SSasymp", "SSbiexp", "SSfol", "SSlogis", "SSmicmen", "VarCorr", "coef", "eta",
    "fitted", "fixef", "logLik", "nlme", "pdDiag", "pdSymm", "phi", "predict", "psi", "ranef", "resid",
    "residuals", "saemix", "saemixData", "saemixModel", "summary", "vcov",
]

_METHODS = {"ML": "LME", "REML": "RELME"}


# --- Model functions of R's stats package (no self-starting: give `start`) ----

def SSfol(Dose, input, lKe, lKa, lCl):  # noqa: N802, N803 (R names)
    """First-order compartment model (R's SSfol)."""
    ke, ka = np.exp(lKe), np.exp(lKa)
    return Dose * np.exp(lKe + lKa - lCl) * (np.exp(-ke * input) - np.exp(-ka * input)) / (ka - ke)


def SSlogis(input, Asym, xmid, scal):  # noqa: N802, N803
    """Logistic model (R's SSlogis)."""
    return Asym / (1 + np.exp((xmid - input) / scal))


def SSbiexp(input, A1, lrc1, A2, lrc2):  # noqa: N802, N803
    """Biexponential model (R's SSbiexp)."""
    return A1 * np.exp(-np.exp(lrc1) * input) + A2 * np.exp(-np.exp(lrc2) * input)


def SSasymp(input, Asym, R0, lrc):  # noqa: N802, N803
    """Asymptotic regression model (R's SSasymp)."""
    return Asym + (R0 - Asym) * np.exp(-np.exp(lrc) * input)


def SSmicmen(input, Vm, K):  # noqa: N802, N803
    """Michaelis-Menten model (R's SSmicmen)."""
    return Vm * input / (K + input)


_FUNCTIONS = {
    "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2, "log1p": np.log1p,
    "expm1": np.expm1, "sqrt": np.sqrt, "abs": np.abs, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "tanh": np.tanh, "pi": np.pi, "pmin": np.minimum, "pmax": np.maximum,
    "ifelse": np.where, "SSfol": SSfol, "SSlogis": SSlogis, "SSbiexp": SSbiexp,
    "SSasymp": SSasymp, "SSmicmen": SSmicmen,
}


# --- Random-effects structures -----------------------------------------------

@dataclass(frozen=True)
class _PD:
    formula: str
    structure: str  # "diag" or "symm"


def pdDiag(formula: str) -> _PD:  # noqa: N802
    """Diagonal random-effects covariance, as R's ``pdDiag``."""
    return _PD(formula, "diag")


def pdSymm(formula: str) -> _PD:  # noqa: N802
    """General (full) random-effects covariance, as R's ``pdSymm`` (the default)."""
    return _PD(formula, "symm")


# --- Formula parsing -----------------------------------------------------------

def _split(formula: str, what: str) -> tuple[str, str]:
    if formula.count("~") != 1:
        raise ValueError(f"{what} must be a formula 'lhs ~ rhs', got '{formula}'")
    lhs, rhs = formula.split("~")
    return lhs.strip(), rhs.strip()


def _terms(side: str) -> list[str]:
    return [t.strip() for t in side.split("+") if t.strip()]


def _parse_fixed(fixed) -> dict[str, list[str]]:
    """{parameter: [covariates]} from 'a + b ~ 1' or ['a ~ 1', 'b ~ wt']."""
    formulas = [fixed] if isinstance(fixed, str) else list(fixed)
    params: dict[str, list[str]] = {}
    for formula in formulas:
        lhs, rhs = _split(formula, "fixed")
        terms = _terms(rhs)
        if any(t in ("0", "-1") or t.startswith("-") for t in terms):
            raise NotImplementedError("fixed effects without an intercept are not supported")
        covariates = [t for t in terms if t != "1"]
        for t in covariates:
            if not t.isidentifier():
                raise NotImplementedError(
                    f"covariate term '{t}': compute it as a data column and use its name")
        for name in _terms(lhs):
            if name in params:
                raise ValueError(f"'{name}' appears twice in fixed")
            params[name] = covariates
    return params


def _parse_random(random) -> tuple[list[str], str, str | None]:
    """(parameters, structure, group) from a formula string or pdDiag/pdSymm."""
    structure = "symm"
    if isinstance(random, _PD):
        random, structure = random.formula, random.structure
    lhs, rhs = _split(random, "random")
    group = None
    if "|" in rhs:
        rhs, group = (s.strip() for s in rhs.split("|"))
    if _terms(rhs) != ["1"]:
        raise NotImplementedError("random effects must be 'params ~ 1' (no covariates)")
    return _terms(lhs), structure, group


def _coef_names(params: Mapping[str, list[str]]) -> list[str]:
    """R's names for the fixed effects: 'p', or 'p.(Intercept)' and 'p.cov'."""
    names = []
    for p, covs in params.items():
        if covs:
            names += [f"{p}.(Intercept)"] + [f"{p}.{c}" for c in covs]
        else:
            names.append(p)
    return names


def _rename(params: Mapping[str, list[str]]) -> dict[str, str]:
    """pynlme.fit's coefficient names -> R's."""
    out = {}
    for p, covs in params.items():
        out[p] = f"{p}.(Intercept)" if covs else p
        out.update({f"{p}.{c}": f"{p}.{c}" for c in covs})
    return out


def _in(data, name: str) -> bool:
    try:
        return name in data
    except TypeError:
        return False


# --- nlme ------------------------------------------------------------------------

def nlme(
    model: str,
    data,
    fixed,
    random=None,
    groups: str | None = None,
    start=None,
    method: str = "ML",
    **options,
) -> dict:
    """Fit a nonlinear mixed-effects model with R nlme's arguments.

    Parameters
    ----------
    model : str
        ``"response ~ expression"``. The expression uses data columns, the
        parameters in ``fixed``, NumPy-backed R functions (``exp``, ``log``,
        ``sqrt``, ...; ``^`` is a power), R's ``SSfol``, ``SSlogis``,
        ``SSbiexp``, ``SSasymp``, ``SSmicmen``, and any function defined where
        ``nlme`` is called. It must be vectorised (it is evaluated on all rows).
    data : dict of arrays or dataframe
        One row per observation (a Polars DataFrame, a dict, ...).
    fixed : str or list of str
        ``"lKe + lKa + lCl ~ 1"``, or per parameter
        ``["lKe ~ 1", "lKa ~ 1", "lCl ~ wt"]`` with group-level covariates.
    random : str or pdDiag/pdSymm, optional
        ``"lKa + lCl ~ 1"`` (general covariance, as R's default pdSymm),
        ``pdDiag("lKa + lCl ~ 1")``, optionally ``"... ~ 1 | subject"``.
        Default: all parameters, general covariance.
    groups : str
        ``"~subject"``; may instead be given in ``random`` after ``|``.
    start : dict or list
        Starting values of the fixed effects: a dict by R coefficient name
        (or parameter name for intercepts; covariate effects default to 0),
        or a list in R's coefficient order. Required, as in R.
    method : str
        "ML" (default, as R's nlme) or "REML". pynlme's REML is a true
        restricted-likelihood fit; R's nlme REML for nonlinear models is not,
        so REML estimates differ from R's.
    **options
        nlmefit options by Python name (e.g. ``error_model``, ``max_iter``).

    Returns
    -------
    dict
        ``pynlme.fit``'s result plus R's quantities: ``fixef``, ``ranef``,
        ``sigma``, ``logLik``, ``AIC``, ``BIC``, ``vcov``, ``fitted``,
        ``residuals``. Read them with the functions of the same names.
    """
    if start is None:
        raise ValueError("start is required (nlme has no self-starting models)")
    method = method.upper()
    if method not in _METHODS:
        raise ValueError("method must be 'ML' or 'REML'")
    response, expression = _split(model, "model")
    params = _parse_fixed(fixed)
    names = list(params)
    if random is None:
        random = " + ".join(names) + " ~ 1"
    re_names, structure, re_group = _parse_random(random)
    if set(re_names) - set(names):
        raise ValueError(f"random names parameters not in fixed: {sorted(set(re_names) - set(names))}")
    group = (groups or re_group or "").lstrip("~").strip()
    if not group:
        raise ValueError("give the grouping: groups='~subject' or random='... ~ 1 | subject'")

    # Starting values, by R coefficient name.
    coefs = _coef_names(params)
    if isinstance(start, Mapping):
        values = {}
        for p, covs in params.items():
            key = f"{p}.(Intercept)" if covs and f"{p}.(Intercept)" in start else p
            if key not in start:
                raise ValueError(f"start has no value for '{p}'")
            values[p] = start[key]
            values.update({f"{p}.{c}": start.get(f"{p}.{c}", 0.0) for c in covs})
        unknown = set(start) - set(coefs) - set(names)
        if unknown:
            raise ValueError(f"start names unknown coefficients: {sorted(unknown)}")
    else:
        start = list(np.asarray(start, dtype=float).reshape(-1))
        if len(start) != len(coefs):
            raise ValueError(f"start has {len(start)} values for {len(coefs)} coefficients {coefs}")
        values = dict(zip(_rename(params), start, strict=True))

    # The model expression, evaluated on all rows at once.
    source = expression.replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"cannot parse the model expression '{expression}'") from err
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    clash = [n for n in names if _in(data, n)]
    if clash:
        raise ValueError(f"parameters {clash} are also data columns")
    columns = sorted(n for n in used - set(names) if _in(data, n))
    caller = inspect.currentframe().f_back
    env = {**caller.f_globals, **caller.f_locals, **_FUNCTIONS, "__builtins__": {}}
    unresolved = sorted(used - set(names) - set(columns) - set(env))
    if unresolved:
        raise NameError(f"unknown names in the model: {unresolved} (not data columns, parameters or functions)")
    if not _in(data, response):
        raise KeyError(f"data has no column '{response}'")
    code = compile(tree, "<nlme model>", "eval")

    def evaluate(p, x):
        out = eval(code, env, {**x, **p})  # noqa: S307 (the user's own formula)
        n = len(next(iter(x.values()))) if x else len(next(iter(p.values())))
        return np.broadcast_to(np.asarray(out, dtype=float), (n,))

    result = fit(
        evaluate, data, response=response, group=group, start=values, predictors=columns,
        random=re_names, covariates={p: c for p, c in params.items() if c},
        method=_METHODS[method], compat="r", cov_pattern=np.ones((len(re_names),) * 2) if structure == "symm" else None,
        **options,
    )

    rename = _rename(params)
    st = result["stats"]
    y = _column(data, response).astype(float)
    sigma = float(np.atleast_1d(st["errorparam"])[0]) if result["error"]["model"] == "constant" else None
    n_obs, n_groups = len(y), len(result["random_effects"])
    data_name = next((k for k, v in caller.f_locals.items() if v is data), "data")
    result.update(
        fixef={rename[k]: v for k, v in result["fixed"].items()},
        se_fixef={rename[k]: v for k, v in (result["se"] or {}).items()},
        ranef={g: {rename[k]: v for k, v in b.items()} for g, b in result["random_effects"].items()},
        sigma=sigma,
        logLik=st["logl"],
        AIC=st["aic"],
        BIC=st["bic"],
        vcov=st["covb"],
        fitted=y - st["ires"],
        residuals=st["ires"],
        call={
            "engine": "nlme", "model": model, "data": data_name, "fixed": fixed, "random": random, "groups": group,
            "method": method, "structure": structure, "re_names": [rename[n] for n in re_names],
            "n_obs": n_obs, "n_groups": n_groups,
        },
    )
    return result


# --- R's accessor functions ----------------------------------------------------

def fixef(fit: dict) -> dict:
    """Fixed effects by R coefficient name."""
    return fit["fixef"]


def ranef(fit: dict) -> dict:
    """Random effects: {group: {name: value}}."""
    return fit["ranef"]


def vcov(fit: dict) -> np.ndarray:
    """Covariance matrix of the fixed effects (R's scaling)."""
    return fit["vcov"]


def logLik(fit: dict) -> float:  # noqa: N802
    return fit["logLik"]


def AIC(fit: dict) -> float:  # noqa: N802
    return fit["AIC"]


def BIC(fit: dict) -> float:  # noqa: N802
    return fit["BIC"]


def fitted(fit: dict) -> np.ndarray:
    """Individual (group-level) predictions, in the order of the data."""
    return fit["fitted"]


def resid(fit: dict, type: str = "response") -> np.ndarray:  # noqa: A002 (R's argument name)
    """Residuals ``y - fitted``; ``type="pearson"`` divides by sigma."""
    if type == "response":
        return fit["residuals"]
    if type == "pearson":
        return fit["residuals"] / fit["sigma"]
    raise ValueError("type must be 'response' or 'pearson'")


residuals = resid


def VarCorr(fit: dict) -> dict:  # noqa: N802
    """Random-effects variances, standard deviations and correlations."""
    names = fit["call"]["re_names"]
    psi = np.asarray(fit["psi"]["matrix"], dtype=float)
    sd = np.sqrt(np.diag(psi))
    out = {n: {"Variance": float(psi[k, k]), "StdDev": float(sd[k])} for k, n in enumerate(names)}
    if fit["sigma"] is not None:
        out["Residual"] = {"Variance": fit["sigma"] ** 2, "StdDev": fit["sigma"]}
    if fit["call"]["structure"] == "symm" and len(names) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["Corr"] = psi / np.outer(sd, sd)
    return out


class _Text(str):
    """A string that displays as itself (summary output at the prompt)."""

    def __repr__(self) -> str:
        return str(self)


def _table(rows: list[str], header: list[str], cells: list[list[str]]) -> list[str]:
    width = max(len(r) for r in rows)
    cols = [max(len(h), *(len(c[j]) for c in cells)) for j, h in enumerate(header)]
    lines = [" " * width + "".join(" " + h.rjust(w) for h, w in zip(header, cols, strict=True))]
    for r, c in zip(rows, cells, strict=True):
        lines.append(r.ljust(width) + "".join(" " + v.rjust(w) for v, w in zip(c, cols, strict=True)))
    return lines


def _g(v: float, digits: int = 7) -> str:
    return f"{v:.{digits}g}"


def predict(fit: dict, newdata=None, level: int = 1, type: str = "ipred") -> np.ndarray:  # noqa: A002
    """Predictions, as R's ``predict``.

    nlme fits: ``level=1`` individual (NaN for groups not in the fit, as R's
    NA), ``level=0`` population. saemix fits: ``type`` "ipred", "icpred",
    "ppred" or "ypred", as saemix's predict.
    """
    if fit["call"]["engine"] == "saemix":
        return predict_saemix(fit, newdata, type)
    return _predict(fit, newdata, level)


def summary(fit: dict) -> str:
    """R's ``summary()`` printout of an nlme or saemix fit."""
    if fit["call"]["engine"] == "saemix":
        return summary_saemix(fit)
    call = fit["call"]
    lines = [
        "Nonlinear mixed-effects model fit by "
        + ("maximum likelihood" if call["method"] == "ML" else "REML"),
        f"  Model: {call['model']}",
        f"  Data: {call['data']}",
        *("  " + line for line in _table([""], ["AIC", "BIC", "logLik"],
                                         [[_g(fit["AIC"]), _g(fit["BIC"]), _g(fit["logLik"])]])),
        "",
        "Random effects:",
        f" Formula: list({', '.join(n + ' ~ 1' for n in fit['psi']['names'])})",
        f" Level: {call['groups']}",
        " Structure: " + ("Diagonal" if call["structure"] == "diag"
                          else "General positive-definite, Log-Cholesky parametrization"),
    ]
    vc = VarCorr(fit)
    names = call["re_names"]
    if "Corr" in vc:
        rows = [*names, "Residual"]
        cells = []
        for k, n in enumerate(names):
            corr = [f"{vc['Corr'][k, j]:.3f}" for j in range(k)]
            cells.append([_g(vc[n]["StdDev"]), " ".join(corr) if k else names[0]])
        cells.append([_g(fit["sigma"]) if fit["sigma"] is not None else "", ""])
        lines += [" " + line for line in _table(rows, ["StdDev", "Corr"], cells)]
    else:
        cols = [*names, "Residual"]
        sds = [_g(vc[n]["StdDev"]) for n in names] + [_g(fit["sigma"]) if fit["sigma"] is not None else ""]
        lines += _table(["StdDev:"], cols, [sds])

    fixed = call["fixed"]
    lines += ["", "Fixed effects:  " + (fixed if isinstance(fixed, str) else f"list({', '.join(fixed)})")]
    coefs = list(fit["fixef"])
    df = call["n_obs"] - call["n_groups"] - len(coefs) + 1
    cells = []
    for c in coefs:
        value, se = fit["fixef"][c], fit["se_fixef"].get(c, np.nan)
        t = value / se
        p = 2 * sps.t.sf(abs(t), df)
        cells.append([_g(value), _g(se), str(df), _g(t), f"{p:.4f}"])
    lines += _table(coefs, ["Value", "Std.Error", "DF", "t-value", "p-value"], cells)
    if len(coefs) > 1 and fit["vcov"] is not None:
        cov = np.asarray(fit["vcov"])
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        lines.append(" Correlation: ")
        lines += _table(coefs[1:], [c[:6] for c in coefs[:-1]],
                        [[f"{corr[i, j]:.3f}" if j < i else "" for j in range(len(coefs) - 1)]
                         for i in range(1, len(coefs))])
    if fit["sigma"] is not None:
        q = np.quantile(fit["residuals"] / fit["sigma"], [0, 0.25, 0.5, 0.75, 1])
        lines += ["", "Standardized Within-Group Residuals:"]
        lines += _table([""], ["Min", "Q1", "Med", "Q3", "Max"], [[_g(v) for v in q]])
    lines += ["", f"Number of Observations: {call['n_obs']}", f"Number of Groups: {call['n_groups']}"]
    return _Text("\n".join(lines))
