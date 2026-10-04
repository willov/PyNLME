"""
R saemix-style interface (``saemixData``, ``saemixModel``, ``saemix``) on top of
``pynlme.fit``. Imported through ``pynlme.r``.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import optimize
from scipy import stats as sps

from .api import _INVERSE, _column, _evaluate, _label, fit

_CONTROL_MAP = {"seed", "nbiter.saemix", "nbiter.sa", "nb.chains", "nbiter.mcmc", "nbiter.burn", "ll.is",
                "nmc.is", "ll.gq", "fim"}
_CONTROL_IGNORED = {"displayProgress", "save", "save.graphs", "print", "warnings", "directory", "map",
                    "nbdisplay"}
_ERROR_MODELS = ("constant", "proportional", "combined", "exponential")
_RESPAR_NAMES = {"constant": ["a.1"], "proportional": ["b.1"], "combined": ["a.1", "b.1"],
                 "exponential": ["a.1"]}


def saemixData(name_data, name_group: str, name_predictors: Sequence[str] | str,  # noqa: N802
               name_response: str, name_covariates: Sequence[str] | str = ()) -> dict:
    """The data, as R's ``saemixData`` (arguments with ``_`` for ``.``)."""
    predictors, covariates = (
        [v] if isinstance(v, str) else list(v) for v in (name_predictors, name_covariates))
    columns = {c: _column(name_data, c) for c in [name_group, name_response, *predictors, *covariates]}
    return {"kind": "saemixData", "data": columns, "group": name_group, "predictors": predictors,
            "response": name_response, "covariates": covariates}


def saemixModel(model, psi0: Mapping, description: str = "", transform_par=None,  # noqa: N802
                covariate_model=None, covariance_model=None, error_model: str = "constant",
                omega_init=None, fixed_estim=None) -> dict:
    """The model, as R's ``saemixModel``.

    ``model(psi, id, xidep)``: ``psi`` has one column per parameter (in the
    order of ``psi0``), ``id`` indexes its rows for each observation
    (0-based), ``xidep`` has the predictors as columns. As in R, write
    ``psi[id, 0] * np.exp(-psi[id, 1] * xidep[:, 0])``.

    ``psi0``: ``{"ka": 1.0, "V": 20.0}``; a list per parameter gives
    the population value then starting values of the covariate effects, as
    the rows of R's psi0 matrix. ``covariate_model`` is R's 0/1 matrix
    (covariates x parameters) or ``{"CL": ["wt"]}``; ``covariance_model``
    R's 0/1 matrix (diagonal, or full among the random parameters).
    """
    if error_model not in _ERROR_MODELS:
        raise ValueError(f"error_model must be one of {_ERROR_MODELS}")
    if omega_init is not None:
        raise NotImplementedError("omega_init is not supported (SAEM starts from MATLAB's default)")
    if fixed_estim is not None and not np.all(np.asarray(fixed_estim) == 1):
        raise NotImplementedError("fixed_estim: all parameters must be estimated")
    names = list(psi0)
    p = len(names)
    transform = [0] * p if transform_par is None else [int(t) for t in transform_par]
    if len(transform) != p:
        raise ValueError(f"transform_par has {len(transform)} entries for {p} parameters")
    cov = np.eye(p) if covariance_model is None else np.asarray(covariance_model, dtype=float)
    if cov.shape != (p, p):
        raise ValueError(f"covariance_model must be {p} x {p}")
    random = [n for k, n in enumerate(names) if cov[k, k] != 0]
    idx = [names.index(n) for n in random]
    off = cov[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)]
    if np.any(off != 0) and not np.all(off != 0):
        raise NotImplementedError("covariance_model: only diagonal or full (among random parameters)")
    return {"kind": "saemixModel", "model": model, "psi0": dict(psi0), "names": names, "description": description,
            "transform": transform, "covariate_model": covariate_model, "random": random,
            "full": bool(np.any(off != 0)), "error_model": error_model}


def _covariate_model(spec, covariates: list[str]) -> dict[str, list[str]]:
    cm = spec["covariate_model"]
    if cm is None:
        return {}
    if isinstance(cm, Mapping):
        out = {k: list(v) for k, v in cm.items()}
    else:
        cm = np.atleast_2d(np.asarray(cm))
        if cm.shape != (len(covariates), len(spec["names"])):
            raise ValueError(f"covariate_model must be {len(covariates)} x {len(spec['names'])} "
                             f"(covariates {covariates} x parameters)")
        out = {n: [c for i, c in enumerate(covariates) if cm[i, k]] for k, n in enumerate(spec["names"])}
    for n, cs in out.items():
        if n not in spec["names"] or set(cs) - set(covariates):
            raise ValueError(f"covariate_model: unknown parameter or covariate in {n}: {cs}")
    return {n: cs for n, cs in out.items() if cs}


def _iterations(control) -> dict:
    """saemix control -> nlmefitsa options."""
    unknown = set(control) - _CONTROL_MAP - _CONTROL_IGNORED
    if unknown:
        raise ValueError(f"unsupported control options: {sorted(unknown)}")
    if control.get("ll.gq"):
        raise NotImplementedError("ll.gq (Gaussian quadrature) is not supported")
    if control.get("nmc.is", 5000) != 5000:
        raise NotImplementedError("nmc.is: importance sampling uses 5000 draws")
    k1, k2 = control.get("nbiter.saemix", (300, 100))
    sa = int(control.get("nbiter.sa", k1 // 2))
    out = {"n_iterations": (sa, int(k1) - sa, int(k2)), "n_chains": int(control.get("nb.chains", 1)),
           "loglik_method": "is" if control.get("ll.is", True) else "lin",
           "compute_std_errors": bool(control.get("fim", True))}
    if "nbiter.mcmc" in control:
        mcmc = list(control["nbiter.mcmc"])
        if len(mcmc) > 3 and mcmc[3] != 0:
            raise NotImplementedError("nbiter.mcmc: the fourth kernel is not supported")
        out["n_mcmc_iterations"] = tuple(int(v) for v in mcmc[:3])
    if "nbiter.burn" in control:
        out["n_burn_in"] = int(control["nbiter.burn"])
    if "seed" in control:
        out["random_state"] = int(control["seed"])
    return out


def saemix(model: dict, data: dict, control: Mapping | None = None) -> dict:
    """Fit by SAEM, as R's ``saemix(model, data, control)``.

    ``control`` takes saemix's names: ``seed``, ``nbiter.saemix``,
    ``nbiter.sa``, ``nb.chains`` (default 1, as saemix), ``nbiter.mcmc``,
    ``nbiter.burn``, ``ll.is``. Display and saving options are accepted and
    ignored.
    """
    if model.get("kind") != "saemixModel" or data.get("kind") != "saemixData":
        raise TypeError("use saemix(saemixModel(...), saemixData(...))")
    control = dict(control or {})
    names, user_model = model["names"], model["model"]
    covariates = _covariate_model(model, data["covariates"])
    start, transform = {}, {}
    for k, n in enumerate(names):
        v = np.atleast_1d(np.asarray(model["psi0"][n], dtype=float))
        start[n] = float(v[0])
        transform[n] = {0: "identity", 1: "log", 2: "probit", 3: "logit"}[model["transform"][k]]
        for c in covariates.get(n, ()):
            row = data["covariates"].index(c) + 1
            start[f"{n}.{c}"] = float(v[row]) if row < len(v) else 0.0

    predictors = data["predictors"]

    def wrapped(p, x):
        n = len(x[predictors[0]]) if predictors else len(next(iter(p.values())))
        psi = np.column_stack([np.broadcast_to(p[k], (n,)) for k in names])
        xidep = np.column_stack([x[c] for c in predictors]) if predictors else np.zeros((n, 0))
        return user_model(psi, np.arange(n), xidep)

    result = fit(
        wrapped, data["data"], response=data["response"], group=data["group"], start=start,
        predictors=predictors, transform=transform, random=model["random"], covariates=covariates,
        method="SAEM", compat="r", error_model=model["error_model"],
        cov_pattern=np.ones((len(model["random"]),) * 2) if model["full"] else None,
        **_iterations(control),
    )

    # saemix's quantities: population parameters on the natural scale (delta
    # method SEs), covariate effects on the transformed scale, omega, respar.
    fixed, se = result["fixed"], result["se"] or {}
    population, se_pop = {}, {}
    for k, n in enumerate(names):
        code = model["transform"][k]
        beta = fixed[n]
        population[n] = float(_INVERSE[code](beta))
        slope = {0: 1.0, 1: np.exp(beta), 2: sps.norm.pdf(beta),
                 3: _INVERSE[3](beta) * (1 - _INVERSE[3](beta))}[code]
        se_pop[n] = float(abs(slope) * se.get(n, np.nan))
    cov_effects = {f"beta_{c}({n})": fixed[f"{n}.{c}"] for n, cs in covariates.items() for c in cs}
    se_cov = {f"beta_{c}({n})": se.get(f"{n}.{c}", np.nan) for n, cs in covariates.items() for c in cs}
    respar = np.atleast_1d(result["stats"]["errorparam"])
    st = result["stats"]
    result.update(
        population=population, se_population=se_pop, covariate_effects=cov_effects, se_covariate_effects=se_cov,
        omega=result["psi"]["matrix"], respar=dict(zip(_RESPAR_NAMES[model["error_model"]], map(float, respar),
                                                       strict=True)),
        logLik=st["logl"], AIC=st["aic"], BIC=st["bic"],
        call={"engine": "saemix", "model": model, "data": data, "covariates": covariates,
              "loglik_method": "importance sampling" if control.get("ll.is", True) else "linearisation",
              "n_obs": len(data["data"][data["response"]]), "n_groups": len(result["random_effects"])},
    )
    return result


# --- Individual parameters ---------------------------------------------------

def _group_rows(fit: dict) -> dict:
    data = fit["call"]["data"]
    groups = np.array([_label(g) for g in data["data"][data["group"]]], dtype=object)
    return {g: np.flatnonzero(groups == g) for g in fit["random_effects"]}


def _subset(columns: dict, rows: np.ndarray) -> dict:
    return {k: np.asarray(v)[rows] for k, v in columns.items()}


def _phi_group(fit: dict, rows: np.ndarray, eta: Mapping[str, float]) -> dict:
    """Transformed individual parameters of one group."""
    spec, cols = fit["_spec"], fit["call"]["data"]["data"]
    out = {}
    for n in spec["names"]:
        v = fit["fixed"][n] + eta.get(n, 0.0)
        for c in spec["covariates"].get(n, ()):
            v += fit["fixed"][f"{n}.{c}"] * float(np.asarray(cols[c])[rows[0]])
        out[n] = float(v)
    return out


def _modes(fit: dict) -> dict:
    """Conditional modes (MAP) of the random effects, saemix's default."""
    if "_modes" in fit:
        return fit["_modes"]
    data = fit["call"]["data"]
    re_names = fit["psi"]["names"]
    omega_inv = np.linalg.pinv(np.asarray(fit["omega"], dtype=float))
    kind = fit["call"]["model"]["error_model"]
    par = fit["respar"]
    a, b = par.get("a.1", 0.0), par.get("b.1", 0.0)
    y_all = np.asarray(data["data"][data["response"]], dtype=float)
    out = {}
    for g, rows in _group_rows(fit).items():
        sub = _subset(data["data"], rows)
        y = y_all[rows]

        def objective(u, sub=sub, y=y, n=len(rows)):
            f = _evaluate(fit, sub, n, {k: np.full(n, u[j]) for j, k in enumerate(re_names)})
            if kind == "exponential":
                y_, f_, sd = np.log(y), np.log(np.maximum(f, 1e-300)), np.full(n, a)
            else:
                y_, f_ = y, f
                sd = {"constant": np.full(n, a), "proportional": b * np.abs(f),
                      "combined": a + b * np.abs(f)}[kind]
            sd = np.maximum(sd, 1e-12)
            val = 0.5 * np.sum(((y_ - f_) / sd) ** 2) + np.sum(np.log(sd)) + 0.5 * u @ omega_inv @ u
            return val if np.isfinite(val) else 1e300

        u0 = np.array([fit["random_effects"][g][k] for k in re_names])
        res = optimize.minimize(objective, u0, method="Nelder-Mead" if len(u0) == 1 else "BFGS",
                                options={"xatol": 1e-10, "fatol": 1e-12} if len(u0) == 1 else {"gtol": 1e-8})
        out[g] = dict(zip(re_names, map(float, res.x), strict=True))
    fit["_modes"] = out
    return out


def _individual(fit: dict, type: str, scale: str) -> dict:  # noqa: A002
    if type not in ("mode", "mean"):
        raise ValueError("type must be 'mode' or 'mean'")
    etas = _modes(fit) if type == "mode" else fit["random_effects"]
    rows = _group_rows(fit)
    if scale == "eta":
        return etas
    out = {}
    for g, eta in etas.items():
        phi = _phi_group(fit, rows[g], eta)
        if scale == "psi":
            codes = fit["_spec"]["codes"]
            phi = {n: float(_INVERSE[c](v)) for (n, v), c in zip(phi.items(), codes, strict=True)}
        out[g] = phi
    return out


def psi(fit: dict, type: str = "mode") -> dict:  # noqa: A002
    """Individual parameters on the natural scale: {group: {name: value}}.

    ``type="mode"`` (saemix's default): conditional modes (MAP);
    ``"mean"``: conditional means from SAEM.
    """
    return _individual(fit, type, "psi")


def phi(fit: dict, type: str = "mode") -> dict:  # noqa: A002
    """Individual parameters on the transformed scale."""
    return _individual(fit, type, "phi")


def eta(fit: dict, type: str = "mode") -> dict:  # noqa: A002
    """Individual random effects."""
    return _individual(fit, type, "eta")


def coef(fit: dict) -> dict:
    """saemix's coef(): fixed effects, population and individual parameters."""
    return {
        "fixed": {**fit["population"], **fit["covariate_effects"]},
        "population": fit["population"],
        "individual": {"map.psi": psi(fit, "mode"), "cond.mean.psi": psi(fit, "mean")},
    }


def predict_saemix(fit: dict, newdata=None, type: str = "ipred") -> np.ndarray:  # noqa: A002
    """saemix's predict(): ``"ipred"`` (individual, MAP), ``"icpred"``
    (individual, conditional means), ``"ppred"`` (population parameters) or
    ``"ypred"`` (mean prediction over the random effects, Monte Carlo).

    ``newdata`` needs the group column; individual predictions for groups
    not in the fit are NaN.
    """
    data = fit["call"]["data"]
    cols = data["data"] if newdata is None else {
        c: _column(newdata, c) for c in [data["group"], *data["predictors"], *data["covariates"]]}
    n = len(cols[data["group"]])
    re_names = fit["psi"]["names"]
    if type == "ppred":
        return _evaluate(fit, cols, n, None)
    if type == "ypred":
        # One draw per group and replicate, shared by the group's rows.
        rng = np.random.default_rng(0)
        groups = np.array([str(_label(g)) for g in cols[data["group"]]])
        _, inv = np.unique(groups, return_inverse=True)
        total = np.zeros(n)
        for _ in range(1000):
            d = rng.multivariate_normal(np.zeros(len(re_names)), np.asarray(fit["omega"]), size=inv.max() + 1)
            total += _evaluate(fit, cols, n, {k: d[inv, j] for j, k in enumerate(re_names)})
        return total / 1000
    if type not in ("ipred", "icpred"):
        raise ValueError("type must be 'ipred', 'icpred', 'ppred' or 'ypred'")
    etas = _modes(fit) if type == "ipred" else fit["random_effects"]
    groups = [_label(g) for g in cols[data["group"]]]
    missing = dict.fromkeys(re_names, np.nan)
    b = {k: np.array([etas.get(g, missing)[k] for g in groups]) for k in re_names}
    return _evaluate(fit, cols, n, b)


def summary_saemix(fit: dict) -> str:
    """saemix's summary() printout."""
    bar = "-" * 52

    def head(title):
        return [bar, title.center(52, "-"), bar]

    model = fit["call"]["model"]
    rows = []
    for n in model["names"]:
        est, se = fit["population"][n], fit["se_population"][n]
        rows.append([n, f"{est:.4g}", f"{se:.3g}", f"{100 * se / abs(est):.2f}", "-"])
    for k, est in fit["covariate_effects"].items():
        se = fit["se_covariate_effects"][k]
        p = 2 * sps.norm.sf(abs(est / se))
        rows.append([k, f"{est:.4g}", f"{se:.3g}", f"{100 * se / abs(est):.2f}", f"{p:.3g}"])
    for k, v in fit["respar"].items():
        rows.append([k, f"{v:.4g}", "-", "-", "-"])
    lines = head("  Fixed effects  ") + _rows(["Parameter", "Estimate", "SE", "CV(%)", "p-value"], rows, index=True)
    omega = np.asarray(fit["omega"], dtype=float)
    re = fit["psi"]["names"]
    lines += head("  Variance of random effects  ")
    lines += _rows(["Parameter", "Estimate"], [[f"omega2.{n}", f"{omega[k, k]:.3g}"] for k, n in enumerate(re)],
                   labels=re)
    sd = np.sqrt(np.diag(omega))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = omega / np.outer(sd, sd)
    lines += head("  Correlation matrix of random effects  ")
    lines += _rows([f"omega2.{n}" for n in re], [[f"{v:.2f}" for v in row] for row in corr],
                   labels=[f"omega2.{n}" for n in re])
    lines += head("  Statistical criteria  ")
    if fit["logLik"] is not None:
        lines += [f"Likelihood computed by {fit['call']['loglik_method']}",
                  f"      -2LL= {-2 * fit['logLik']:.7g}", f"      AIC = {fit['AIC']:.7g}",
                  f"      BIC = {fit['BIC']:.7g}"]
    lines.append(bar)
    from .r import _Text
    return _Text("\n".join(lines))


def _rows(header, rows, index=False, labels=None):
    labels = [str(i + 1) for i in range(len(rows))] if index else (labels or [""] * len(rows))
    width = max([len(lbl) for lbl in labels] + [0])
    cols = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(header)]
    out = [" " * width + "".join(" " + h.rjust(w) for h, w in zip(header, cols, strict=True))]
    for lbl, r in zip(labels, rows, strict=True):
        out.append(lbl.rjust(width) + "".join(" " + v.rjust(w) for v, w in zip(r, cols, strict=True)))
    return out
