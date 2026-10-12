"""
Validation cases shared by the reference tests and the benchmark script.

Each case mirrors the corresponding fit in validation/r/nlme_references.R and
validation/matlab/run_references.m, using MATLAB-style name-value arguments.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"
DATASETS = VALIDATION_DIR / "datasets"
REFERENCES = VALIDATION_DIR / "reference"


def load_csv(name: str) -> dict[str, np.ndarray]:
    with open(DATASETS / name, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def load_reference(name: str) -> dict:
    with open(REFERENCES / name) as fh:
        return json.load(fh)


def per_group(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """First value of a per-observation column for each group (sorted order)."""
    _, first = np.unique(group, return_index=True)
    return values[first]


@dataclass
class Case:
    name: str
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    V: np.ndarray | None
    modelfun: object
    beta0: np.ndarray
    kwargs: dict = field(default_factory=dict)


def biexponential(phi, t):
    return phi[0] * np.exp(-phi[1] * t) + phi[2] * np.exp(-phi[3] * t)


def one_compartment_oral(phi, t, dose):
    ke, ka, cl = phi
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def logistic(phi, t):
    return phi[0] / (1 + np.exp(-(t - phi[1]) / phi[2]))


def linear(phi, x):
    return phi[0] + phi[1] * x


# Models defined by an ODE, solved per subject: these cannot be vectorised.
ODE_TOL = {"rtol": 1e-10, "atol": 1e-12}


def _solve(rhs, y0, t):
    from scipy.integrate import solve_ivp

    t = np.ravel(t)
    order = np.argsort(t)
    sol = solve_ivp(
        rhs, (0.0, float(t.max())), y0, t_eval=t[order], method="LSODA", **ODE_TOL
    )
    out = np.empty_like(t, dtype=float)
    out[order] = sol.y[-1]
    return out


def michaelis_menten_ode(phi, t, dose):
    ka, V, vmax, km = phi
    return _solve(
        lambda _, s: [-ka * s[0], ka * s[0] / V - vmax * s[1] / (km + s[1])],
        [float(dose[0]), 0.0],
        t,
    )


def one_compartment_oral_ode(phi, t, dose):
    """The closed-form one_compartment_oral model, solved as an ODE."""
    ke, ka, cl = phi
    V = cl / ke
    return _solve(
        lambda _, s: [-ka * s[0], ka * s[0] / V - ke * s[1]], [float(dose[0]), 0.0], t
    )


def _mm_rhs(t, y, phi, dose):
    ka, V, vmax, km = phi.T
    gut, c = y.T
    return np.column_stack([-ka * gut, ka * gut / V - vmax * c / (km + c)])


def michaelis_menten_batched():
    """michaelis_menten_ode via pynlme.ode_model: works with Vectorization="Full"."""
    from pynlme import ode_model

    return ode_model(
        _mm_rhs,
        y0=lambda phi, dose: np.column_stack([dose[:, 0], np.zeros(len(dose))]),
        output=lambda y, phi, dose: y[:, 1],
    )


# Vectorised versions (Vectorization="Full"): phi has one row per observation,
# x is (n, h) and v is (n, g). Indexing the last axis also works for a single
# parameter vector.
def biexponential_v(phi, t):
    t = t[:, 0]
    return phi[..., 0] * np.exp(-phi[..., 1] * t) + phi[..., 2] * np.exp(
        -phi[..., 3] * t
    )


def one_compartment_oral_v(phi, t, dose):
    ke, ka, cl, t, dose = phi[..., 0], phi[..., 1], phi[..., 2], t[:, 0], dose[..., 0]
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def logistic_v(phi, t):
    return phi[..., 0] / (1 + np.exp(-(t[:, 0] - phi[..., 1]) / phi[..., 2]))


def linear_v(phi, x):
    return phi[..., 0] + phi[..., 1] * x[:, 0]


# Analytic derivatives d f / d phi (one column per parameter), same calling
# conventions as the vectorised models, for pynlme.with_jacobian.
def biexponential_jac(phi, t):
    t = t[:, 0]
    e1, e2 = np.exp(-phi[..., 1] * t), np.exp(-phi[..., 3] * t)
    return np.column_stack(
        [e1 + 0 * t, -phi[..., 0] * t * e1, e2 + 0 * t, -phi[..., 2] * t * e2]
    )


def one_compartment_oral_jac(phi, t, dose):
    ke, ka, cl, t, dose = phi[..., 0], phi[..., 1], phi[..., 2], t[:, 0], dose[..., 0]
    e1, e2 = np.exp(-ke * t), np.exp(-ka * t)
    k = ke * ka / (ka - ke)
    c = dose / cl
    f = c * k * (e1 - e2)
    d_ke = c * (ka**2 / (ka - ke) ** 2 * (e1 - e2) - k * t * e1)
    d_ka = c * (-(ke**2) / (ka - ke) ** 2 * (e1 - e2) + k * t * e2)
    return np.column_stack([d_ke, d_ka, -f / cl])


def logistic_jac(phi, t):
    t = t[:, 0]
    a, xmid, scal = phi[..., 0], phi[..., 1], phi[..., 2]
    q = np.exp(-(t - xmid) / scal)
    d = (1 + q) ** 2
    return np.column_stack(
        [1 / (1 + q), -a * q / (scal * d), -a * q * (t - xmid) / (scal**2 * d)]
    )


def linear_jac(phi, x):
    x = x[:, 0]
    return np.column_stack([np.ones_like(x), x])


JACOBIANS = {
    biexponential_v: biexponential_jac,
    one_compartment_oral_v: one_compartment_oral_jac,
    logistic_v: logistic_jac,
    linear_v: linear_jac,
}


def with_derivatives(case: Case, full: bool = True) -> Case:
    """The case with an analytic Jacobian attached (vectorised model)."""
    from pynlme import with_jacobian

    v = vectorized(case)
    model = with_jacobian(v.modelfun, JACOBIANS[v.modelfun])
    kwargs = (
        v.kwargs
        if full
        else {k: val for k, val in v.kwargs.items() if k != "Vectorization"}
    )
    return Case(
        case.name, case.X, case.y, case.group, case.V, model, case.beta0, kwargs
    )


VECTORIZED = {
    biexponential: biexponential_v,
    one_compartment_oral: one_compartment_oral_v,
    logistic: logistic_v,
    linear: linear_v,
}


def vectorized(case: Case) -> Case:
    """The same case with the vectorised model and Vectorization="Full"."""
    return Case(
        case.name,
        case.X,
        case.y,
        case.group,
        case.V,
        VECTORIZED[case.modelfun],
        case.beta0,
        {**case.kwargs, "Vectorization": "Full"},
    )


def indometh(beta0=(0.5, -1.0, 2.5, 0.5)) -> Case:
    d = load_csv("indometh.csv")
    return Case(
        "indometh",
        d["time"],
        d["conc"],
        d["subject"],
        None,
        biexponential,
        np.array(beta0),
        {"ParamTransform": [0, 1, 0, 1]},
    )


def theoph() -> Case:
    d = load_csv("theoph.csv")
    return Case(
        "theoph",
        d["time"],
        d["conc"],
        d["subject"],
        per_group(d["dose"], d["subject"]),
        one_compartment_oral,
        np.array([-2.5, 0.5, -3.2]),
        {"ParamTransform": [1, 1, 1], "REParamsSelect": [2, 3]},
    )


def theoph_wt() -> Case:
    d = load_csv("theoph.csv")
    wtc = per_group(d["wt"], d["subject"]) - 70
    A = np.zeros((3, 4, len(wtc)))
    A[0, 0] = A[1, 1] = A[2, 2] = 1
    A[2, 3] = wtc
    return Case(
        "theoph_wt",
        d["time"],
        d["conc"],
        d["subject"],
        per_group(d["dose"], d["subject"]),
        one_compartment_oral,
        np.array([-2.5, 0.5, -3.2, 0.0]),
        {"ParamTransform": [1, 1, 1], "REParamsSelect": [2, 3], "FEGroupDesign": A},
    )


def orange() -> Case:
    d = load_csv("orange.csv")
    return Case(
        "orange",
        d["age"],
        d["circumference"],
        d["tree"],
        None,
        logistic,
        np.array([192.0, 727.0, 348.0]),
        {"REParamsSelect": 1},
    )


def linear_sim() -> Case:
    d = load_csv("linear_sim.csv")
    return Case(
        "linear",
        d["x"],
        d["y"],
        d["id"],
        None,
        linear,
        np.array([1.0, 1.0]),
        {"CovPattern": np.ones((2, 2))},
    )


def mm_pk() -> Case:
    d = load_csv("mm_pk.csv")
    return Case(
        "mm_pk",
        d["time"],
        d["conc"],
        d["subject"],
        per_group(d["dose"], d["subject"]),
        michaelis_menten_ode,
        np.log([0.8, 12.0, 2.5, 1.0]),
        {"ParamTransform": [1, 1, 1, 1], "REParamsSelect": [2, 3]},
    )


def theoph_ode() -> Case:
    c = theoph()
    return Case(
        "theoph_ode",
        c.X,
        c.y,
        c.group,
        c.V,
        one_compartment_oral_ode,
        c.beta0,
        c.kwargs,
    )


# (case factory, R reference file, extra nlmefit arguments)
R_NLME_CASES = {
    "indometh_lme": (indometh, "r_nlme_indometh_lme.json", {}),
    "theoph_lme": (theoph, "r_nlme_theoph_lme.json", {}),
    "theoph_wt_lme": (theoph_wt, "r_nlme_theoph_wt_lme.json", {}),
    "orange_lme": (orange, "r_nlme_orange_lme.json", {}),
    "linear_lme": (linear_sim, "r_nlme_linear_lme.json", {}),
    # nlme's REML (compat="r" follows it); lme's true REML is checked
    # separately against the default RELME.
    "theoph_nlme_reml": (
        theoph,
        "r_nlme_theoph_nlme_reml.json",
        {"ApproximationType": "RELME"},
    ),
    "linear_nlme_reml": (
        linear_sim,
        "r_nlme_linear_nlme_reml.json",
        {"ApproximationType": "RELME"},
    ),
}
