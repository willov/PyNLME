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


# Vectorised versions (Vectorization="Full"): phi has one row per observation,
# x is (n, h) and v is (n, g). Indexing the last axis also works for a single
# parameter vector.
def biexponential_v(phi, t):
    t = t[:, 0]
    return phi[..., 0] * np.exp(-phi[..., 1] * t) + phi[..., 2] * np.exp(-phi[..., 3] * t)


def one_compartment_oral_v(phi, t, dose):
    ke, ka, cl, t, dose = phi[..., 0], phi[..., 1], phi[..., 2], t[:, 0], dose[..., 0]
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def logistic_v(phi, t):
    return phi[..., 0] / (1 + np.exp(-(t[:, 0] - phi[..., 1]) / phi[..., 2]))


def linear_v(phi, x):
    return phi[..., 0] + phi[..., 1] * x[:, 0]


VECTORIZED = {biexponential: biexponential_v, one_compartment_oral: one_compartment_oral_v,
              logistic: logistic_v, linear: linear_v}


def vectorized(case: Case) -> Case:
    """The same case with the vectorised model and Vectorization="Full"."""
    return Case(case.name, case.X, case.y, case.group, case.V, VECTORIZED[case.modelfun],
                case.beta0, {**case.kwargs, "Vectorization": "Full"})


def indometh(beta0=(0.5, -1.0, 2.5, 0.5)) -> Case:
    d = load_csv("indometh.csv")
    return Case(
        "indometh", d["time"], d["conc"], d["subject"], None, biexponential,
        np.array(beta0), {"ParamTransform": [0, 1, 0, 1]},
    )


def theoph() -> Case:
    d = load_csv("theoph.csv")
    return Case(
        "theoph", d["time"], d["conc"], d["subject"],
        per_group(d["dose"], d["subject"]), one_compartment_oral,
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
        "theoph_wt", d["time"], d["conc"], d["subject"],
        per_group(d["dose"], d["subject"]), one_compartment_oral,
        np.array([-2.5, 0.5, -3.2, 0.0]),
        {"ParamTransform": [1, 1, 1], "REParamsSelect": [2, 3], "FEGroupDesign": A},
    )


def orange() -> Case:
    d = load_csv("orange.csv")
    return Case(
        "orange", d["age"], d["circumference"], d["tree"], None, logistic,
        np.array([192.0, 727.0, 348.0]), {"REParamsSelect": 1},
    )


def linear_sim() -> Case:
    d = load_csv("linear_sim.csv")
    return Case(
        "linear", d["x"], d["y"], d["id"], None, linear,
        np.array([1.0, 1.0]), {"CovPattern": np.ones((2, 2))},
    )


# (case factory, R reference file, extra nlmefit arguments)
R_NLME_CASES = {
    "indometh_lme": (indometh, "r_nlme_indometh_lme.json", {}),
    "theoph_lme": (theoph, "r_nlme_theoph_lme.json", {}),
    "theoph_wt_lme": (theoph_wt, "r_nlme_theoph_wt_lme.json", {}),
    "orange_lme": (orange, "r_nlme_orange_lme.json", {}),
    "linear_lme": (linear_sim, "r_nlme_linear_lme.json", {}),
    "linear_relme": (linear_sim, "r_nlme_linear_relme.json", {"ApproximationType": "RELME"}),
}
