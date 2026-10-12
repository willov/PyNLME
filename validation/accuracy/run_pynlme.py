"""
Fit the accuracy-benchmark replicates (validation/accuracy/simulate.py) with pynlme:
estimates, standard errors, variance components and error parameter per replicate.

    uv run python validation/accuracy/run_pynlme.py [scenario ...]

Writes validation/accuracy/results/<scenario>/pynlme.json.
"""

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pynlme import nlmefit, nlmefitsa  # noqa: E402
from tests.validation.cases import one_compartment_oral_v  # noqa: E402
from validation.accuracy.simulate import DATA, SCENARIOS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

# nlmefitsa computes standard errors only on request (as MATLAB's); saemix does by
# default.
METHODS = {
    "LME, MATLAB mode": (nlmefit, {}),
    "LME, R mode": (nlmefit, {"compat": "r"}),
    "FOCE": (nlmefit, {"ApproximationType": "FOCE"}),
    "SAEM, MATLAB mode": (nlmefitsa, {"ComputeStdErrors": True}),
    "SAEM, saemix mode": (nlmefitsa, {"compat": "r", "ComputeStdErrors": True}),
}


def load(path: Path):
    """x (time), y, group, and per group [dose, log(wt / 70)]."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    group = np.array([int(r["id"]) for r in rows])
    t = np.array([[float(r["time"])] for r in rows])
    y = np.array([float(r["conc"]) for r in rows])
    _, first = np.unique(group, return_index=True)
    dose = np.array([float(rows[i]["dose"]) for i in first])
    lwt = np.array([np.log(float(rows[i].get("wt", 70)) / 70) for i in first])
    return t, y, group, np.column_stack([dose, lwt])


def setup(scenario: str, V: np.ndarray):
    """Start values and nlmefit options for a scenario."""
    if scenario == "basic":
        return [-2.5, 0.5, -3.2], {
            "ParamTransform": [1, 1, 1],
            "REParamsSelect": [2, 3],
        }
    m = len(V)
    A = np.zeros((3, 4, m))
    A[0, 0] = A[1, 1] = A[2, 2] = 1
    A[2, 3] = V[:, 1]  # log(wt / 70) on log CL
    # The covariate effect starts at 0.5, in every program: from 0 nlmixr2's FOCEi
    # left it there.
    return [-2.5, 0.5, -3.2, 0.5], {
        "ParamTransform": [1, 1, 1],
        "FEGroupDesign": A,
        "CovPattern": np.ones((3, 3)),
        "ErrorModel": "proportional",
    }


def main() -> None:
    warnings.simplefilter("ignore")
    for scenario in sys.argv[1:] or list(SCENARIOS):
        directory = DATA / scenario
        truth = json.loads((directory / "truth.json").read_text())
        results = {method: [] for method in METHODS}
        for k in range(truth["replicates"]):
            t, y, group, V = load(directory / f"rep_{k}.csv")
            beta0, options = setup(scenario, V)
            for method, (f, extra) in METHODS.items():
                if f is nlmefitsa:
                    extra = {**extra, "random_state": k}
                try:
                    beta, psi, stats, _ = f(
                        t,
                        y,
                        group,
                        V,
                        one_compartment_oral_v,
                        beta0,
                        **options,
                        **extra,
                        Vectorization="Full",
                    )
                except Exception as error:  # noqa: BLE001 - recorded
                    results[method].append({"error": repr(error)})
                    continue
                se = stats.sebeta
                results[method].append(
                    {
                        "beta": [float(v) for v in beta],
                        "se": None if se is None else [float(v) for v in np.ravel(se)],
                        "omega2": [float(v) for v in np.diag(np.atleast_2d(psi))],
                        "sigma": float(np.ravel(stats.errorparam)[0]),
                    }
                )
            print(f"{scenario} replicate {k} done", flush=True)
        (RESULTS / scenario).mkdir(parents=True, exist_ok=True)
        (RESULTS / scenario / "pynlme.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
