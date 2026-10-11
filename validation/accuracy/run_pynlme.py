"""
Fit the accuracy-benchmark replicates (validation/accuracy/simulate.py) with pynlme:
estimates, standard errors, variance components and error SD per replicate.

    uv run python validation/accuracy/run_pynlme.py

Writes validation/accuracy/results/pynlme.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pynlme import nlmefit, nlmefitsa  # noqa: E402
from tests.validation.cases import one_compartment_oral_v  # noqa: E402
from validation.accuracy.simulate import DATA  # noqa: E402
from validation.scaling.run_pynlme import load_csv  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
KW = {"ParamTransform": [1, 1, 1], "REParamsSelect": [2, 3], "Vectorization": "Full"}
BETA0 = [-2.5, 0.5, -3.2]

METHODS = {
    "LME, MATLAB mode": (nlmefit, {}),
    "LME, R mode": (nlmefit, {"compat": "r"}),
    "FOCE": (nlmefit, {"ApproximationType": "FOCE"}),
    # nlmefitsa computes standard errors only on request (as MATLAB's); saemix does
    # by default.
    "SAEM, MATLAB mode": (nlmefitsa, {"ComputeStdErrors": True}),
    "SAEM, saemix mode": (nlmefitsa, {"compat": "r", "ComputeStdErrors": True}),
}


def main() -> None:
    warnings.simplefilter("ignore")
    RESULTS.mkdir(exist_ok=True)
    truth = json.loads((DATA / "truth.json").read_text())
    results = {method: [] for method in METHODS}
    for k in range(truth["replicates"]):
        t, y, group, dose = load_csv(DATA / f"rep_{k}.csv")
        for method, (f, extra) in METHODS.items():
            if f is nlmefitsa:
                extra = {**extra, "random_state": k}
            try:
                beta, psi, stats, _ = f(
                    t, y, group, dose, one_compartment_oral_v, BETA0, **KW, **extra
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
        print(f"replicate {k} done", flush=True)
    (RESULTS / "pynlme.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
