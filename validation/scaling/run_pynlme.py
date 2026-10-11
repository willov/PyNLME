"""
Time pynlme on the scaling datasets (validation/scaling/make_data.py).

Each method runs with its defaults. The ODE solvers use a common tolerance (relative
1e-8, absolute 1e-10), except SUND, which keeps its tighter defaults (see mm_sund). Sizes run in increasing order; once a run
takes longer than LIMIT_S, the larger sizes are skipped (they would take ten times
longer). Runs shorter than 5 s are repeated and the best of 3 is kept.

    uv run python validation/scaling/run_pynlme.py

Writes validation/scaling/results/pynlme.json.
"""

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pynlme import nlmefit, nlmefitsa, ode_model  # noqa: E402
from tests.validation.cases import one_compartment_oral_v  # noqa: E402
from validation.scaling.make_data import DATA, SIZES  # noqa: E402

LIMIT_S = 60.0
# SCALING_SIZES=10,100 runs only these sizes (as the R and MATLAB runners).
SIZES_TO_RUN = tuple(
    int(n)
    for n in os.environ.get("SCALING_SIZES", ",".join(map(str, SIZES))).split(",")
)
RTOL, ATOL = 1e-8, 1e-10
RESULTS = Path(__file__).resolve().parent / "results"
MODELS = Path(__file__).resolve().parents[1] / "models"


def load(name: str, n: int):
    with open(DATA / f"{name}_{n}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    group = np.array([int(r["id"]) for r in rows])
    t = np.array([float(r["time"]) for r in rows])
    y = np.array([float(r["conc"]) for r in rows])
    dose = np.array([float(r["dose"]) for r in rows])
    _, first = np.unique(group, return_index=True)
    return t[:, None], y, group, dose[first][:, None]


def _mm_rhs(t, y, phi, dose):
    ka, V, vmax, km = phi.T
    gut, c = y.T
    return np.column_stack([-ka * gut, ka * gut / V - vmax * c / (km + c)])


def mm_scipy():
    return ode_model(
        _mm_rhs,
        y0=lambda phi, dose: np.column_stack([dose[:, 0], np.zeros(len(dose))]),
        output=lambda y, phi, dose: y[:, 1],
        rtol=RTOL,
        atol=ATOL,
    )


def mm_sund():
    import sund

    from pynlme import sund_model

    sund.install_model(os.path.relpath(MODELS / "mm_pk_sund.txt"))
    # SUND at its default tolerances (1e-10, 1e-12): at the common 1e-8 its
    # solver error spoils the MATLAB mode's forward-difference derivatives (the fit
    # at 1000 subjects ended 115 lower in log-likelihood).
    return sund_model("mm_pk", "conc", ["ka", "V", "vmax", "km"], v_states=["gut"])


def saemix_oral(t, y, group, dose):
    from pynlme.r import saemix, saemixData, saemixModel

    data = {
        "id": group,
        "dose": np.repeat(dose[:, 0], np.bincount(group)[1:]),
        "time": t[:, 0],
        "y": y,
    }

    def model(psi, id, x):
        ke, ka, cl = psi[id, 0], psi[id, 1], psi[id, 2]
        return (
            x[:, 0]
            * ke
            * ka
            / (cl * (ka - ke))
            * (np.exp(-ke * x[:, 1]) - np.exp(-ka * x[:, 1]))
        )

    sdata = saemixData(
        data, name_group="id", name_predictors=["dose", "time"], name_response="y"
    )
    smodel = saemixModel(
        model,
        dict(zip(["ke", "ka", "cl"], np.exp([-2.5, 0.5, -3.2]), strict=True)),
        transform_par=[1, 1, 1],
        covariance_model=np.diag([0, 1, 1]),
    )
    return lambda: saemix(smodel, sdata, {"seed": 1})


def runs(name: str, n: int):
    """(method, function returning the log-likelihood) for one dataset."""
    t, y, group, dose = load(name, n)
    if name == "oral":
        args = (t, y, group, dose, one_compartment_oral_v, [-2.5, 0.5, -3.2])
        kw = {
            "ParamTransform": [1, 1, 1],
            "REParamsSelect": [2, 3],
            "Vectorization": "Full",
        }
        models = {"": None}
    else:
        beta0 = np.log([0.8, 12.0, 2.5, 1.0])
        kw = {
            "ParamTransform": [1, 1, 1, 1],
            "REParamsSelect": [2, 3],
            "Vectorization": "Full",
        }
        models = {"SUND": mm_sund(), "SciPy": mm_scipy()}
    out = []
    for label, model in models.items():
        if model is not None:
            args = (t, y, group, dose, model, beta0)
        suffix = f" ({label})" if label else ""

        def fit(f=nlmefit, args=args, **extra):
            def run():
                _, _, stats, _ = f(*args, **kw, **extra)
                return stats.logl

            return run

        out += [
            (f"LME, MATLAB mode{suffix}", fit()),
            (f"LME, R mode{suffix}", fit(compat="r")),
            (f"FOCE{suffix}", fit(ApproximationType="FOCE")),
            (f"SAEM{suffix}", fit(nlmefitsa, random_state=1)),
        ]
    if name == "oral":
        fit_saemix = saemix_oral(t, y, group, dose)
        out.append(("SAEM, pynlme.r.saemix", lambda: None if fit_saemix() else None))
    return out


def main() -> None:
    """Runs the models named on the command line (default both); results of
    other models already in pynlme.json are kept."""
    warnings.simplefilter("ignore")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "pynlme.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in sys.argv[1:] or ("oral", "mm"):
        results[name] = {}
        too_slow = set()
        for n in SIZES_TO_RUN:
            for method, run in runs(name, n):
                entry = results.setdefault(name, {}).setdefault(method, {})
                if method in too_slow:
                    entry[str(n)] = {"skipped": True}
                    continue
                times = []
                try:
                    for _ in range(3):
                        t0 = time.perf_counter()
                        logl = run()
                        times.append(time.perf_counter() - t0)
                        if times[-1] > 5:
                            break
                except Exception as error:  # noqa: BLE001 - recorded in the results
                    entry[str(n)] = {"error": repr(error)}
                    print(f"{name} {n} {method}: {error!r}", flush=True)
                    continue
                entry[str(n)] = {"time_s": min(times), "logl": logl}
                print(f"{name} {n:>5} {method:<28} {min(times):9.3f} s", flush=True)
                if min(times) > LIMIT_S:
                    too_slow.add(method)
    path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
