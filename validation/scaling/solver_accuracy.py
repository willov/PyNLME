"""
How accurate each program's ODE solver is at the benchmark's settings (relative
1e-10, absolute 1e-12; each program's method for the model: non-stiff for mm,
stiff for tmdd), so that the timings compare solvers of equal accuracy.

20 parameter sets around the simulation's values (random effects as in
make_data.py), at the observation times of the scaling datasets. The reference is
SciPy's Radau at relative 1e-13, checked against a second method at 1e-13. R
(deSolve's lsoda, rxode2) and MATLAB (ode45 for mm, ode15s for tmdd) solve the same
problems with solver_accuracy.R and solver_accuracy.m; their values are read from
results/solver_values_<program>_<model>.csv if present.

    uv run python validation/scaling/solver_accuracy.py write   # the problems
    Rscript validation/scaling/solver_accuracy.R               # R's values
    # MATLAB: addpath validation/scaling; solver_accuracy
    uv run python validation/scaling/solver_accuracy.py         # the tables

Error: mm, the largest absolute difference in concentration relative to the largest
concentration; tmdd, the largest absolute difference in log free drug (the observed
quantity, so a relative error).
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scaling.make_data import MM, TMDD  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL_FILES = HERE.parent / "models"
RTOL, ATOL = 1e-10, 1e-12


def problems(model: str) -> np.ndarray:
    """mm: rows of ka, V, Vmax, Km, dose; tmdd: rows of kel, kint, R0, dose."""
    rng = np.random.default_rng(7)
    spec = MM if model == "mm" else TMDD
    p = len(spec["beta"])
    eta = spec["beta"] + np.zeros((20, p))
    random = [1, 2] if model == "mm" else [0, 1]
    eta[:, random] += rng.normal(size=(20, 2)) * spec["sd"]
    doses = [50.0, 75.0, 100.0] if model == "mm" else [1.0, 10.0, 100.0]
    return np.column_stack([np.exp(eta), rng.choice(doses, size=20)])


def times(model: str) -> np.ndarray:
    return np.array((MM if model == "mm" else TMDD)["times"], dtype=float)


def solve(model: str, p, method, rtol, atol):
    if model == "mm":
        ka, V, vmax, km, dose = p

        def rhs(_, s):
            return [-ka * s[0], ka * s[0] / V - vmax * s[1] / (km + s[1])]

        y0 = [dose, 0.0]
    else:
        kel, kint, R0, dose = p
        kon, koff, kdeg = (TMDD["fixed"][k] for k in ("kon", "koff", "kdeg"))

        def rhs(_, s):
            binding = kon * s[0] * (R0 + s[1]) - koff * s[2]
            return [
                -kel * s[0] - binding,
                -kdeg * s[1] - binding,
                binding - kint * s[2],
            ]

        y0 = [dose, 0.0, 0.0]
    t = times(model)
    sol = solve_ivp(
        rhs, (0.0, t.max()), y0, t_eval=t, method=method, rtol=rtol, atol=atol
    )
    return sol.y[1] if model == "mm" else np.log(sol.y[0])


def write() -> None:
    (HERE / "data").mkdir(exist_ok=True)
    for model in ("mm", "tmdd"):
        path = HERE / "data" / f"solver_problems_{model}.csv"
        names = ["ka", "V", "vmax", "km"] if model == "mm" else ["kel", "kint", "R0"]
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([*names, "dose"])
            writer.writerows(problems(model))
        print(f"wrote {path}")


def python_values(model: str) -> dict:
    """SciPy and SUND at the settings: the method the benchmark uses first, the
    other one for comparison."""
    import sund

    P = problems(model)
    t = times(model)
    if model == "mm":
        out = {
            "SciPy DOP853 (pynlme ode_model)": [
                solve(model, p, "DOP853", RTOL, ATOL) for p in P
            ]
        }
        sund_methods = (("SUND Adams (pynlme)", 1.0), ("SUND BDF (not used)", 0.0))
        feature, sund_model = 0, "mm_pk"
    else:
        out = {
            "SciPy LSODA (pynlme ode_model)": [
                solve(model, p, "LSODA", RTOL, ATOL) for p in P
            ]
        }
        sund_methods = (("SUND BDF (pynlme)", 0.0),)
        feature, sund_model = 0, "tmdd"
    sund.install_model(os.path.relpath(MODEL_FILES / f"{sund_model}_sund.txt"))
    compiled = sund.import_model(sund_model)
    for name, method in sund_methods:
        sim = sund.Simulation(models=[compiled()], time_unit="h")
        sim.set_options({"rel_tol": RTOL, "abs_tol": ATOL, "method": method})
        rows = []
        for p in P:
            if model == "mm":
                params, states = p[:4], np.array([p[4], 0.0])
            else:
                fixed = [TMDD["fixed"][k] for k in ("kon", "koff", "kdeg")]
                params, states = np.r_[p[:3], fixed], np.array([p[3], 0.0, 0.0])
            sim.simulate(
                time_vector=np.r_[0.0, t], parameter_values=params, state_values=states
            )
            rows.append(np.asarray(sim.feature_values)[1:, feature])
        out[name] = rows
    return {name: np.array(v) for name, v in out.items()}


def other_values(model: str) -> dict:
    out = {}
    for program in ("r", "matlab"):
        path = RESULTS / f"solver_values_{program}_{model}.csv"
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header, values = rows[0], np.array(rows[1:], dtype=float)
        for name in dict.fromkeys(header):
            out[name] = values[:, [j for j, h in enumerate(header) if h == name]]
    return out


def main() -> None:
    if sys.argv[1:] == ["write"]:
        return write()
    errors = {}
    for model in ("mm", "tmdd"):
        P = problems(model)
        reference = np.array([solve(model, p, "Radau", 1e-13, 1e-15) for p in P])
        second = "DOP853" if model == "mm" else "LSODA"
        rows = {
            f"reference check ({second} at 1e-13)": np.array(
                [solve(model, p, second, 1e-13, 1e-15) for p in P]
            )
        }
        rows.update(python_values(model))
        rows.update(other_values(model))
        scale = np.abs(reference).max() if model == "mm" else 1.0
        errors[model] = {
            name: float(np.abs(v - reference).max() / scale) for name, v in rows.items()
        }
        print(f"\n{model}:\n\n| Solver | Error |\n|---|---|")
        for name, error in errors[model].items():
            print(f"| {name} | {error:.1e} |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "solver_accuracy.json").write_text(json.dumps(errors, indent=1))


if __name__ == "__main__":
    main()
