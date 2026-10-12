"""
How accurate each program's ODE solver is at the benchmark's settings (relative
1e-10, absolute 1e-12; each program's method for non-stiff models), so that the
timings compare solvers of equal accuracy.

The Michaelis-Menten model at 20 parameter sets around the simulation's values
(random effects on V and Vmax, three dose levels), at the observation times of the
scaling datasets. The reference is SciPy's Radau at relative 1e-13 (checked against
DOP853 at 1e-13). R (deSolve's lsoda, rxode2) and MATLAB (ode45) solve the same
problems with validation/scaling/solver_accuracy.R and solver_accuracy.m; their
values are read from results/solver_values_<program>.csv if present.

    uv run python validation/scaling/solver_accuracy.py write   # the problems
    Rscript validation/scaling/solver_accuracy.R               # R's values
    # MATLAB: addpath validation/scaling; solver_accuracy
    uv run python validation/scaling/solver_accuracy.py         # the table

Error: the largest absolute difference from the reference, relative to the largest
concentration.
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scaling.make_data import MM  # noqa: E402

HERE = Path(__file__).resolve().parent
PROBLEMS = HERE / "data" / "solver_problems.csv"
RESULTS = HERE / "results"
RTOL, ATOL = 1e-10, 1e-12
TIMES = np.array(MM["times"], dtype=float)


def problems() -> np.ndarray:
    """Rows of ka, V, Vmax, Km, dose."""
    rng = np.random.default_rng(7)
    eta = MM["beta"] + np.zeros((20, 4))
    eta[:, 1:3] += rng.normal(size=(20, 2)) * MM["sd"]
    dose = rng.choice([50.0, 75.0, 100.0], size=20)
    return np.column_stack([np.exp(eta), dose])


def rhs(_, s, ka, V, vmax, km):
    return [-ka * s[0], ka * s[0] / V - vmax * s[1] / (km + s[1])]


def solve(p, method, rtol, atol):
    sol = solve_ivp(
        rhs,
        (0.0, TIMES.max()),
        [p[4], 0.0],
        t_eval=TIMES,
        args=tuple(p[:4]),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return sol.y[1]


def write() -> None:
    PROBLEMS.parent.mkdir(exist_ok=True)
    with open(PROBLEMS, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ka", "V", "vmax", "km", "dose"])
        writer.writerows(problems())
    print(f"wrote {PROBLEMS}")


def python_values() -> dict:
    """SciPy (ode_model's DOP853) and SUND (Adams and BDF) at the settings."""
    import sund

    P = problems()
    out = {
        "SciPy DOP853 (pynlme ode_model)": np.array(
            [solve(p, "DOP853", RTOL, ATOL) for p in P]
        )
    }
    sund.install_model(os.path.relpath(HERE.parent / "models" / "mm_pk_sund.txt"))
    model = sund.import_model("mm_pk")
    for name, method in (("SUND Adams (pynlme)", 1.0), ("SUND BDF", 0.0)):
        sim = sund.Simulation(models=[model()], time_unit="h")
        sim.set_options({"rel_tol": RTOL, "abs_tol": ATOL, "method": method})
        rows = []
        for p in P:
            sim.simulate(
                time_vector=np.r_[0.0, TIMES],
                parameter_values=p[:4],
                state_values=np.array([p[4], 0.0]),
            )
            rows.append(np.asarray(sim.feature_values)[1:, 0])
        out[name] = np.array(rows)
    return out


def other_values() -> dict:
    out = {}
    for program in ("r", "matlab"):
        path = RESULTS / f"solver_values_{program}.csv"
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header, values = rows[0], np.array(rows[1:], dtype=float)
        # Columns: solver name per block of len(TIMES) values, one row per problem.
        names = list(dict.fromkeys(header))
        for name in names:
            cols = [j for j, h in enumerate(header) if h == name]
            out[name] = values[:, cols]
    return out


def main() -> None:
    if sys.argv[1:] == ["write"]:
        return write()
    P = problems()
    reference = np.array([solve(p, "Radau", 1e-13, 1e-15) for p in P])
    check = np.array([solve(p, "DOP853", 1e-13, 1e-15) for p in P])
    scale = np.abs(reference).max()
    rows = {"reference check (DOP853 at 1e-13)": check}
    rows.update(python_values())
    rows.update(other_values())
    errors = {
        name: float(np.abs(v - reference).max() / scale) for name, v in rows.items()
    }
    print("| Solver | Relative error |")
    print("|---|---|")
    for name, error in errors.items():
        print(f"| {name} | {error:.1e} |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "solver_accuracy.json").write_text(json.dumps(errors, indent=1))


if __name__ == "__main__":
    main()
