"""
Simulate the Michaelis-Menten PK dataset (validation/datasets/mm_pk.csv).

Oral dose, first-order absorption, saturable (Michaelis-Menten) elimination:

    dA/dt = -ka A,            A(0) = dose
    dC/dt = ka A / V - Vmax C / (Km + C),  C(0) = 0

There is no closed-form solution, so the model needs an ODE solve per
subject: an example of a model that cannot be vectorised. Random effects on
log V and log Vmax. Run from the repository root:

    uv run python validation/make_mm_dataset.py
"""

import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

TRUTH = {
    "beta": np.log([1.0, 10.0, 2.0, 1.5]),  # log ka (1/h), log V (L), log Vmax (mg/h), log Km (mg/L)
    "sd": [0.2, 0.25],  # random effects on log V and log Vmax
    "sigma": 0.15,  # mg/L
}


def concentrations(ka, V, vmax, km, dose, t):
    def rhs(_, s):
        a, c = s
        return [-ka * a, ka * a / V - vmax * c / (km + c)]

    sol = solve_ivp(rhs, (0.0, float(t.max())), [dose, 0.0], t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
    return sol.y[1]


def main() -> None:
    rng = np.random.default_rng(20261004)
    rows = []
    for subject in range(1, 17):
        dose = rng.choice([50.0, 75.0, 100.0])
        t = np.sort(rng.choice([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 24], size=rng.integers(7, 11), replace=False))
        eta = TRUTH["beta"].copy()
        eta[1:3] += rng.normal(0, TRUTH["sd"])
        ka, V, vmax, km = np.exp(eta)
        c = concentrations(ka, V, vmax, km, dose, t)
        y = c + rng.normal(0, TRUTH["sigma"], len(t))
        rows += [(subject, dose, ti, round(yi, 6)) for ti, yi in zip(t, y, strict=True)]
    out = Path(__file__).parent / "datasets" / "mm_pk.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "dose", "time", "conc"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} observations)")


if __name__ == "__main__":
    main()
