"""
Simulated datasets for the scaling benchmark: 10, 100 and 1000 subjects.

- oral: one-compartment model with first-order absorption, closed form (as
  Theophylline): log ke, log ka, log CL; random effects on log ka and log CL.
- mm: oral dose with Michaelis-Menten elimination, an ODE (as
  validation/make_mm_dataset.py): log ka, log V, log Vmax, log Km; random effects
  on log V and log Vmax.

Additive error. Writes validation/scaling/data/{oral,mm}_{N}.csv with columns
id, time, dose, conc. Run from the repository root:

    uv run python validation/scaling/make_data.py
"""

import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

SIZES = (10, 100, 1000)
DATA = Path(__file__).resolve().parent / "data"

ORAL = {
    "beta": np.log([0.08, 1.5, 0.04]),
    "sd": [0.4, 0.2],  # log ka, log CL
    "sigma": 0.5,
    "times": [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12, 24],
}
MM = {
    "beta": np.log([1.0, 10.0, 2.0, 1.5]),
    "sd": [0.2, 0.25],  # log V, log Vmax
    "sigma": 0.15,
    "times": [0.5, 1, 2, 3, 4, 6, 8, 12],
}


def oral(ke, ka, cl, dose, t):
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def mm(ka, V, vmax, km, dose, t):
    def rhs(_, s):
        return [-ka * s[0], ka * s[0] / V - vmax * s[1] / (km + s[1])]

    sol = solve_ivp(
        rhs,
        (0.0, max(t)),
        [dose, 0.0],
        t_eval=t,
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.y[1]


def simulate(name: str, n: int, rng) -> list[dict]:
    spec = ORAL if name == "oral" else MM
    t = np.array(spec["times"], dtype=float)
    rows = []
    for subject in range(1, n + 1):
        eta = spec["beta"].copy()
        random = [1, 2]  # oral: ka, CL; mm: V, Vmax
        eta[random] += rng.normal(size=2) * spec["sd"]
        phi = np.exp(eta)
        if name == "oral":
            dose = rng.uniform(3, 6)
            f = oral(*phi, dose, t)
        else:
            dose = float(rng.choice([50.0, 75.0, 100.0]))
            f = mm(*phi, dose, t)
        y = f + rng.normal(size=t.size) * spec["sigma"]
        rows += [
            {"id": subject, "time": tk, "dose": dose, "conc": yk}
            for tk, yk in zip(t, y, strict=True)
        ]
    return rows


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for k, name in enumerate(("oral", "mm")):
        for n in SIZES:
            rows = simulate(name, n, np.random.default_rng(1000 * k + n))
            with open(DATA / f"{name}_{n}.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["id", "time", "dose", "conc"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"{name}_{n}.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
