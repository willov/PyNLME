"""
Replicate datasets for the accuracy benchmark: how close each program's estimates
come to the true values, and how often its 95% confidence intervals contain them.

Scenarios (the one-compartment oral model: log ke, log ka, log CL):

- basic: random effects on log ka and log CL (diagonal), additive error; 50
  subjects with 10 observations.
- hard: random effects on all three with a full covariance (correlations), a
  covariate log(wt/70) on log CL, proportional error; 60 subjects with 10
  observations.

Writes validation/accuracy/data/<scenario>/rep_{k}.csv (id, time, dose, wt, conc)
and truth.json.

    uv run python validation/accuracy/simulate.py [scenario ...]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scaling.make_data import simulate as simulate_scaling  # noqa: E402

REPLICATES = 100
DATA = Path(__file__).resolve().parent / "data"
TIMES = np.array([0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12, 24])


def _cov(sd, corr):
    sd = np.asarray(sd, dtype=float)
    return np.outer(sd, sd) * np.asarray(corr, dtype=float)


SCENARIOS = {
    "basic": {
        "subjects": 50,
        "beta": np.log([0.08, 1.5, 0.04]),  # log ke, log ka, log CL
        "random": [1, 2],  # log ka, log CL
        "omega": _cov([0.4, 0.2], np.eye(2)),
        "error": ("constant", 0.5),
        "wt_effect": 0.0,
    },
    "hard": {
        "subjects": 60,
        "beta": np.log([0.08, 1.5, 0.04]),
        "random": [0, 1, 2],
        "omega": _cov(
            [0.2, 0.4, 0.25], [[1.0, 0.0, 0.3], [0.0, 1.0, -0.2], [0.3, -0.2, 1.0]]
        ),
        "error": ("proportional", 0.15),
        "wt_effect": 0.75,  # log CL = beta_CL + 0.75 log(wt / 70)
    },
}


def oral(ke, ka, cl, dose, t):
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def simulate(spec: dict, rng) -> list[dict]:
    rows = []
    for subject in range(1, spec["subjects"] + 1):
        wt = rng.uniform(50, 100)
        dose = rng.uniform(3, 6)
        eta = spec["beta"].copy()
        eta[2] += spec["wt_effect"] * np.log(wt / 70)
        eta[spec["random"]] += rng.multivariate_normal(
            np.zeros(len(spec["random"])), spec["omega"]
        )
        f = oral(*np.exp(eta), dose, TIMES)
        kind, value = spec["error"]
        sd = value if kind == "constant" else value * np.abs(f)
        y = f + rng.normal(size=f.size) * sd
        rows += [
            {"id": subject, "time": t, "dose": dose, "wt": wt, "conc": c}
            for t, c in zip(TIMES, y, strict=True)
        ]
    return rows


def main() -> None:
    for name in sys.argv[1:] or list(SCENARIOS):
        spec = SCENARIOS[name]
        directory = DATA / name
        directory.mkdir(parents=True, exist_ok=True)
        for k in range(REPLICATES):
            if name == "basic":  # as first published (validation/scaling/make_data.py)
                rows = simulate_scaling(
                    "oral", spec["subjects"], np.random.default_rng(50_000 + k)
                )
            else:
                rows = simulate(spec, np.random.default_rng(60_000 + k))
            with open(directory / f"rep_{k}.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        truth = {
            "beta": [float(v) for v in spec["beta"]]
            + ([spec["wt_effect"]] if spec["wt_effect"] else []),
            "omega2": [float(v) for v in np.diag(spec["omega"])],
            "error": spec["error"][0],
            "sigma": spec["error"][1],
            "replicates": REPLICATES,
            "subjects": spec["subjects"],
        }
        (directory / "truth.json").write_text(json.dumps(truth, indent=1))
        print(f"{name}: {REPLICATES} datasets of {spec['subjects']} subjects")


if __name__ == "__main__":
    main()
