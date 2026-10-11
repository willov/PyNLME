"""
Replicate datasets for the accuracy benchmark: how close each program's estimates
come to the true values, and how often its 95% confidence intervals contain them.

The closed-form oral model of the scaling benchmark (validation/scaling/make_data.py:
log ke, log ka, log CL; random effects on log ka and log CL; additive error), 50
subjects with 10 observations each, REPLICATES datasets from the same true values.
Writes validation/accuracy/data/rep_{k}.csv (id, time, dose, conc) and truth.json.

    uv run python validation/accuracy/simulate.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.scaling.make_data import ORAL, simulate  # noqa: E402

REPLICATES = 100
SUBJECTS = 50
DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for k in range(REPLICATES):
        rows = simulate("oral", SUBJECTS, np.random.default_rng(50_000 + k))
        with open(DATA / f"rep_{k}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "time", "dose", "conc"])
            writer.writeheader()
            writer.writerows(rows)
    truth = {
        "beta": [float(v) for v in ORAL["beta"]],  # log ke, log ka, log CL
        "omega2": [float(s) ** 2 for s in ORAL["sd"]],  # variances of log ka, log CL
        "sigma": ORAL["sigma"],
        "replicates": REPLICATES,
        "subjects": SUBJECTS,
    }
    (DATA / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"{REPLICATES} datasets of {SUBJECTS} subjects in {DATA}")


if __name__ == "__main__":
    main()
