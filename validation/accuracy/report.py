"""
Tables of the accuracy benchmark from validation/accuracy/results/.

For each program and method over the replicates: bias and RMSE of the fixed effects
(log ke, log ka, log CL), the coverage of their 95% confidence intervals (estimate
+- 1.96 SE; about 95% if the standard errors are right), and the relative bias of the
random-effect variances and the error SD.

    uv run python validation/accuracy/report.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["log ke", "log ka", "log CL"]

# (row label, program, method key in its results file)
ROWS = [
    ("LME", "pynlme, MATLAB mode", "pynlme", "LME, MATLAB mode"),
    ("LME", "MATLAB `nlmefit`", "matlab", "LME"),
    ("LME", "pynlme, R mode", "pynlme", "LME, R mode"),
    ("LME", "R `nlme`", "r", "LME, nlme"),
    ("FOCE", "pynlme", "pynlme", "FOCE"),
    ("FOCE", "MATLAB `nlmefit`", "matlab", "FOCE"),
    ("FOCE", "nlmixr2 (FOCEi)", "r", "FOCEi, nlmixr2"),
    ("SAEM", "pynlme, MATLAB mode", "pynlme", "SAEM, MATLAB mode"),
    ("SAEM", "MATLAB `nlmefitsa`", "matlab", "SAEM"),
    ("SAEM", "pynlme, saemix mode", "pynlme", "SAEM, saemix mode"),
    ("SAEM", "R `saemix`", "r", "SAEM, saemix"),
    ("SAEM", "nlmixr2", "r", "SAEM, nlmixr2"),
]


def summarise(fits: list, truth: dict):
    ok = [f for f in fits if isinstance(f, dict) and "beta" in f]
    beta = np.array([f["beta"] for f in ok], dtype=float)
    true_beta = np.array(truth["beta"])
    error = beta - true_beta
    with_se = [f for f in ok if f.get("se") is not None]
    covered = None
    if with_se:
        se = np.array([f["se"] for f in with_se], dtype=float)
        estimates = np.array([f["beta"] for f in with_se])
        covered = (np.abs(estimates - true_beta) <= 1.96 * se).mean(axis=0)
    omega2 = np.array([f["omega2"] for f in ok], dtype=float)
    sigma = np.array([f["sigma"] for f in ok], dtype=float)
    return {
        "fits": len(ok),
        "failed": len(fits) - len(ok),
        "bias": error.mean(axis=0),
        "rmse": np.sqrt((error**2).mean(axis=0)),
        "coverage": covered,
        "omega2_bias": omega2.mean(axis=0) / np.array(truth["omega2"]) - 1,
        "sigma_bias": sigma.mean() / truth["sigma"] - 1,
    }


def main() -> None:
    truth = json.loads((HERE / "data" / "truth.json").read_text())
    results = {
        program: json.loads((HERE / "results" / f"{program}.json").read_text())
        for program in ("pynlme", "matlab", "r")
        if (HERE / "results" / f"{program}.json").exists()
    }
    print(
        f"{truth['replicates']} replicates of {truth['subjects']} subjects. Fixed "
        "effects: bias / RMSE / coverage of the 95% CI; variances and error SD: "
        "relative bias.\n"
    )
    header = ["Method", "Program", "Fits"] + NAMES + ["omega2 ka", "omega2 CL", "sigma"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for method, label, program, key in ROWS:
        fits = results.get(program, {}).get(key)
        if not fits:
            continue
        s = summarise(fits, truth)
        cells = [
            method,
            label,
            f"{s['fits']}" + (f" ({s['failed']} failed)" if s["failed"] else ""),
        ]
        for j in range(3):
            cov = "–" if s["coverage"] is None else f"{100 * s['coverage'][j]:.0f}%"
            cells.append(f"{s['bias'][j]:+.3f} / {s['rmse'][j]:.3f} / {cov}")
        cells += [f"{100 * v:+.0f}%" for v in s["omega2_bias"]]
        cells.append(f"{100 * s['sigma_bias']:+.1f}%")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
