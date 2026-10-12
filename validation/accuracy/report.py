"""
Tables of the accuracy benchmark from validation/accuracy/results/.

For each program and method over the replicates: bias and RMSE of the fixed effects
(log ke, log ka, log CL), the coverage of their 95% confidence intervals (estimate
+- 1.96 SE; about 95% if the standard errors are right), and the relative bias of the
random-effect variances and the error SD.

    uv run python validation/accuracy/report.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.accuracy.simulate import REPLICATES, SCENARIOS  # noqa: E402

HERE = Path(__file__).resolve().parent
TITLES = {
    "basic": "Basic: random effects on log ka and log CL, additive error",
    "hard": "Hard: random effects on all three parameters with a full covariance, "
    "log(wt/70) on log CL, proportional error",
}


def truth_of(name: str) -> dict:
    spec = SCENARIOS[name]
    return {
        "beta": list(spec["beta"]) + ([spec["wt_effect"]] if spec["wt_effect"] else []),
        "omega2": list(np.diag(spec["omega"])),
        "sigma": spec["error"][1],
        "random": spec["random"],
        "subjects": spec["subjects"],
    }


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
    for scenario in sys.argv[1:] or list(SCENARIOS):
        table(scenario)


def table(scenario: str) -> None:
    truth = truth_of(scenario)
    results = {
        program: json.loads(path.read_text())
        for program in ("pynlme", "matlab", "r")
        if (path := HERE / "results" / scenario / f"{program}.json").exists()
    }
    if not results:
        return
    names = ["log ke", "log ka", "log CL", "wt on CL"][: len(truth["beta"])]
    variances = [f"omega2 {['ke', 'ka', 'CL'][j]}" for j in truth["random"]]
    error = (
        "prop. error" if SCENARIOS[scenario]["error"][0] != "constant" else "error SD"
    )
    print(
        f"**{TITLES[scenario]}** ({REPLICATES} replicates of {truth['subjects']} "
        "subjects). Fixed effects: bias / RMSE / coverage of the 95% CI; variances "
        "and error parameter: relative bias.\n"
    )
    header = ["Method", "Program", "Fits", *names, *variances, error]
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
        for j in range(len(names)):
            cov = "–" if s["coverage"] is None else f"{100 * s['coverage'][j]:.0f}%"
            cells.append(f"{s['bias'][j]:+.3f} / {s['rmse'][j]:.3f} / {cov}")
        cells += [f"{100 * v:+.0f}%" for v in s["omega2_bias"]]
        cells.append(f"{100 * s['sigma_bias']:+.1f}%")
        print("| " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    main()
