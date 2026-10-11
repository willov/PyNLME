"""
Tables of the scaling benchmark (seconds per fit) from validation/scaling/results/.

    uv run python validation/scaling/report.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SIZES = ("10", "100", "1000")

# (row label, program, method key in its results file)
ROWS = {
    "oral": [
        ("LME", "pynlme, MATLAB mode", "pynlme", "LME, MATLAB mode"),
        ("LME", "MATLAB `nlmefit`", "matlab", "LME"),
        ("LME", "pynlme, R mode", "pynlme", "LME, R mode"),
        ("LME", "R `nlme`", "r", "LME, nlme"),
        ("FOCE", "pynlme", "pynlme", "FOCE"),
        ("FOCE", "MATLAB `nlmefit`", "matlab", "FOCE"),
        ("FOCE", "nlmixr2 (FOCEi)", "r", "FOCEi, nlmixr2"),
        ("SAEM", "pynlme `nlmefitsa`", "pynlme", "SAEM"),
        ("SAEM", "MATLAB `nlmefitsa`", "matlab", "SAEM"),
        ("SAEM", "pynlme `pynlme.r.saemix`", "pynlme", "SAEM, pynlme.r.saemix"),
        ("SAEM", "R `saemix`", "r", "SAEM, saemix"),
        ("SAEM", "nlmixr2", "r", "SAEM, nlmixr2"),
    ],
    "mm": [
        ("LME", "pynlme, MATLAB mode (SUND)", "pynlme", "LME, MATLAB mode (SUND)"),
        ("LME", "pynlme, MATLAB mode (SciPy)", "pynlme", "LME, MATLAB mode (SciPy)"),
        ("LME", "MATLAB `nlmefit` (ode45)", "matlab", "LME"),
        ("LME", "pynlme, R mode (SUND)", "pynlme", "LME, R mode (SUND)"),
        ("LME", "R `nlme` (deSolve)", "r", "LME, nlme"),
        ("FOCE", "pynlme (SUND)", "pynlme", "FOCE (SUND)"),
        ("FOCE", "MATLAB `nlmefit` (ode45)", "matlab", "FOCE"),
        ("FOCE", "nlmixr2 (FOCEi, rxode2)", "r", "FOCEi, nlmixr2"),
        ("SAEM", "pynlme `nlmefitsa` (SUND)", "pynlme", "SAEM (SUND)"),
        ("SAEM", "MATLAB `nlmefitsa` (ode45)", "matlab", "SAEM"),
        ("SAEM", "nlmixr2 (rxode2)", "r", "SAEM, nlmixr2"),
    ],
}
TITLES = {
    "oral": "Closed-form oral PK model (3 parameters, 2 random effects, 10 "
    "observations per subject)",
    "mm": "Michaelis-Menten ODE model (4 parameters, 2 random effects, 8 "
    "observations per subject)",
}


def cell(entry: dict | None) -> str:
    if entry is None:
        return "–"
    if entry.get("skipped"):
        return "skipped"
    if "error" in entry:
        return "failed"
    t = entry["time_s"]
    return f"{t:.3f}" if t < 1 else f"{t:.2f}" if t < 10 else f"{t:.0f}"


def logl_cell(entry: dict | None) -> str:
    if not entry or "logl" not in entry or entry["logl"] in (None, "NA", []):
        return "–"
    value = entry["logl"]
    return f"{value:.2f}" if isinstance(value, int | float) else "–"


def main() -> None:
    results = {
        program: json.loads((RESULTS / f"{program}.json").read_text())
        for program in ("pynlme", "matlab", "r")
        if (RESULTS / f"{program}.json").exists()
    }
    for model, rows in ROWS.items():
        print(f"{TITLES[model]}; seconds per fit:\n")
        print("| Method | Program | 10 subjects | 100 subjects | 1000 subjects |")
        print("|---|---|---|---|---|")
        for method, label, program, key in rows:
            entries = results.get(program, {}).get(model, {}).get(key, {})
            # MATLAB's JSON has keys n10, n100, ...
            cells = [cell(entries.get(n, entries.get(f"n{n}"))) for n in SIZES]
            print(f"| {method} | {label} | " + " | ".join(cells) + " |")
        print()
        print(
            "Log-likelihood each program reports, as a check that the fits are "
            "equally good (LME and FOCE; within a method the approximations "
            "agree):\n"
        )
        print("| Method | Program | 10 subjects | 100 subjects | 1000 subjects |")
        print("|---|---|---|---|---|")
        for method, label, program, key in rows:
            entries = results.get(program, {}).get(model, {}).get(key, {})
            cells = [logl_cell(entries.get(n, entries.get(f"n{n}"))) for n in SIZES]
            if method != "SAEM" and any(c != "–" for c in cells):
                print(f"| {method} | {label} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
