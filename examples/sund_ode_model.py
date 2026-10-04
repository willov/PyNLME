"""
Fitting a nonlinear mixed-effects model defined by ODEs, compiled with SUND.

The model (validation/models/mm_pk_sund.txt): an oral dose in the gut,
first-order absorption, saturable (Michaelis-Menten) elimination. It has no
closed-form solution. SUND compiles it to C, which simulates it ~40x faster
than a SciPy ODE solve.

Requires the optional dependency:  uv add "pynlme[sund]"  (or pip install sund)
"""

import csv
import os
from pathlib import Path

import numpy as np
import sund

from pynlme import nlmefit, nlmefitsa, sund_model

ROOT = Path(__file__).resolve().parents[1]


def load_data():
    with open(ROOT / "validation" / "datasets" / "mm_pk.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    subject = np.array([int(r["subject"]) for r in rows])
    time = np.array([float(r["time"]) for r in rows])
    conc = np.array([float(r["conc"]) for r in rows])
    dose = np.array([float(r["dose"]) for r in rows])
    _, first = np.unique(subject, return_index=True)
    return time, conc, subject, dose[first]


def main():
    # Compile and load the SUND model (sund.install_model wants a relative path).
    sund.install_model(os.path.relpath(ROOT / "validation" / "models" / "mm_pk_sund.txt"))

    # phi = [ka, V, vmax, km]; the group predictor (dose) sets the initial gut
    # amount. Given the model name, sund_model simulates many subjects per call
    # using SUND containers (with Vectorization="Full").
    model = sund_model("mm_pk", feature="conc", parameters=["ka", "V", "vmax", "km"], v_states=["gut"])

    time, conc, subject, dose = load_data()
    beta0 = np.log([0.8, 12.0, 2.5, 1.0])
    options = {"ParamTransform": [1, 1, 1, 1], "REParamsSelect": [2, 3], "Vectorization": "Full"}

    beta, psi, stats, b = nlmefit(time, conc, subject, dose, model, beta0, **options)
    print("nlmefit (LME)")
    print("  ka, V, Vmax, Km =", np.round(np.exp(beta), 3), " (simulated with 1, 10, 2, 1.5)")
    print("  SD of random effects on log V, log Vmax =", np.round(np.sqrt(np.diag(psi)), 3), " (0.2, 0.25)")
    print("  error SD =", np.round(stats.errorparam, 3), " (0.15)")

    # SAEM uses the exact likelihood; it simulates every subject several times
    # per iteration, so fewer iterations here for a quick look.
    beta, psi, stats, b = nlmefitsa(time, conc, subject, dose, model, beta0, **options,
                                    NIterations=[50, 50, 50], random_state=1)
    print("nlmefitsa (SAEM, shortened)")
    print("  ka, V, Vmax, Km =", np.round(np.exp(beta), 3))


if __name__ == "__main__":
    main()
