"""
The MATLAB nlmefit documentation example "Transform and Plot Fitted Model",
written with pynlme.

MATLAB:

    load indomethacin
    model = @(phi,t)(phi(1).*exp(-phi(2).*t)+phi(3).*exp(-phi(4).*t));
    phi0 = [1 1 1 1];
    xform = [0 1 0 1];
    [beta,psi,stats,b] = nlmefit(time,concentration,subject,[],model,phi0, ...
        ParamTransform=xform)

The data are the same as R's Indometh dataset (validation/datasets/indometh.csv).

The other documentation examples need data files that ship with MATLAB
(orangetreedata.mat, nonlineardata.mat) or nlmefitsa, which is not validated
yet. validation/matlab/run_references.m exports those files.
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pynlme import nlmefit

DATA = Path(__file__).resolve().parents[1] / "validation" / "datasets" / "indometh.csv"
OUTPUT_DIR = os.path.join(
    "examples", "matlab_documentation_examples_output", "example2_indomethacin"
)

# Printed in the MATLAB documentation (4 decimals).
MATLAB_BETA = np.array([0.4606, -1.3459, 2.8277, 0.7729])
MATLAB_PSI_DIAG = np.array([0.0124, 0.0000, 0.3264, 0.0250])


def model(phi, t):
    return phi[0] * np.exp(-phi[1] * t) + phi[2] * np.exp(-phi[3] * t)


def load_indomethacin():
    with open(DATA, newline="") as fh:
        rows = list(csv.DictReader(fh))
    time = np.array([float(r["time"]) for r in rows])
    concentration = np.array([float(r["conc"]) for r in rows])
    subject = np.array([int(r["subject"]) for r in rows])
    return time, concentration, subject


def main():
    time, concentration, subject = load_indomethacin()

    phi0 = [1, 1, 1, 1]
    xform = [0, 1, 0, 1]
    beta, psi, stats, b = nlmefit(
        time, concentration, subject, None, model, phi0, ParamTransform=xform
    )

    # The start [1 1 1 1] is symmetric in the two exponential terms, so the
    # fit may label them in the opposite order to MATLAB. Put the slow term
    # first, as MATLAB does, before comparing.
    if beta[0] > beta[2]:
        order = [2, 3, 0, 1]
        beta, psi, b = beta[order], psi[np.ix_(order, order)], b[order]

    np.set_printoptions(precision=4, suppress=True)
    print("MATLAB nlmefit documentation example: indomethacin")
    print(f"  beta           = {beta}")
    print(f"  MATLAB beta    = {MATLAB_BETA}")
    print(f"  diag(psi)      = {np.diag(psi)}")
    print(f"  MATLAB psi     = {MATLAB_PSI_DIAG}")
    print(
        f"  logl = {stats.logl:.4f}, rmse = {stats.rmse:.4f}, sebeta = {stats.sebeta}"
    )
    match = np.allclose(beta, MATLAB_BETA, atol=5e-5) and np.allclose(
        np.diag(psi), MATLAB_PSI_DIAG, atol=5e-5
    )
    print(f"  Matches MATLAB to the printed 4 decimals: {match}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tt = np.linspace(0, 8, 200)
    phi_pop = beta.copy()
    phi_pop[[1, 3]] = np.exp(phi_pop[[1, 3]])  # ParamTransform: log scale

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in np.unique(subject):
        mask = subject == s
        ax.plot(time[mask], concentration[mask], "o", label=f"subject {s}")
    ax.plot(tt, model(phi_pop, tt), "k-", lw=2, label="population fit")
    ax.set(
        xlabel="Time (hours)",
        ylabel="Concentration (mcg/ml)",
        title="Indomethacin elimination",
    )
    ax.legend()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "indomethacin_population.png"),
        dpi=150,
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(np.unique(subject)):
        mask = subject == s
        phi_i = beta + b[:, i]
        phi_i[[1, 3]] = np.exp(phi_i[[1, 3]])
        (line,) = ax.plot(time[mask], concentration[mask], "o", label=f"subject {s}")
        ax.plot(tt, model(phi_i, tt), "-", color=line.get_color())
    ax.set(
        xlabel="Time (hours)", ylabel="Concentration (mcg/ml)", title="Individual fits"
    )
    ax.legend()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "indomethacin_individual.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
