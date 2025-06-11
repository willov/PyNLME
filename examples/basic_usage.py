#!/usr/bin/env python3
"""
PyNLME Demo - Nonlinear Mixed-Effects Models
=============================================

This demo shows the basic usage of PyNLME for fitting nonlinear mixed-effects models
with both MLE (nlmefit) and SAEM (nlmefitsa) algorithms.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pynlme import nlmefit, nlmefitsa


def exponential_model(phi, x, v=None):
    """Two-parameter exponential decay model."""
    return phi[0] * np.exp(-phi[1] * x.ravel())


def generate_pharmacokinetic_data():
    """Generate sample pharmacokinetic data with random effects."""
    np.random.seed(42)

    # Time points
    t = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 12, 24])

    # Population parameters
    pop_clearance = 0.8  # CL/F
    pop_volume = 10.0  # Dose/C0

    # Generate data for 4 subjects
    subjects = []
    all_data = []

    for subj_id in range(1, 5):
        # Random effects (inter-individual variability)
        eta_cl = np.random.normal(0, 0.3)
        eta_v = np.random.normal(0, 0.2)

        # Individual parameters
        cl_i = pop_clearance * np.exp(eta_cl)
        v_i = pop_volume * np.exp(eta_v)

        # Concentration predictions
        conc = (100 / v_i) * np.exp(-cl_i * t)

        # Add residual error
        conc_obs = conc * np.exp(np.random.normal(0, 0.15, len(t)))

        # Store data
        for i, (time, obs) in enumerate(zip(t, conc_obs, strict=False)):
            all_data.append([subj_id, time, obs])

    data = np.array(all_data)
    return data[:, 1:2], data[:, 2], data[:, 0].astype(int) - 1  # X, y, group


def main():
    print("PyNLME Demo: Pharmacokinetic Modeling")
    print("=" * 50)

    # Generate sample data
    X, y, group = generate_pharmacokinetic_data()

    print(f"Dataset: {len(y)} observations from {len(np.unique(group))} subjects")
    print(f"Time range: {X.min():.1f} to {X.max():.1f} hours")
    print(f"Concentration range: {y.min():.2f} to {y.max():.2f} ng/mL")
    print()

    # Initial parameter estimates
    beta0 = np.array([10.0, 0.5])  # [amplitude, decay rate]

    # Fit using MLE (traditional approach)
    print("1. Maximum Likelihood Estimation (nlmefit)")
    print("-" * 45)

    beta_mle, psi_mle, stats_mle, b_mle = nlmefit(
        X, y, group, None, exponential_model, beta0, verbose=1
    )

    print("Fixed effects:")
    print(f"  Amplitude: {beta_mle[0]:.3f}")
    print(f"  Decay rate: {beta_mle[1]:.3f}")
    print("Random effects covariance matrix:")
    for i, row in enumerate(psi_mle):
        print(f"  {row}")
    print(f"Log-likelihood: {stats_mle.logl:.3f}")
    print(f"AIC: {stats_mle.aic:.3f}")
    print()

    # Fit using SAEM (stochastic approach)
    print("2. Stochastic Approximation EM (nlmefitsa)")
    print("-" * 47)

    beta_saem, psi_saem, stats_saem, b_saem = nlmefitsa(
        X, y, group, None, exponential_model, beta0, max_iter=50, verbose=1
    )

    print("Fixed effects:")
    print(f"  Amplitude: {beta_saem[0]:.3f}")
    print(f"  Decay rate: {beta_saem[1]:.3f}")
    print("Random effects covariance matrix:")
    for i, row in enumerate(psi_saem):
        print(f"  {row}")
    if stats_saem.logl is not None:
        print(f"Log-likelihood: {stats_saem.logl:.3f}")
    if stats_saem.aic is not None:
        print(f"AIC: {stats_saem.aic:.3f}")
    print()

    # Create visualization
    print("3. Visualization")
    print("-" * 16)

    plt.figure(figsize=(12, 8))

    # Plot individual data
    colors = ["blue", "red", "green", "orange"]
    unique_groups = np.unique(group)

    for i, grp in enumerate(unique_groups):
        mask = group == grp
        plt.subplot(2, 2, 1)
        plt.semilogy(
            X[mask],
            y[mask],
            "o",
            color=colors[i],
            alpha=0.7,
            label=f"Subject {grp + 1}",
        )

    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration (ng/mL)")
    plt.title("Individual Data")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot population fits
    t_pred = np.linspace(0.5, 24, 100)

    plt.subplot(2, 2, 2)
    y_mle = exponential_model(beta_mle, t_pred)
    y_saem = exponential_model(beta_saem, t_pred)

    plt.semilogy(t_pred, y_mle, "b-", label="MLE fit", linewidth=2)
    plt.semilogy(t_pred, y_saem, "r--", label="SAEM fit", linewidth=2)

    # Add data points
    for i, grp in enumerate(unique_groups):
        mask = group == grp
        plt.semilogy(X[mask], y[mask], "o", color=colors[i], alpha=0.5)

    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration (ng/mL)")
    plt.title("Population Fits Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Residuals plot
    plt.subplot(2, 2, 3)
    y_pred_mle = exponential_model(beta_mle, X)
    residuals_mle = y - y_pred_mle
    plt.scatter(y_pred_mle, residuals_mle, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("MLE Residuals")
    plt.grid(True, alpha=0.3)

    # Random effects
    plt.subplot(2, 2, 4)
    if b_mle is not None:
        plt.scatter(b_mle[0, :], b_mle[1, :], alpha=0.7, s=60)
        plt.xlabel("Random Effect 1")
        plt.ylabel("Random Effect 2")
        plt.title("Individual Random Effects")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(
            0.5,
            0.5,
            "Random effects\nnot available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()

    # Save to examples folder
    examples_dir = os.path.dirname(__file__)
    plot_path = os.path.join(examples_dir, "pynlme_demo.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved as '{plot_path}'")

    print("\nDemo completed successfully!")
    print("PyNLME provides MATLAB-compatible interfaces for NLME modeling.")


if __name__ == "__main__":
    main()
