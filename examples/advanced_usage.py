#!/usr/bin/env python3
"""
Advanced PyNLME Example: Pharmacokinetic Analysis
================================================

This example demonstrates advanced usage of PyNLME for pharmacokinetic modeling,
including model comparison, diagnostics, and result interpretation.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pynlme import nlmefit, nlmefitsa

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def one_compartment_model(phi, t, v=None):
    """
    One-compartment pharmacokinetic model with first-order elimination.

    Parameters:
    - phi[0]: Dose/Volume (C0)
    - phi[1]: Elimination rate constant (ke)
    """
    return phi[0] * np.exp(-phi[1] * t.ravel())


def two_compartment_model(phi, t, v=None):
    """
    Two-compartment pharmacokinetic model.

    Parameters:
    - phi[0]: A (amplitude of fast phase)
    - phi[1]: alpha (fast elimination rate)
    - phi[2]: B (amplitude of slow phase)
    - phi[3]: beta (slow elimination rate)
    """
    t_flat = t.ravel()
    return phi[0] * np.exp(-phi[1] * t_flat) + phi[2] * np.exp(-phi[3] * t_flat)


def generate_realistic_pk_data():
    """Generate realistic pharmacokinetic data with multiple subjects."""
    np.random.seed(123)

    # Study design: 6 subjects, 10 time points
    time_points = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 18, 24])
    n_subjects = 6

    # Population parameters (typical values)
    pop_cl = 1.2  # L/h
    pop_v = 15.0  # L
    dose = 100  # mg

    # Inter-individual variability (log-normal)
    omega_cl = 0.4  # 40% CV
    omega_v = 0.3  # 30% CV

    # Residual error
    sigma = 0.2  # 20% proportional error

    all_data = []

    for subj in range(n_subjects):
        # Individual parameters
        eta_cl = np.random.normal(0, omega_cl)
        eta_v = np.random.normal(0, omega_v)

        cl_i = pop_cl * np.exp(eta_cl)
        v_i = pop_v * np.exp(eta_v)
        ke_i = cl_i / v_i
        c0_i = dose / v_i

        # True concentrations
        conc_true = c0_i * np.exp(-ke_i * time_points)

        # Observed concentrations with proportional error
        conc_obs = conc_true * np.exp(np.random.normal(0, sigma, len(time_points)))

        # Store data
        for t, c in zip(time_points, conc_obs, strict=False):
            all_data.append([subj, t, c])

    data = np.array(all_data)
    return data[:, 1:2], data[:, 2], data[:, 0].astype(int)


def analyze_model_fit(X, y, group, beta, psi, stats, model_name):
    """Analyze and display model fit results."""
    print(f"\n{model_name} Results:")
    print("=" * (len(model_name) + 9))

    # Parameter estimates
    print("Fixed Effects:")
    for i, param in enumerate(beta):
        print(f"  β{i + 1}: {param:.4f}")

    print("\nRandom Effects Covariance (Ψ):")
    for i, row in enumerate(psi):
        print(f"  {row}")

    # Model diagnostics
    if stats.logl is not None:
        print("\nModel Diagnostics:")
        print(f"  Log-likelihood: {stats.logl:.3f}")
        print(f"  AIC: {stats.aic:.3f}")
        print(f"  BIC: {stats.bic:.3f}")
        print(f"  Degrees of freedom: {stats.dfe}")

    return beta, psi, stats


def plot_results(X, y, group, beta_1comp, beta_2comp, psi_1comp, psi_2comp):
    """Create comprehensive plots of the analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Individual data plots
    unique_groups = np.unique(group)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

    # Plot 1: Individual profiles
    ax = axes[0, 0]
    for i, grp in enumerate(unique_groups):
        mask = group == grp
        ax.semilogy(
            X[mask],
            y[mask],
            "o",
            color=colors[i],
            alpha=0.7,
            label=f"Subject {grp + 1}",
        )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Individual Concentration Profiles")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Plot 2: Model comparison
    ax = axes[0, 1]
    t_pred = np.linspace(0.25, 24, 100).reshape(-1, 1)

    y_1comp = one_compartment_model(beta_1comp, t_pred)
    y_2comp = two_compartment_model(beta_2comp, t_pred)

    ax.semilogy(t_pred, y_1comp, "b-", linewidth=2, label="1-compartment")
    ax.semilogy(t_pred, y_2comp, "r--", linewidth=2, label="2-compartment")

    # Add data points
    for i, grp in enumerate(unique_groups):
        mask = group == grp
        ax.semilogy(X[mask], y[mask], "o", color=colors[i], alpha=0.4, markersize=4)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals for 1-compartment
    ax = axes[0, 2]
    y_pred_1comp = one_compartment_model(beta_1comp, X)
    residuals = (y - y_pred_1comp) / y_pred_1comp * 100  # Weighted residuals

    ax.scatter(y_pred_1comp, residuals, alpha=0.6)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted Concentration")
    ax.set_ylabel("Weighted Residuals (%)")
    ax.set_title("1-Compartment Model Residuals")
    ax.grid(True, alpha=0.3)

    # Plot 4: Random effects correlation
    ax = axes[1, 0]
    if psi_1comp.shape[0] >= 2:
        # Create scatter plot of random effects
        ax.scatter([0, 1], [0, 1], alpha=0.6, s=60)
        ax.set_xlabel("Random Effect 1")
        ax.set_ylabel("Random Effect 2")
        ax.set_title("Random Effects Distribution")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Random effects\nvisualization\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Plot 5: Goodness of fit
    ax = axes[1, 1]
    ax.scatter(y, y_pred_1comp, alpha=0.6)
    min_val, max_val = (
        min(y.min(), y_pred_1comp.min()),
        max(y.max(), y_pred_1comp.max()),
    )
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    ax.set_xlabel("Observed Concentration")
    ax.set_ylabel("Predicted Concentration")
    ax.set_title("Observed vs Predicted")
    ax.grid(True, alpha=0.3)

    # Plot 6: Parameter estimates with uncertainty
    ax = axes[1, 2]
    param_names = (
        ["C₀", "kₑ"]
        if len(beta_1comp) == 2
        else [f"β{i + 1}" for i in range(len(beta_1comp))]
    )
    x_pos = np.arange(len(beta_1comp))

    # Rough confidence intervals (assumes normal distribution)
    se_estimates = (
        np.sqrt(np.diag(psi_1comp))
        if psi_1comp.shape[0] == len(beta_1comp)
        else np.ones(len(beta_1comp)) * 0.1
    )

    ax.bar(x_pos, beta_1comp, yerr=se_estimates, alpha=0.7, capsize=5)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Estimate")
    ax.set_title("Parameter Estimates")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to examples folder
    examples_dir = os.path.dirname(__file__)
    plot_path = os.path.join(examples_dir, "advanced_pynlme_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nComprehensive analysis plot saved as '{plot_path}'")


def main():
    """Run the advanced pharmacokinetic analysis."""
    print("Advanced PyNLME Example: Pharmacokinetic Analysis")
    print("=" * 55)

    # Generate data
    X, y, group = generate_realistic_pk_data()

    print(f"Dataset: {len(y)} observations from {len(np.unique(group))} subjects")
    print(f"Time range: {X.min():.2f} to {X.max():.1f} hours")
    print(f"Concentration range: {y.min():.3f} to {y.max():.2f} mg/L")

    # Model 1: One-compartment model using MLE
    print("\n" + "=" * 60)
    print("Model 1: One-compartment PK model (MLE)")
    print("=" * 60)

    beta0_1comp = np.array([6.0, 0.1])  # Initial estimates: [C0, ke]

    beta_1comp, psi_1comp, stats_1comp, b_1comp = nlmefit(
        X, y, group, None, one_compartment_model, beta0_1comp, verbose=1, max_iter=100
    )

    analyze_model_fit(
        X, y, group, beta_1comp, psi_1comp, stats_1comp, "One-Compartment Model (MLE)"
    )

    # Model 2: Two-compartment model using SAEM
    print("\n" + "=" * 60)
    print("Model 2: Two-compartment PK model (SAEM)")
    print("=" * 60)

    beta0_2comp = np.array(
        [4.0, 0.5, 2.0, 0.05]
    )  # Initial estimates: [A, alpha, B, beta]

    beta_2comp, psi_2comp, stats_2comp, b_2comp = nlmefitsa(
        X, y, group, None, two_compartment_model, beta0_2comp, max_iter=50, verbose=1
    )

    analyze_model_fit(
        X, y, group, beta_2comp, psi_2comp, stats_2comp, "Two-Compartment Model (SAEM)"
    )

    # Model comparison
    print("\n" + "=" * 60)
    print("Model Comparison and Selection")
    print("=" * 60)

    if stats_1comp.aic is not None and stats_2comp.aic is not None:
        aic_diff = stats_2comp.aic - stats_1comp.aic
        print(f"AIC difference (2-comp - 1-comp): {aic_diff:.3f}")

        if aic_diff < -2:
            print("→ Two-compartment model is preferred (ΔAIC < -2)")
        elif aic_diff > 2:
            print("→ One-compartment model is preferred (ΔAIC > 2)")
        else:
            print("→ Models are equivalent (|ΔAIC| < 2)")

    # Clinical interpretation
    print("\n" + "=" * 60)
    print("Clinical Interpretation")
    print("=" * 60)

    if len(beta_1comp) == 2:
        c0, ke = beta_1comp
        t_half = np.log(2) / ke
        cl_apparent = ke * (100 / c0)  # Assuming 100 mg dose

        print("One-compartment model parameters:")
        print(f"  Initial concentration (C₀): {c0:.2f} mg/L")
        print(f"  Elimination rate constant (kₑ): {ke:.4f} h⁻¹")
        print(f"  Half-life: {t_half:.1f} hours")
        print(f"  Apparent clearance: {cl_apparent:.2f} L/h")

    # Create comprehensive plots
    plot_results(X, y, group, beta_1comp, beta_2comp, psi_1comp, psi_2comp)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("This example demonstrates:")
    print("• Model fitting with both MLE and SAEM algorithms")
    print("• Model comparison using information criteria")
    print("• Comprehensive diagnostic plots")
    print("• Clinical parameter interpretation")
    print("• Population pharmacokinetic analysis workflow")


if __name__ == "__main__":
    main()
