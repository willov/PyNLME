"""
Example usage of the Python nlmefitsa implementation.
"""

import numpy as np

from pynlme import nlmefit, nlmefitsa


def exponential_decay_model(phi, t, v=None):
    """
    Exponential decay model: y = phi[0] * exp(-phi[1] * t)

    This is a common pharmacokinetic model where:
    - phi[0] is the initial concentration (amplitude)
    - phi[1] is the elimination rate constant
    - t is time

    Parameters
    ----------
    phi : array_like, shape (2,)
        Parameters [amplitude, decay_rate]
    t : array_like
        Time points
    v : None
        Group-level covariates (not used in this example)

    Returns
    -------
    y : ndarray
        Model predictions
    """
    if hasattr(t, "ravel"):
        t = t.ravel()
    if len(phi) < 2:
        raise ValueError("phi must have at least 2 parameters")

    # Ensure positive parameters
    amplitude = max(phi[0], 1e-6)
    decay_rate = max(phi[1], 1e-6)

    return amplitude * np.exp(-decay_rate * t)


def two_compartment_model(phi, t, v=None):
    """
    Two-compartment pharmacokinetic model.

    Model: y = phi[0] * exp(-phi[1] * t) + phi[2] * exp(-phi[3] * t)

    Parameters
    ----------
    phi : array_like, shape (4,)
        Parameters [A1, alpha, A2, beta] where:
        - A1, A2: amplitudes for fast and slow phases
        - alpha, beta: elimination rates for fast and slow phases
    t : array_like
        Time points
    v : None
        Group-level covariates

    Returns
    -------
    y : ndarray
        Model predictions
    """
    if hasattr(t, "ravel"):
        t = t.ravel()
    if len(phi) < 4:
        raise ValueError("phi must have at least 4 parameters")

    # Ensure positive parameters
    A1 = max(phi[0], 1e-6)
    alpha = max(phi[1], 1e-6)
    A2 = max(phi[2], 1e-6)
    beta = max(phi[3], 1e-6)

    return A1 * np.exp(-alpha * t) + A2 * np.exp(-beta * t)


def generate_sample_data(
    n_subjects=5,
    n_timepoints=10,
    noise_level=0.5,
    true_params=None,
    random_seed=42,
    model_type="exponential",
):
    """
    Generate sample data for nonlinear mixed effects modeling.

    Parameters
    ----------
    n_subjects : int
        Number of subjects
    n_timepoints : int
        Number of time points per subject
    noise_level : float
        Standard deviation of observation noise
    true_params : dict, optional
        True population parameters
    random_seed : int
        Random seed for reproducibility
    model_type : str
        Type of model ('exponential' or 'two_compartment')

    Returns
    -------
    X : ndarray
        Predictor variables (time)
    y : ndarray
        Response variable (concentration)
    group : ndarray
        Grouping variable (subject ID)
    V : None
        Group-level covariates (not used)
    """
    if true_params is None:
        if model_type == "exponential":
            true_params = {"amplitude": 10.0, "decay_rate": 0.5}
        else:  # two_compartment
            true_params = {"A1": 8.0, "alpha": 1.5, "A2": 2.0, "beta": 0.2}

    np.random.seed(random_seed)

    # Time points
    t_max = 8.0 if model_type == "exponential" else 12.0
    t = np.linspace(0.5, t_max, n_timepoints)  # Start at 0.5 to avoid issues at t=0

    # Create grouped data
    time_all = np.tile(t, n_subjects)
    group_all = np.repeat(range(n_subjects), n_timepoints)

    # Generate individual parameters with random effects
    if model_type == "exponential":
        amplitude_pop = true_params["amplitude"]
        decay_rate_pop = true_params["decay_rate"]

        # Random effects (small variations around population values)
        amplitude_re = np.random.normal(0, 2.0, n_subjects)
        decay_rate_re = np.random.normal(0, 0.1, n_subjects)

        # Individual parameters
        amplitude_indiv = amplitude_pop + amplitude_re
        decay_rate_indiv = decay_rate_pop + decay_rate_re

        # Ensure positive parameters
        amplitude_indiv = np.maximum(amplitude_indiv, 1.0)
        decay_rate_indiv = np.maximum(decay_rate_indiv, 0.1)

        # Generate responses
        y_all = np.zeros(len(time_all))
        for i in range(n_subjects):
            mask = group_all == i
            phi_i = [amplitude_indiv[i], decay_rate_indiv[i]]
            y_all[mask] = exponential_decay_model(phi_i, time_all[mask])

    else:  # two_compartment
        A1_pop = true_params["A1"]
        alpha_pop = true_params["alpha"]
        A2_pop = true_params["A2"]
        beta_pop = true_params["beta"]

        # Random effects
        A1_re = np.random.normal(0, 1.5, n_subjects)
        alpha_re = np.random.normal(0, 0.2, n_subjects)
        A2_re = np.random.normal(0, 0.5, n_subjects)
        beta_re = np.random.normal(0, 0.05, n_subjects)

        # Individual parameters
        A1_indiv = np.maximum(A1_pop + A1_re, 1.0)
        alpha_indiv = np.maximum(alpha_pop + alpha_re, 0.5)
        A2_indiv = np.maximum(A2_pop + A2_re, 0.5)
        beta_indiv = np.maximum(beta_pop + beta_re, 0.05)

        # Generate responses
        y_all = np.zeros(len(time_all))
        for i in range(n_subjects):
            mask = group_all == i
            phi_i = [A1_indiv[i], alpha_indiv[i], A2_indiv[i], beta_indiv[i]]
            y_all[mask] = two_compartment_model(phi_i, time_all[mask])

    # Add proportional + additive noise
    y_all_noisy = y_all * (1 + np.random.normal(0, noise_level * 0.1, len(y_all)))
    y_all_noisy += np.random.normal(0, noise_level, len(y_all))

    # Ensure positive concentrations
    y_all_noisy = np.maximum(y_all_noisy, 0.01)

    return time_all.reshape(-1, 1), y_all_noisy, group_all, None


def compare_algorithms():
    """Compare MLE (nlmefit) vs SAEM (nlmefitsa) algorithms."""
    print("=== Comparing MLE vs SAEM Algorithms ===")

    # Generate data
    X, y, group, v = generate_sample_data(
        n_subjects=8,
        n_timepoints=12,
        noise_level=0.3,
        true_params={"amplitude": 15.0, "decay_rate": 0.8},
        model_type="exponential",
    )

    print(f"Generated data: {len(np.unique(group))} subjects, {len(y)} observations")
    print(f"Time range: {X.min():.2f} to {X.max():.2f}")
    print(f"Concentration range: {y.min():.2f} to {y.max():.2f}")

    # Initial parameter estimates
    beta0 = np.array([12.0, 0.6])  # [amplitude, decay_rate]

    print("\n--- Fitting with MLE (nlmefit) ---")
    try:
        beta_mle, psi_mle, stats_mle, b_mle = nlmefit(
            X, y, group, v, exponential_decay_model, beta0, max_iter=100, verbose=1
        )

        print("MLE Results:")
        print(
            f"  Fixed effects: amplitude={beta_mle[0]:.3f}, decay_rate={beta_mle[1]:.3f}"
        )
        print("  Random effects covariance:")
        print(f"    {psi_mle}")
        if stats_mle.logl is not None:
            print(f"  Log-likelihood: {stats_mle.logl:.3f}")
        if stats_mle.aic is not None:
            print(f"  AIC: {stats_mle.aic:.3f}")
        if stats_mle.rmse is not None:
            print(f"  RMSE: {stats_mle.rmse:.3f}")

    except Exception as e:
        print(f"MLE fitting failed: {e}")
        beta_mle, psi_mle, stats_mle, b_mle = None, None, None, None

    print("\n--- Fitting with SAEM (nlmefitsa) ---")
    try:
        beta_saem, psi_saem, stats_saem, b_saem = nlmefitsa(
            X,
            y,
            group,
            v,
            exponential_decay_model,
            beta0,
            n_iterations=(50, 50, 25),  # Reduced for faster demo
            verbose=1,
        )

        print("SAEM Results:")
        print(
            f"  Fixed effects: amplitude={beta_saem[0]:.3f}, decay_rate={beta_saem[1]:.3f}"
        )
        print("  Random effects covariance:")
        print(f"    {psi_saem}")
        if stats_saem.logl is not None:
            print(f"  Log-likelihood: {stats_saem.logl:.3f}")
        if stats_saem.aic is not None:
            print(f"  AIC: {stats_saem.aic:.3f}")
        if stats_saem.rmse is not None:
            print(f"  RMSE: {stats_saem.rmse:.3f}")

    except Exception as e:
        print(f"SAEM fitting failed: {e}")
        beta_saem, psi_saem, stats_saem, b_saem = None, None, None, None

    return {
        "mle": (beta_mle, psi_mle, stats_mle, b_mle),
        "saem": (beta_saem, psi_saem, stats_saem, b_saem),
        "data": (X, y, group, v),
    }


def pharmacokinetic_example():
    """Demonstrate NLME fitting on a realistic pharmacokinetic dataset."""
    print("\n=== Pharmacokinetic Example ===")

    # Generate more realistic PK data
    X, y, group, v = generate_sample_data(
        n_subjects=12,
        n_timepoints=8,
        noise_level=0.2,
        true_params={"amplitude": 25.0, "decay_rate": 0.3},
        random_seed=123,
        model_type="exponential",
    )

    print(f"Pharmacokinetic dataset: {len(np.unique(group))} subjects")

    # Initial estimates based on visual inspection of data
    beta0 = np.array([20.0, 0.4])

    print("Fitting exponential decay model...")
    try:
        beta, psi, stats, b = nlmefit(
            X,
            y,
            group,
            v,
            exponential_decay_model,
            beta0,
            param_transform=np.array([1, 1]),  # Log-transform both parameters
            error_model="proportional",
            verbose=1,
        )

        print("\nResults:")
        print(f"  Population clearance: {beta[1]:.4f} h⁻¹")
        print(f"  Population volume: {beta[0]:.2f} mg/L")
        print("  Between-subject variability:")
        print(f"    CV% on clearance: {100 * np.sqrt(psi[1, 1]):.1f}%")
        print(f"    CV% on volume: {100 * np.sqrt(psi[0, 0]):.1f}%")

        if stats.rmse is not None:
            print(f"  Residual error: {stats.rmse:.3f}")

        return beta, psi, stats, b, (X, y, group)

    except Exception as e:
        print(f"Fitting failed: {e}")
        return None


def plot_results(results):
    """Plot fitting results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return

    if results is None:
        return

    if len(results) == 4:  # pharmacokinetic_example results
        beta, psi, stats, b, (X, y, group) = results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Individual fits
        time_pred = np.linspace(X.min(), X.max(), 100)

        # Population prediction
        y_pop = exponential_decay_model(beta, time_pred)
        ax1.plot(time_pred.ravel(), y_pop, "k-", linewidth=2, label="Population")

        # Individual data and predictions
        for i in range(len(np.unique(group))):
            mask = group == i
            ax1.plot(X[mask], y[mask], "o", alpha=0.7, markersize=4)

            if b is not None and i < b.shape[1]:
                # Individual prediction (if random effects available)
                phi_i = beta + b[:, i]
                y_ind = exponential_decay_model(phi_i, time_pred)
                ax1.plot(time_pred.ravel(), y_ind, "--", alpha=0.6)

        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Concentration (mg/L)")
        ax1.set_title("Individual and Population Fits")
        ax1.legend()
        ax1.set_yscale("log")

        # Residuals plot
        if stats.residuals is not None:
            y_pred_pop = exponential_decay_model(beta, X)
            residuals = y - y_pred_pop
            ax2.scatter(y_pred_pop, residuals, alpha=0.6)
            ax2.axhline(y=0, color="k", linestyle="--")
            ax2.set_xlabel("Population Predictions")
            ax2.set_ylabel("Residuals")
            ax2.set_title("Residuals vs Predictions")

        plt.tight_layout()
        plt.show()

    else:  # compare_algorithms results
        data = results["data"]
        X, y, group, v = data

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot data by subject
        for i in range(len(np.unique(group))):
            mask = group == i
            axes[0, 0].plot(X[mask], y[mask], "o-", alpha=0.7, label=f"Subject {i + 1}")

        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Concentration")
        axes[0, 0].set_title("Raw Data by Subject")
        axes[0, 0].legend()

        # Plot population fits if available
        time_pred = np.linspace(X.min(), X.max(), 100)

        if results["mle"][0] is not None:
            beta_mle = results["mle"][0]
            y_mle = exponential_decay_model(beta_mle, time_pred)
            axes[0, 1].plot(time_pred.ravel(), y_mle, "b-", linewidth=2, label="MLE")

        if results["saem"][0] is not None:
            beta_saem = results["saem"][0]
            y_saem = exponential_decay_model(beta_saem, time_pred)
            axes[0, 1].plot(time_pred.ravel(), y_saem, "r-", linewidth=2, label="SAEM")

        # Add data points
        axes[0, 1].scatter(X, y, alpha=0.3, color="gray")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Concentration")
        axes[0, 1].set_title("Population Fits Comparison")
        axes[0, 1].legend()

        plt.tight_layout()
        plt.show()


def main():
    """Run all examples."""
    print("PyNLME - Nonlinear Mixed-Effects Models for Python")
    print("=" * 50)

    # Example 1: Compare algorithms
    comparison_results = compare_algorithms()

    # Example 2: Pharmacokinetic modeling
    pk_results = pharmacokinetic_example()

    # Plot results if matplotlib is available
    try:
        plot_results(pk_results)
        plot_results(comparison_results)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("\n=== Examples completed ===")
    print("For more advanced usage, see the documentation at:")
    print("https://pynlme.readthedocs.io")


if __name__ == "__main__":
    main()
