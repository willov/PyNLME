#!/usr/bin/env python3
"""
Demonstration of PyNLME simplified function interface.

This example shows how to use the essential NLME functionality through different
function names:
- MATLAB-style: nlmefit, nlmefitsa
- Python-style: fit_nlme (unified interface)
"""

import numpy as np

import pynlme


def exponential_decay_model(phi, x, v=None):
    """Simple exponential decay model: y = phi[0] * exp(-phi[1] * t)"""
    return phi[0] * np.exp(-phi[1] * x[:, 0])


def generate_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)

    # Time points
    times = np.array([0, 1, 2, 3, 4, 6, 8, 12])
    n_subjects = 3

    X_list, y_list, group_list = [], [], []

    for subject_id in range(n_subjects):
        # Individual parameters (with some variability)
        true_amp = 10 + np.random.normal(0, 1)
        true_decay = 0.3 + np.random.normal(0, 0.05)

        # Generate observations
        y_true = true_amp * np.exp(-true_decay * times)
        y_obs = y_true + np.random.normal(0, 0.5, len(times))

        X_list.append(times.reshape(-1, 1))
        y_list.append(y_obs)
        group_list.append([subject_id] * len(times))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    group = np.concatenate(group_list)

    return X, y, group


def main():
    """Demonstrate different function naming styles."""
    print("PyNLME Simplified Function Interface Demo")
    print("=" * 50)

    # Generate sample data
    print("Generating sample exponential decay data...")
    X, y, group = generate_sample_data()
    beta0 = np.array([10.0, 0.3])  # Initial parameter estimates

    print(f"Data: {len(y)} observations from {len(np.unique(group))} subjects")
    print(f"Time range: {X.min():.1f} to {X.max():.1f}")
    print(f"Response range: {y.min():.2f} to {y.max():.2f}")
    print()

    # Common fitting options
    fit_options = {"max_iter": 50, "verbose": 0}

    # ========================================================================
    # MATLAB-style functions
    # ========================================================================
    print("1. MATLAB-style functions")
    print("-" * 30)

    try:
        print("Using nlmefit() [MATLAB MLE style]...")
        beta_matlab_mle, psi_matlab_mle, stats_matlab_mle, b_matlab_mle = (
            pynlme.nlmefit(
                X, y, group, None, exponential_decay_model, beta0, **fit_options
            )
        )
        print(
            f"   Fixed effects: amplitude={beta_matlab_mle[0]:.3f}, decay={beta_matlab_mle[1]:.3f}"
        )
        if stats_matlab_mle.logl is not None:
            print(f"   Log-likelihood: {stats_matlab_mle.logl:.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    try:
        print("Using nlmefitsa() [MATLAB SAEM style]...")
        beta_matlab_saem, psi_matlab_saem, stats_matlab_saem, b_matlab_saem = (
            pynlme.nlmefitsa(
                X, y, group, None, exponential_decay_model, beta0, **fit_options
            )
        )
        print(
            f"   Fixed effects: amplitude={beta_matlab_saem[0]:.3f}, decay={beta_matlab_saem[1]:.3f}"
        )
        if stats_matlab_saem.logl is not None:
            print(f"   Log-likelihood: {stats_matlab_saem.logl:.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    print()

    # ========================================================================
    # Python-style unified interface
    # ========================================================================
    print("2. Python-style unified interface")
    print("-" * 30)

    try:
        print("Using fit_nlme() with method='ML' [Python unified style]...")
        beta_python_ml, psi_python_ml, stats_python_ml, b_python_ml = pynlme.fit_nlme(
            X,
            y,
            group,
            None,
            exponential_decay_model,
            beta0,
            method="ML",
            **fit_options,
        )
        print(
            f"   Fixed effects: amplitude={beta_python_ml[0]:.3f}, decay={beta_python_ml[1]:.3f}"
        )
        if stats_python_ml.logl is not None:
            print(f"   Log-likelihood: {stats_python_ml.logl:.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    try:
        print("Using fit_nlme() with method='SAEM' [Python unified style]...")
        beta_python_saem, psi_python_saem, stats_python_saem, b_python_saem = (
            pynlme.fit_nlme(
                X,
                y,
                group,
                None,
                exponential_decay_model,
                beta0,
                method="SAEM",
                **fit_options,
            )
        )
        print(
            f"   Fixed effects: amplitude={beta_python_saem[0]:.3f}, decay={beta_python_saem[1]:.3f}"
        )
        if stats_python_saem.logl is not None:
            print(f"   Log-likelihood: {stats_python_saem.logl:.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("3. Summary")
    print("-" * 30)
    print("PyNLME provides a clean, simple interface with just 3 essential functions:")
    print()
    print("MATLAB users:")
    print("  • nlmefit()                    - Maximum Likelihood Estimation")
    print("  • nlmefitsa()                  - Stochastic Approximation EM")
    print()
    print("Python users:")
    print("  • fit_nlme(method='ML')        - Unified interface, ML method")
    print("  • fit_nlme(method='SAEM')      - Unified interface, SAEM method")
    print()
    print("This simplified interface makes PyNLME easy to learn and use!")
    print("All functions produce identical results for the same algorithm.")


if __name__ == "__main__":
    main()
