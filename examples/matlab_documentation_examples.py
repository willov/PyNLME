"""
Recreation of MATLAB nlmefit documentation examples.

This file recreates the exact examples from the MATLAB Statistics and Machine Learning
Toolbox documentation for nlmefit:
https://se.mathworks.com/help/stats/nlmefit.html

The examples demonstrate:
1. Specify Group (using nlmefitsa with group-level predictors)
2. Transform and Plot Fitted Model (using indomethacin pharmacokinetic data)
"""

import numpy as np

from pynlme import nlmefit, nlmefitsa


def example_1_specify_group():
    """
    Example 1: Specify Group

    This example demonstrates nlmefitsa with group-level predictors using the stochastic EM algorithm.
    MATLAB call: beta = nlmefitsa(X,y,group,V,model,[1 1 1])

    Expected parameter estimates: [1.0008, 4.9980, 6.9999]
    """
    print("Example 1: Specify Group")
    print("-" * 25)

    # MATLAB data
    # MATLAB data for group predictors example
    # Direct definition of X, y, V, group instead of extracting from matrix

    # X predictors (3 columns: X1, X2, X3)
    X = np.array(
        [
            [8.1472, 0.7060, 75.1267],
            [9.0579, 0.0318, 25.5095],
            [1.2699, 0.2769, 50.5957],
            [9.1338, 0.0462, 69.9077],
            [6.3236, 0.0971, 89.0903],
            [0.9754, 0.8235, 95.9291],
            [2.7850, 0.6948, 54.7216],
            [5.4688, 0.3171, 13.8624],
            [9.5751, 0.9502, 14.9294],
            [9.6489, 0.0344, 25.7508],
            [1.5761, 0.4387, 84.0717],
            [9.7059, 0.3816, 25.4282],
            [9.5717, 0.7655, 81.4285],
            [4.8538, 0.7952, 24.3525],
            [8.0028, 0.1869, 92.9264],
        ]
    )

    # Response variable y
    y = np.array(
        [
            573.4851,
            188.3748,
            356.7075,
            499.6050,
            631.6939,
            679.1466,
            398.8715,
            109.1202,
            207.5047,
            190.7724,
            593.2222,
            203.1922,
            634.8833,
            205.9043,
            663.2529,
        ]
    )

    # Group identifiers
    group = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Group-level covariates V
    V = np.array([2, 3])

    print(f"Data: {len(y)} observations, {len(np.unique(group))} group(s)")
    print("Model: y = φ₁ * X₁ * exp(φ₂ * X₂ / V) + φ₃ * X₃")

    # Test our nlmefitsa implementation
    try:
        initial_params = np.array([1.0, 1.0, 1.0])

        beta, psi, stats, b  = nlmefitsa(
            X=X,
            y=y,
            group=group,
            V=V,
            modelfun=model_function_group_predictors,
            beta0=initial_params,
        )

        result = beta
        print(f"✓ Estimated parameters: {result}")
        print("  Expected parameters:  [1.0008, 4.9980, 6.9999]")

        # Check accuracy
        expected = np.array([1.0008, 4.9980, 6.9999])
        if hasattr(result, "__len__") and len(result) >= 3:
            diff = np.abs(np.array(result[:3]) - expected)
            if np.all(diff < 0.1):
                print("✓ Results match MATLAB expected values!")
            else:
                print(f"⚠ Results differ from expected (max diff: {diff.max():.4f})")

    except Exception as e:
        print(f"✗ Error: {e}")

    return X, y, group, V


def model_function_group_predictors(phi, x, v=None):
    """
    Model function for group predictors example.

    MATLAB equivalent:
    model = @(PHI,XFUN,VFUN)(PHI(1).*XFUN(:,1).*exp(PHI(2).*XFUN(:,2)./VFUN)+PHI(3).*XFUN(:,3))

    Parameters
    ----------
    phi : array_like
        Parameters [phi1, phi2, phi3]
    x : array_like
        Predictor matrix with columns [x1, x2, x3]
    v : array_like, optional
        Group-level covariates (VFUN in MATLAB)

    Returns
    -------
    y : ndarray
        Model predictions
    """
    if v is None:
        raise ValueError("Group-level covariates V must be provided")

    # MATLAB: PHI(1).*XFUN(:,1).*exp(PHI(2).*XFUN(:,2)./VFUN)+PHI(3).*XFUN(:,3)
    # Note: In Python, indexing is 0-based while MATLAB is 1-based

    phi1, phi2, phi3 = phi[0], phi[1], phi[2]
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]  # Now we have all 3 columns

    # Apply the MATLAB model formula
    # PHI(1) * XFUN(:,1) * exp(PHI(2) * XFUN(:,2) / VFUN) + PHI(3) * XFUN(:,3)
    # For single group, use first element of V
    v_scalar = v[0] if hasattr(v, '__len__') else v
    result = phi1 * x1 * np.exp(phi2 * x2 / v_scalar) + phi3 * x3

    return result


def example_2_transform_and_plot():
    """
    Example 2: Transform and Plot Fitted Model

    This example demonstrates parameter transformations and result visualization
    using the indomethacin pharmacokinetic data.

    Expected parameter estimates: [0.4606, -1.3459, 2.8277, 0.7729]
    These correspond to: [ka, log(V), log(Cl), log(σ²)]
    """
    print("Example 2: Transform and Plot Fitted Model")
    print("-" * 40)

    # Indomethacin pharmacokinetic data - 6 subjects, 11 time points each
    # Data organized by subject (one row per subject)
    concentration_by_subject = np.array(
        [
            [
                1.5000,
                0.9400,
                0.7800,
                0.4800,
                0.3700,
                0.1900,
                0.1200,
                0.1100,
                0.0800,
                0.0700,
                0.0500,
            ],  # Subject 1
            [
                2.0300,
                1.6300,
                0.7100,
                0.7000,
                0.6400,
                0.3600,
                0.3200,
                0.2000,
                0.2500,
                0.1200,
                0.0800,
            ],  # Subject 2
            [
                2.7200,
                1.4900,
                1.1600,
                0.8000,
                0.8000,
                0.3900,
                0.2200,
                0.1200,
                0.1100,
                0.0800,
                0.0800,
            ],  # Subject 3
            [
                1.8500,
                1.3900,
                1.0200,
                0.8900,
                0.5900,
                0.4000,
                0.1600,
                0.1100,
                0.1000,
                0.0700,
                0.0700,
            ],  # Subject 4
            [
                2.0500,
                1.0400,
                0.8100,
                0.3900,
                0.3000,
                0.2300,
                0.1300,
                0.1100,
                0.0800,
                0.1000,
                0.0600,
            ],  # Subject 5
            [
                2.3100,
                1.4400,
                1.0300,
                0.8400,
                0.6400,
                0.4200,
                0.2400,
                0.1700,
                0.1300,
                0.1000,
                0.0900,
            ],  # Subject 6
        ]
    )

    # Time points (same for all subjects)
    time_points = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00]
    )

    # Flatten data for nlmefit (which expects 1D arrays)
    concentration = concentration_by_subject.flatten()
    time = np.tile(time_points, 6)  # Repeat time points for each subject
    subject = np.repeat(np.arange(1, 7), 11)  # Subject IDs: 1,1,1,...,2,2,2,...,6,6,6

    print(
        f"Data: {len(concentration)} observations, {len(np.unique(subject))} subjects"
    )
    print("Model: One-compartment with first-order absorption")

    # Fit the model
    try:
        initial_params = np.array(
            [0.5, -1.0, 2.5, 0.5]
        )  # [ka, log(V), log(Cl), log(σ²)]

        beta, psi, stats, b = nlmefit(
            X=time.reshape(-1, 1),  # Time as X predictor
            y=concentration,
            group=subject,
            V=None,  # No group-level predictors for this example
            modelfun=indomethacin_model,
            beta0=initial_params,
        )

        result = beta
        print(f"✓ Estimated parameters: {result}")
        print("  Expected parameters:  [0.4606, -1.3459, 2.8277, 0.7729]")

        # Check accuracy
        expected = np.array([0.4606, -1.3459, 2.8277, 0.7729])
        if hasattr(result, "__len__") and len(result) >= 4:
            diff = np.abs(np.array(result[:4]) - expected)
            if np.all(diff < 0.1):
                print("✓ Results match MATLAB expected values!")
            else:
                print(f"⚠ Results differ from expected (max diff: {diff.max():.4f})")

        # Parameter transformations
        if hasattr(result, "__len__") and len(result) >= 3:
            print("\nParameter transformations:")
            print(f"  ka (absorption rate): {result[0]:.4f}")
            print(f"  V (volume):          {np.exp(result[1]):.4f}")
            print(f"  Cl (clearance):      {np.exp(result[2]):.4f}")
            if len(result) > 3:
                print(f"  σ² (residual var):   {np.exp(result[3]):.4f}")

    except Exception as e:
        print(f"✗ Error: {e}")

    return concentration, time, subject


def indomethacin_model(phi, t, dose=None):
    """
    One-compartment model with first-order absorption for indomethacin.

    C(t) = (F*D*ka)/(V*(ka-k)) * (exp(-k*t) - exp(-ka*t))

    Where:
    - ka = absorption rate constant (phi[0])
    - V = volume of distribution (exp(phi[1]))
    - Cl = clearance (exp(phi[2]))
    - k = elimination rate constant = Cl/V
    - F = bioavailability (assumed = 1)
    - D = dose

    Parameters
    ----------
    phi : array_like
        Parameters [ka, log(V), log(Cl), ...]
    t : array_like
        Time points
    dose : float, optional
        Dose amount

    Returns
    -------
    concentration : ndarray
        Predicted concentration
    """
    if dose is None:
        dose = 1.0  # Default dose

    ka = phi[0]  # Absorption rate constant
    V = np.exp(phi[1])  # Volume of distribution
    Cl = np.exp(phi[2])  # Clearance
    k = Cl / V  # Elimination rate constant

    # Avoid division by zero
    if abs(ka - k) < 1e-10:
        ka = k + 1e-10

    # One-compartment model with first-order absorption
    concentration = (dose * ka) / (V * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))

    return concentration


def run_all_examples():
    """
    Run all MATLAB documentation examples.
    """
    print("MATLAB nlmefit Documentation Examples")
    print("=====================================")
    print()

    # Run each example
    example_1_specify_group()
    print()

    example_2_transform_and_plot()


if __name__ == "__main__":
    run_all_examples()
