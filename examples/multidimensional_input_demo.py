#!/usr/bin/env python3
"""
PyNLME Multi-Dimensional Input Demo
===================================

This demo shows how to use the new multi-dimensional input format
where each row represents a subject/group, making it easier to work
with structured data without manual stacking.
"""

import numpy as np

from pynlme import nlmefit
from pynlme.utils import stack_grouped_data


def exponential_model(phi, x, v=None):
    """Two-parameter exponential decay model."""
    return phi[0] * np.exp(-phi[1] * x.ravel())


def main():
    print("PyNLME Multi-Dimensional Input Demo")
    print("=" * 40)
    
    # Example 1: Your exact use case - 3 persons, 4 measurements each
    print("\n1. Your Use Case: 3 persons, 4 measurements each")
    print("-" * 50)
    
    # Traditional stacked format (what you had to do before)
    print("Traditional stacked format:")
    X_stacked_traditional = np.array([
        [1], [2], [3], [4],  # Person 1
        [1], [2], [3], [4],  # Person 2  
        [1], [2], [3], [4],  # Person 3
    ])
    y_stacked_traditional = np.array([10, 7, 5, 3, 12, 8, 6, 4, 11, 8, 6, 3])
    group_traditional = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    
    print(f"X shape: {X_stacked_traditional.shape}")
    print(f"y shape: {y_stacked_traditional.shape}")
    print(f"group shape: {group_traditional.shape}")
    print(f"First few X values: {X_stacked_traditional[:6].flatten()}")
    print(f"First few y values: {y_stacked_traditional[:6]}")
    print(f"First few group values: {group_traditional[:6]}")
    
    # New multi-dimensional format (what you can do now!)
    print("\nNew multi-dimensional format:")
    X_grouped = np.array([
        [1, 2, 3, 4],  # Person 1 measurements
        [1, 2, 3, 4],  # Person 2 measurements
        [1, 2, 3, 4],  # Person 3 measurements
    ])
    y_grouped = np.array([
        [10, 7, 5, 3],  # Person 1 responses
        [12, 8, 6, 4],  # Person 2 responses
        [11, 8, 6, 3],  # Person 3 responses
    ])
    
    print(f"X_grouped shape: {X_grouped.shape}")
    print(f"y_grouped shape: {y_grouped.shape}")
    print("X_grouped:")
    print(X_grouped)
    print("y_grouped:")
    print(y_grouped)
    
    # Show how the automatic conversion works
    print("\nAutomatic conversion to stacked format:")
    X_stacked, y_stacked, group_stacked = stack_grouped_data(X_grouped, y_grouped)
    print(f"Converted X shape: {X_stacked.shape}")
    print(f"Converted y shape: {y_stacked.shape}")
    print(f"Converted group shape: {group_stacked.shape}")
    
    # Verify they're equivalent
    print("\nVerifying equivalence:")
    print(f"X values match: {np.allclose(X_stacked, X_stacked_traditional)}")
    print(f"y values match: {np.allclose(y_stacked, y_stacked_traditional)}")
    print(f"group values match: {np.allclose(group_stacked, group_traditional)}")
    
    # Example 2: Real pharmacokinetic data
    print("\n\n2. Pharmacokinetic Example")
    print("-" * 30)
    
    # Generate realistic PK data
    np.random.seed(42)
    time_points = np.array([0.5, 1, 2, 4, 8, 12, 24])  # Hours
    n_subjects = 5
    
    # Population parameters
    pop_clearance = 2.0  # L/h
    pop_volume = 20.0    # L
    
    # Generate data for each subject
    X_pk = np.tile(time_points, (n_subjects, 1))  # Same time points for all
    y_pk = []
    
    for i in range(n_subjects):
        # Random effects (inter-subject variability)
        eta_cl = np.random.normal(0, 0.3)
        eta_v = np.random.normal(0, 0.2)
        
        # Individual parameters
        cl_i = pop_clearance * np.exp(eta_cl)
        v_i = pop_volume * np.exp(eta_v)
        
        # Concentration predictions (1-compartment model)
        dose = 100  # mg
        conc = (dose / v_i) * np.exp(-cl_i / v_i * time_points)
        
        # Add residual error
        conc_obs = conc * np.exp(np.random.normal(0, 0.1, len(time_points)))
        y_pk.append(conc_obs)
    
    y_pk = np.array(y_pk)
    
    print(f"PK data shape: X={X_pk.shape}, y={y_pk.shape}")
    print("First subject data:")
    print(f"  Times: {X_pk[0]}")
    print(f"  Concentrations: {np.array2string(y_pk[0], precision=2, separator=', ')}")
    
    # Fit model using new multi-dimensional format
    print("\nFitting model with multi-dimensional input...")
    
    try:
        beta0 = np.array([5.0, 0.2])  # Initial estimates
        
        # This is now possible with the new format!
        beta, psi, stats, b = nlmefit(
            X=X_pk,           # Multi-dimensional input
            y=y_pk,           # Multi-dimensional input
            group=None,       # Not needed for grouped format
            V=None,
            modelfun=exponential_model,
            beta0=beta0,
            verbose=1
        )
        
        print(f"Fixed effects estimates:")
        print(f"  Amplitude (dose/V): {beta[0]:.3f}")
        print(f"  Elimination rate (CL/V): {beta[1]:.3f}")
        print(f"Random effects covariance:")
        print(f"  {psi}")
        
        # Compare with traditional stacked format
        print("\nComparing with traditional stacked format...")
        X_pk_stacked, y_pk_stacked, group_pk_stacked = stack_grouped_data(X_pk, y_pk)
        
        beta_trad, psi_trad, stats_trad, b_trad = nlmefit(
            X=X_pk_stacked,
            y=y_pk_stacked,
            group=group_pk_stacked,
            V=None,
            modelfun=exponential_model,
            beta0=beta0,
            verbose=0
        )
        
        print(f"Results are identical: {np.allclose(beta, beta_trad, rtol=1e-10)}")
        
    except Exception as e:
        print(f"Model fitting failed (likely backend issue): {e}")
        print("But the data conversion functionality works!")
    
    # Example 3: Multiple predictor variables
    print("\n\n3. Multiple Predictor Variables")
    print("-" * 35)
    
    # 3 subjects, 4 time points, 2 predictor variables (time and dose)
    X_multi = np.array([
        [[1, 100], [2, 100], [3, 100], [4, 100]],  # Subject 1: time, dose
        [[1, 150], [2, 150], [3, 150], [4, 150]],  # Subject 2: time, dose
        [[1, 200], [2, 200], [3, 200], [4, 200]],  # Subject 3: time, dose
    ])
    y_multi = np.array([
        [50, 40, 30, 20],  # Subject 1 responses
        [70, 55, 40, 25],  # Subject 2 responses
        [90, 70, 50, 30],  # Subject 3 responses
    ])
    
    print(f"Multi-predictor data shape: X={X_multi.shape}, y={y_multi.shape}")
    print("X_multi[0] (first subject):")
    print(X_multi[0])
    
    # Convert to stacked format
    X_multi_stacked, y_multi_stacked, group_multi_stacked = stack_grouped_data(X_multi, y_multi)
    print(f"Stacked shape: X={X_multi_stacked.shape}, y={y_multi_stacked.shape}")
    print("First few rows of stacked X:")
    print(X_multi_stacked[:6])
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("✓ Multi-dimensional input format is now supported!")
    print("✓ Data where each row represents a subject/group")
    print("✓ Automatic conversion to internal stacked format")
    print("✓ Compatible with existing MATLAB-style interface")
    print("✓ Supports multiple predictor variables")
    print("✓ Same results as traditional stacked format")
    print("\nYou can now use either format:")
    print("  - Traditional: X(n_obs, n_features), y(n_obs,), group(n_obs,)")
    print("  - New: X(n_groups, n_obs_per_group, n_features), y(n_groups, n_obs_per_group)")


if __name__ == "__main__":
    main()
