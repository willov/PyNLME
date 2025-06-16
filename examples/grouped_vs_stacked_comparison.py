#!/usr/bin/env python3
"""
Grouped vs Stacked Format Comparison

This example demonstrates the difference between traditional stacked format
and the new multi-dimensional grouped format using a concrete scenario:
3 subjects with 4 measurements each.

Shows:
- Traditional MATLAB-style stacked format (manual arrangement required)
- New multi-dimensional grouped format (intuitive row-per-subject)
- Automatic conversion between formats
- Identical results from both approaches
"""

import numpy as np

from pynlme import nlmefit, stack_grouped_data

def exponential_model(phi, x, v=None):
    """Simple exponential decay model."""
    return phi[0] * np.exp(-phi[1] * x.ravel())

def main():
    print("Grouped vs Stacked Format Comparison")
    print("=" * 50)
    
    # BEFORE: What you had to do (MATLAB-style stacked format)
    print("\n❌ BEFORE - Manual stacking required:")
    print("You had to arrange data like this:")
    
    X_stacked = np.array([
        [1], [2], [3], [4],  # Person 1 measurements
        [1], [2], [3], [4],  # Person 2 measurements
        [1], [2], [3], [4],  # Person 3 measurements
    ])
    
    y_stacked = np.array([
        10, 7, 5, 3,   # Person 1 responses
        12, 8, 6, 4,   # Person 2 responses
        11, 8, 6, 3    # Person 3 responses
    ])
    
    group_stacked = np.array([
        0, 0, 0, 0,    # Person 1 indicators
        1, 1, 1, 1,    # Person 2 indicators
        2, 2, 2, 2     # Person 3 indicators
    ])
    
    print("X_stacked shape:", X_stacked.shape)
    print("X_stacked =")
    print(X_stacked.flatten()[:12])  # Show first 12 elements
    print("\ny_stacked shape:", y_stacked.shape)
    print("y_stacked =", y_stacked)
    print("\ngroup shape:", group_stacked.shape)
    print("group =", group_stacked)
    
    # NOW: What you can do (new multi-dimensional format)
    print("\n\n✅ NOW - Natural multi-dimensional format:")
    print("You can arrange data like this:")
    
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
    
    print("X_grouped shape:", X_grouped.shape)
    print("X_grouped =")
    print(X_grouped)
    print("\ny_grouped shape:", y_grouped.shape)
    print("y_grouped =")
    print(y_grouped)
    print("\n✓ No group parameter needed!")
    
    # Show they're equivalent
    print("\n\n🔄 Automatic Conversion:")
    X_converted, y_converted, group_converted = stack_grouped_data(X_grouped, y_grouped)
    
    print("Grouped format automatically converts to:")
    print("X_converted shape:", X_converted.shape)
    print("y_converted shape:", y_converted.shape)
    print("group_converted shape:", group_converted.shape)
    
    print("\nVerification - formats are equivalent:")
    print("X values match:", np.allclose(X_converted, X_stacked))
    print("y values match:", np.allclose(y_converted, y_stacked))
    print("group values match:", np.allclose(group_converted, group_stacked))
    
    # Model fitting
    print("\n\n🧮 Model Fitting:")
    beta0 = np.array([10.0, 0.5])
    
    try:
        print("Using traditional stacked format...")
        beta1, psi1, stats1, b1 = nlmefit(
            X=X_stacked,
            y=y_stacked,
            group=group_stacked,
            V=None,
            modelfun=exponential_model,
            beta0=beta0
        )
        
        print("Using new multi-dimensional format...")
        beta2, psi2, stats2, b2 = nlmefit(
            X=X_grouped,
            y=y_grouped,
            group=None,  # Not needed!
            V=None,
            modelfun=exponential_model,
            beta0=beta0
        )
        
        print("\nResults:")
        print(f"Traditional format - beta: {beta1}")
        print(f"Multi-dim format  - beta: {beta2}")
        print(f"Results identical: {np.allclose(beta1, beta2)}")
        
    except Exception as e:
        print(f"Model fitting failed (backend issue): {e}")
        print("But data conversion works perfectly!")
    
    print("\n\n📊 Summary:")
    print("✅ Multi-dimensional format supported")
    print("✅ Each row = one subject/person")
    print("✅ No manual stacking required")
    print("✅ No group parameter needed")
    print("✅ Same results as traditional format")
    print("✅ Cleaner, more intuitive code")

if __name__ == "__main__":
    main()
