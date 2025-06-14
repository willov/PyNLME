#!/usr/bin/env python3
"""
PyNLME API Comparison Demo
==========================

This script demonstrates the different API styles available in PyNLME:
1. Python-style unified interface (recommended)
2. Python-style direct algorithm interfaces
3. MATLAB-compatible interfaces (for migration)
"""

import numpy as np

import pynlme


def exponential_model(phi, x, v=None):
    """Simple exponential decay model."""
    return phi[0] * np.exp(-phi[1] * x.ravel())


def main():
    """Demonstrate all API styles."""
    print("PyNLME API Comparison Demo")
    print("=" * 50)
    
    # Generate simple test data
    np.random.seed(42)
    X = np.array([[0.5, 1, 2, 3, 4, 5]]).T
    y = np.array([9.8, 7.2, 4.1, 2.8, 1.9, 1.3])
    group = np.array([0, 0, 0, 0, 0, 0])
    beta0 = np.array([10.0, 0.5])
    
    print(f"Data: {len(y)} observations")
    print(f"Initial parameters: {beta0}")
    print()
    
    # =========================================================================
    # 1. Python-style unified interface (RECOMMENDED)
    # =========================================================================
    print("1. Python-Style Unified Interface (RECOMMENDED)")
    print("-" * 50)
    
    # Default MLE
    print("Using fit_nlme() with default MLE method:")
    beta1, psi1, stats1, b1 = pynlme.fit_nlme(
        X, y, group, None, exponential_model, beta0
    )
    print(f"  β = {beta1}")
    print(f"  Log-likelihood = {stats1.logl:.3f}")
    
    # Explicit MLE
    print("\nUsing fit_nlme() with explicit MLE method:")
    beta1b, psi1b, stats1b, b1b = pynlme.fit_nlme(
        X, y, group, None, exponential_model, beta0, method='MLE'
    )
    print(f"  β = {beta1b}")
    print(f"  Same result: {np.allclose(beta1, beta1b)}")
    
    # SAEM method
    print("\nUsing fit_nlme() with SAEM method:")
    beta1c, psi1c, stats1c, b1c = pynlme.fit_nlme(
        X, y, group, None, exponential_model, beta0, method='SAEM'
    )
    print(f"  β = {beta1c}")
    print(f"  Log-likelihood = {stats1c.logl:.3f}")
    print()
    
    # =========================================================================
    # 2. Python-style direct interfaces
    # =========================================================================
    print("2. Python-Style Direct Algorithm Interfaces")
    print("-" * 45)
    
    # Direct MLE
    print("Using fit_mle() directly:")
    beta2a, psi2a, stats2a, b2a = pynlme.fit_mle(
        X, y, group, None, exponential_model, beta0
    )
    print(f"  β = {beta2a}")
    print(f"  Same as fit_nlme: {np.allclose(beta1, beta2a)}")
    
    # Direct SAEM
    print("\nUsing fit_saem() directly:")
    beta2b, psi2b, stats2b, b2b = pynlme.fit_saem(
        X, y, group, None, exponential_model, beta0
    )
    print(f"  β = {beta2b}")
    print(f"  Same as fit_nlme SAEM: {np.allclose(beta1c, beta2b)}")
    print()
    
    # =========================================================================
    # 3. MATLAB-compatible interfaces
    # =========================================================================
    print("3. MATLAB-Compatible Interfaces (for migration)")
    print("-" * 50)
    
    # MATLAB MLE style
    print("Using nlmefit() (MATLAB MLE style):")
    beta3a, psi3a, stats3a, b3a = pynlme.nlmefit(
        X, y, group, None, exponential_model, beta0
    )
    print(f"  β = {beta3a}")
    print(f"  Same as Python MLE: {np.allclose(beta1, beta3a)}")
    
    # MATLAB SAEM style
    print("\nUsing nlmefitsa() (MATLAB SAEM style):")
    beta3b, psi3b, stats3b, b3b = pynlme.nlmefitsa(
        X, y, group, None, exponential_model, beta0
    )
    print(f"  β = {beta3b}")
    print(f"  Same as Python SAEM: {np.allclose(beta1c, beta3b)}")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("Summary & Recommendations")
    print("-" * 30)
    print("✅ For NEW Python projects: Use fit_nlme() with method parameter")
    print("✅ For algorithm-specific needs: Use fit_mle() or fit_saem() directly")
    print("✅ For MATLAB migration: Use nlmefit() and nlmefitsa()")
    print("\nAll interfaces produce identical results for the same algorithm!")
    print("Note: SAEM is stochastic - results vary between runs (expected behavior)")
     # Verify consistency
    all_mle_same = (
        np.allclose(beta1, beta1b) and
        np.allclose(beta1, beta2a) and
        np.allclose(beta1, beta3a)
    )
    
    # Note: SAEM is stochastic, so different runs may produce different results
    # This is expected behavior for Monte Carlo algorithms
    print(f"\n✓ All MLE interfaces consistent: {all_mle_same}")
    print("ℹ SAEM interfaces may vary between runs (stochastic algorithm)")
    print("  This is normal behavior for Monte Carlo-based methods")


if __name__ == "__main__":
    main()
