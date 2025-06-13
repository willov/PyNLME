"""
Test comparing Rust and Python backends to ensure they both work correctly.

This test verifies that both backends produce reasonable results for the same
optimization problem, helping catch regressions in either implementation.
"""

import os
import sys

import numpy as np
import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pynlme import nlmefitsa


def model_function(phi, x, v=None):
    """Simple model function for testing both backends."""
    if v is None:
        raise ValueError("Group-level covariates V must be provided")

    phi1, phi2, phi3 = phi[0], phi[1], phi[2]
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    
    # Handle V properly - extract scalar value
    if hasattr(v, "shape") and len(v.shape) > 0:
        v_scalar = v[0] if len(v.shape) == 1 else v[0, 0]
    else:
        v_scalar = v
    
    result = phi1 * x1 * np.exp(phi2 * x2 / v_scalar) + phi3 * x3
    return result


@pytest.fixture
def test_data():
    """Fixture providing test data for backend comparison."""
    # Test data (subset from MATLAB documentation)
    X = np.array([
        [8.1472, 0.7060, 75.1267],
        [9.0579, 0.0318, 25.5095],
        [1.2699, 0.2769, 50.5957],
        [9.1338, 0.0462, 69.9077],
        [6.3236, 0.0971, 89.0903],
    ])
    
    y = np.array([573.4851, 188.3748, 356.7075, 499.6050, 631.6939])
    group = np.array([0, 0, 0, 0, 0])  # All same group
    V = np.array([[2, 3]])  # Group-level covariates
    initial_params = np.array([1.0, 1.0, 1.0])
    expected_params = np.array([1.0008, 4.9980, 6.9999])
    
    return {
        'X': X,
        'y': y,
        'group': group,
        'V': V,
        'initial_params': initial_params,
        'expected_params': expected_params
    }


def test_rust_backend_available():
    """Test if Rust backend is available and working."""
    try:
        import pynlme.nlmefit  # noqa: F401
        nlmefit_module = sys.modules['pynlme.nlmefit']
        rust_available = nlmefit_module.RUST_AVAILABLE

        # This is informational - we don't fail if Rust is not available
        if rust_available:
            print("Rust backend is available")
        else:
            print("Rust backend is not available")

    except Exception as e:
        pytest.fail(f"Failed to check Rust backend availability: {e}")


def test_python_backend_works(test_data):
    """Test that Python backend produces reasonable results."""
    # Force Python backend
    import pynlme.nlmefit  # noqa: F401
    nlmefit_module = sys.modules['pynlme.nlmefit']
    rust_backup = nlmefit_module.RUST_AVAILABLE
    nlmefit_module.RUST_AVAILABLE = False

    try:
        beta, psi, stats, b = nlmefitsa(
            X=test_data['X'],
            y=test_data['y'],
            group=test_data['group'],
            V=test_data['V'],
            modelfun=model_function,
            beta0=test_data['initial_params'],
            verbose=0,  # Quiet for testing
        )

        # Check that we got reasonable results
        assert len(beta) == 3, "Should return 3 parameters"
        assert np.isfinite(beta).all(), "Parameters should be finite"

        # Check that results are in reasonable range (loose bounds)
        assert 0.1 < beta[0] < 10.0, f"First parameter {beta[0]} out of reasonable range"
        assert 1.0 < beta[1] < 10.0, f"Second parameter {beta[1]} out of reasonable range"
        assert 1.0 < beta[2] < 15.0, f"Third parameter {beta[2]} out of reasonable range"

        print(f"Python backend result: {beta}")

    finally:
        # Restore Rust backend availability
        nlmefit_module.RUST_AVAILABLE = rust_backup


def test_rust_backend_works_if_available(test_data):
    """Test that Rust backend produces reasonable results if available."""
    import pynlme.nlmefit  # noqa: F401
    nlmefit_module = sys.modules['pynlme.nlmefit']

    if not nlmefit_module.RUST_AVAILABLE:
        pytest.skip("Rust backend not available")

    beta, psi, stats, b = nlmefitsa(
        X=test_data['X'],
        y=test_data['y'],
        group=test_data['group'],
        V=test_data['V'],
        modelfun=model_function,
        beta0=test_data['initial_params'],
        verbose=0,  # Quiet for testing
    )

    # Check that we got reasonable results
    assert len(beta) == 3, "Should return 3 parameters"
    assert np.isfinite(beta).all(), "Parameters should be finite"

    # Check that results are in reasonable range (loose bounds)
    assert 0.1 < beta[0] < 10.0, f"First parameter {beta[0]} out of reasonable range"
    assert 1.0 < beta[1] < 10.0, f"Second parameter {beta[1]} out of reasonable range"
    assert 1.0 < beta[2] < 15.0, f"Third parameter {beta[2]} out of reasonable range"

    print(f"Rust backend result: {beta}")


def test_backend_comparison(test_data):
    """Compare results between Rust and Python backends if both are available."""
    import pynlme.nlmefit  # noqa: F401
    nlmefit_module = sys.modules['pynlme.nlmefit']

    if not nlmefit_module.RUST_AVAILABLE:
        pytest.skip("Rust backend not available for comparison")

    # Test Rust backend first
    beta_rust, _, _, _ = nlmefitsa(
        X=test_data['X'],
        y=test_data['y'],
        group=test_data['group'],
        V=test_data['V'],
        modelfun=model_function,
        beta0=test_data['initial_params'],
        verbose=0,
    )

    # Test Python backend
    rust_backup = nlmefit_module.RUST_AVAILABLE
    nlmefit_module.RUST_AVAILABLE = False

    try:
        beta_python, _, _, _ = nlmefitsa(
            X=test_data['X'],
            y=test_data['y'],
            group=test_data['group'],
            V=test_data['V'],
            modelfun=model_function,
            beta0=test_data['initial_params'],
            verbose=0,
        )
    finally:
        nlmefit_module.RUST_AVAILABLE = rust_backup

    # Compare results
    diff = np.abs(np.array(beta_rust) - np.array(beta_python))
    max_diff = diff.max()

    print(f"Rust result:   {beta_rust}")
    print(f"Python result: {beta_python}")
    print(f"Max difference: {max_diff:.6f}")

    # Allow for reasonable numerical differences between implementations
    # This is quite generous to account for different optimization paths
    assert max_diff < 1.0, (
        f"Backends differ too much (max diff: {max_diff:.6f}). "
        f"Rust: {beta_rust}, Python: {beta_python}"
    )

    if max_diff < 0.001:
        print("✓ Results match very closely!")
    elif max_diff < 0.1:
        print("✓ Results match acceptably")
    else:
        print("⚠ Results differ but within tolerance")


def test_both_backends_close_to_expected(test_data):
    """Test that both backends produce results reasonably close to expected values."""
    expected = test_data['expected_params']
    tolerance = 2.0  # Generous tolerance for optimization problems

    import pynlme.nlmefit  # noqa: F401
    nlmefit_module = sys.modules['pynlme.nlmefit']

    # Test Python backend
    rust_backup = nlmefit_module.RUST_AVAILABLE
    nlmefit_module.RUST_AVAILABLE = False

    try:
        beta_python, _, _, _ = nlmefitsa(
            X=test_data['X'],
            y=test_data['y'],
            group=test_data['group'],
            V=test_data['V'],
            modelfun=model_function,
            beta0=test_data['initial_params'],
            verbose=0,
        )

        python_diff = np.abs(np.array(beta_python) - expected)
        assert python_diff.max() < tolerance, (
            f"Python backend too far from expected. "
            f"Got: {beta_python}, Expected: {expected}, Max diff: {python_diff.max()}"
        )

    finally:
        nlmefit_module.RUST_AVAILABLE = rust_backup

    # Test Rust backend if available
    if nlmefit_module.RUST_AVAILABLE:
        beta_rust, _, _, _ = nlmefitsa(
            X=test_data['X'],
            y=test_data['y'],
            group=test_data['group'],
            V=test_data['V'],
            modelfun=model_function,
            beta0=test_data['initial_params'],
            verbose=0,
        )

        rust_diff = np.abs(np.array(beta_rust) - expected)
        assert rust_diff.max() < tolerance, (
            f"Rust backend too far from expected. "
            f"Got: {beta_rust}, Expected: {expected}, Max diff: {rust_diff.max()}"
        )
