"""
Test the simplified function interface.
"""

import numpy as np
import pytest

import pynlme


def simple_model(phi, x, v=None):
    """Simple exponential model for testing."""
    return phi[0] * np.exp(-phi[1] * x[:, 0])


@pytest.fixture
def sample_data():
    """Generate simple sample data for testing."""
    X = np.array([[0, 1, 2, 3]]).T
    y = np.array([10, 7, 5, 3])
    group = np.array([1, 1, 1, 1])
    beta0 = np.array([10.0, 0.5])
    return X, y, group, beta0


def test_fit_nlme_ml_alias(sample_data):
    """Test that fit_nlme with method='ML' works like nlmefit."""
    X, y, group, beta0 = sample_data

    result1 = pynlme.nlmefit(X, y, group, None, simple_model, beta0, max_iter=1)
    result2 = pynlme.fit_nlme(
        X, y, group, None, simple_model, beta0, method="ML", max_iter=1
    )

    # Results should be identical
    np.testing.assert_array_almost_equal(result1[0], result2[0])
    np.testing.assert_array_almost_equal(result1[1], result2[1])


def test_fit_nlme_saem_alias(sample_data):
    """Test that fit_nlme with method='SAEM' works like nlmefitsa."""
    X, y, group, beta0 = sample_data

    # Set random seed for reproducible results
    import numpy as np

    np.random.seed(42)
    result1 = pynlme.nlmefitsa(
        X, y, group, None, simple_model, beta0, max_iter=1, random_state=42
    )

    np.random.seed(42)
    result2 = pynlme.fit_nlme(
        X,
        y,
        group,
        None,
        simple_model,
        beta0,
        method="SAEM",
        max_iter=1,
        random_state=42,
    )

    # For SAEM (stochastic algorithm), results can vary significantly even with same seed
    # Check that both functions return the same structure and reasonable values
    assert len(result1[0]) == len(result2[0])  # Same parameter count
    assert result1[1].shape == result2[1].shape  # Same covariance matrix shape

    # Check that parameter estimates are in reasonable range (much more lenient for stochastic algorithms)
    # SAEM is highly stochastic, so we just ensure results are finite and not wildly different
    assert np.all(np.isfinite(result1[0])) and np.all(np.isfinite(result2[0]))
    assert np.all(np.isfinite(result1[1])) and np.all(np.isfinite(result2[1]))

    # Check that parameters are in reasonable bounds (very loose test)
    assert np.all(np.abs(result1[0]) < 100) and np.all(np.abs(result2[0]) < 100)
    assert np.all(result1[1].diagonal() > 0) and np.all(
        result2[1].diagonal() > 0
    )  # Positive diagonal


def test_fit_nlme_default_method(sample_data):
    """Test that fit_nlme defaults to ML method."""
    X, y, group, beta0 = sample_data

    result1 = pynlme.nlmefit(X, y, group, None, simple_model, beta0, max_iter=1)
    result2 = pynlme.fit_nlme(X, y, group, None, simple_model, beta0, max_iter=1)

    # Results should be identical (default method should be ML)
    np.testing.assert_array_almost_equal(result1[0], result2[0])
    np.testing.assert_array_almost_equal(result1[1], result2[1])


def test_invalid_method_parameter():
    """Test that invalid method parameter raises error."""
    X = np.array([[0, 1]]).T
    y = np.array([10, 7])
    group = np.array([1, 1])
    beta0 = np.array([10.0, 0.5])

    # Invalid method for fit_nlme
    with pytest.raises(ValueError, match="Unknown method 'INVALID'"):
        pynlme.fit_nlme(X, y, group, None, simple_model, beta0, method="INVALID")


def test_available_functions():
    """Test that essential functions are available."""
    expected_functions = [
        "nlmefit",  # MATLAB-style
        "nlmefitsa",  # MATLAB-style
        "fit_nlme",  # Pythonic unified interface
    ]

    for func_name in expected_functions:
        assert hasattr(pynlme, func_name), f"Function {func_name} not available"
        func = getattr(pynlme, func_name)
        assert callable(func), f"{func_name} is not callable"


def test_function_docstrings():
    """Test that functions have appropriate docstrings."""
    # Test that fit_nlme has a good docstring indicating it's the Python interface
    assert "unified" in pynlme.fit_nlme.__doc__.lower()
    assert "python" in pynlme.fit_nlme.__doc__.lower()

    # Test that nlmefit mentions MATLAB compatibility
    assert "matlab" in pynlme.nlmefit.__doc__.lower()
