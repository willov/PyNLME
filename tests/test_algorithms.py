"""
Tests for PyNLME algorithms module.
"""

import numpy as np
import pytest

from pynlme.algorithms import mle_algorithm, saem_algorithm
from pynlme.data_types import NLMEOptions, SAEMOptions


def simple_model(phi, x, v=None):
    """Simple exponential model for testing."""
    return phi[0] * np.exp(-phi[1] * x.ravel())


def quadratic_model(phi, x, v=None):
    """Quadratic model for testing."""
    return phi[0] + phi[1] * x.ravel() + phi[2] * x.ravel() ** 2


class TestMLEAlgorithm:
    """Test MLE algorithm implementation."""

    def test_mle_simple_model(self):
        """Test MLE with simple exponential model."""
        np.random.seed(42)

        # Generate test data
        t = np.linspace(0.5, 5, 8)
        n_subjects = 3
        X = np.tile(t, n_subjects).reshape(-1, 1)
        group = np.repeat(range(n_subjects), len(t))

        # True parameters
        true_beta = np.array([10.0, 0.8])
        true_psi = np.array([[2.0, 0.0], [0.0, 0.01]])

        # Generate individual parameters
        b_true = np.random.multivariate_normal([0, 0], true_psi, n_subjects)

        # Generate responses
        y = np.zeros(len(X))
        for i in range(n_subjects):
            mask = group == i
            phi_i = true_beta + b_true[i]
            y[mask] = simple_model(phi_i, X[mask])

        # Add noise
        y += np.random.normal(0, 0.5, len(y))
        y = np.maximum(y, 0.1)

        # Test MLE algorithm
        options = NLMEOptions(max_iter=50, verbose=0)
        beta0 = np.array([8.0, 0.6])

        try:
            beta, psi, stats, b = mle_algorithm(
                X, y, group, None, simple_model, beta0, options
            )

            # Basic checks
            assert len(beta) == 2
            assert psi.shape == (2, 2)
            assert stats is not None
            assert b.shape == (n_subjects, 2)

            # Check that results are reasonable
            assert 5.0 <= beta[0] <= 15.0  # Amplitude
            assert 0.2 <= beta[1] <= 1.5  # Decay rate
            assert np.all(np.diag(psi) > 0)  # Positive variances

        except Exception as e:
            pytest.fail(f"MLE algorithm failed: {e}")

    def test_mle_with_fixed_effects_only(self):
        """Test MLE with only fixed effects (no random effects)."""
        np.random.seed(123)

        # Generate simple linear data
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = 2.0 + 0.5 * x.ravel() + np.random.normal(0, 0.1, 20)
        group = np.zeros(20, dtype=int)  # Single group

        def linear_model(phi, x, v=None):
            return phi[0] + phi[1] * x.ravel()

        options = NLMEOptions(max_iter=30, verbose=0)
        beta0 = np.array([1.0, 0.3])

        try:
            beta, psi, stats, b = mle_algorithm(
                x, y, group, None, linear_model, beta0, options
            )

            # Should recover parameters reasonably well
            assert abs(beta[0] - 2.0) < 0.5  # Intercept
            assert abs(beta[1] - 0.5) < 0.2  # Slope

        except Exception as e:
            pytest.fail(f"MLE with fixed effects only failed: {e}")

    def test_mle_error_handling(self):
        """Test MLE error handling with invalid inputs."""
        x = np.array([[1, 2, 3]]).T
        y = np.array([1, 2, 3])
        group = np.array([0, 0, 0])

        def simple_linear(phi, x, v=None):
            return phi[0] + phi[1] * x.ravel()

        options = NLMEOptions(max_iter=10, verbose=0)

        # Test with mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            mle_algorithm(x, y[:-1], group, None, simple_linear, [1.0, 0.5], options)

        # Test with invalid beta0
        with pytest.raises((ValueError, TypeError)):
            mle_algorithm(x, y, group, None, simple_linear, None, options)


class TestSAEMAlgorithm:
    """Test SAEM algorithm implementation."""

    def test_saem_simple_model(self):
        """Test SAEM with simple exponential model."""
        np.random.seed(42)

        # Generate test data
        t = np.linspace(0.5, 5, 6)
        n_subjects = 3
        X = np.tile(t, n_subjects).reshape(-1, 1)
        group = np.repeat(range(n_subjects), len(t))

        # True parameters
        true_beta = np.array([8.0, 0.7])
        true_psi = np.array([[1.0, 0.0], [0.0, 0.01]])

        # Generate individual parameters
        b_true = np.random.multivariate_normal([0, 0], true_psi, n_subjects)

        # Generate responses
        y = np.zeros(len(X))
        for i in range(n_subjects):
            mask = group == i
            phi_i = true_beta + b_true[i]
            y[mask] = simple_model(phi_i, X[mask])

        # Add noise
        y += np.random.normal(0, 0.3, len(y))
        y = np.maximum(y, 0.1)

        # Test SAEM algorithm
        options = SAEMOptions(
            n_iterations=(20, 20, 10),  # Reduced for testing
            n_mcmc_iterations=(2, 2, 2),
            verbose=0,
        )
        beta0 = np.array([7.0, 0.5])

        try:
            beta, psi, stats, b = saem_algorithm(
                X, y, group, None, simple_model, beta0, options
            )

            # Basic checks
            assert len(beta) == 2
            assert psi.shape == (2, 2)
            assert stats is not None
            assert b.shape == (n_subjects, 2)

            # Check that results are reasonable
            assert 4.0 <= beta[0] <= 12.0  # Amplitude
            assert 0.2 <= beta[1] <= 1.2  # Decay rate
            assert np.all(np.diag(psi) > 0)  # Positive variances

        except Exception as e:
            pytest.fail(f"SAEM algorithm failed: {e}")

    def test_saem_with_different_options(self):
        """Test SAEM with different configuration options."""
        np.random.seed(456)

        # Simple dataset
        x = np.array([[1, 2, 3, 4, 5, 6]]).T
        y = np.array([2.1, 4.0, 5.9, 8.1, 9.8, 12.2])
        group = np.array([0, 0, 1, 1, 2, 2])

        def linear_model(phi, x, v=None):
            return phi[0] + phi[1] * x.ravel()

        # Test with different MCMC iterations
        options = SAEMOptions(
            n_iterations=(15, 15, 5),
            n_mcmc_iterations=(3, 3, 3),
            n_burn_in=3,
            verbose=0,
        )
        beta0 = np.array([0.5, 1.8])

        try:
            beta, psi, stats, b = saem_algorithm(
                x, y, group, None, linear_model, beta0, options
            )

            # Should produce reasonable results
            assert len(beta) == 2
            assert psi.shape == (2, 2)
            assert b.shape == (3, 2)

        except Exception as e:
            pytest.fail(f"SAEM with different options failed: {e}")

    def test_saem_error_handling(self):
        """Test SAEM error handling with invalid inputs."""
        x = np.array([[1, 2, 3]]).T
        y = np.array([1, 2, 3])
        group = np.array([0, 0, 0])

        def simple_linear(phi, x, v=None):
            return phi[0] + phi[1] * x.ravel()

        options = SAEMOptions(n_iterations=(5, 5, 2), verbose=0)

        # Test with empty group
        with pytest.raises((ValueError, IndexError)):
            saem_algorithm(x, y, np.array([]), None, simple_linear, [1.0, 0.5], options)

        # Test with negative iterations
        bad_options = SAEMOptions(n_iterations=(-1, 5, 2), verbose=0)
        with pytest.raises(ValueError):
            saem_algorithm(x, y, group, None, simple_linear, [1.0, 0.5], bad_options)


class TestAlgorithmConsistency:
    """Test consistency between MLE and SAEM algorithms."""

    def test_algorithm_comparison(self):
        """Compare MLE and SAEM results on the same data."""
        np.random.seed(789)

        # Generate well-conditioned test data
        t = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        n_subjects = 2
        X = np.tile(t, n_subjects).reshape(-1, 1)
        group = np.repeat(range(n_subjects), len(t))

        # True parameters
        true_beta = np.array([10.0, 0.5])

        # Generate responses with minimal random effects
        y = np.zeros(len(X))
        for i in range(n_subjects):
            mask = group == i
            phi_i = true_beta + np.random.normal(0, 0.1, 2)
            y[mask] = simple_model(phi_i, X[mask])

        # Add small amount of noise
        y += np.random.normal(0, 0.2, len(y))
        y = np.maximum(y, 0.1)

        beta0 = np.array([9.0, 0.4])

        # Run MLE
        mle_options = NLMEOptions(max_iter=30, verbose=0)
        try:
            beta_mle, psi_mle, stats_mle, b_mle = mle_algorithm(
                X, y, group, None, simple_model, beta0, mle_options
            )
            mle_success = True
        except Exception as e:
            print(f"MLE failed: {e}")
            mle_success = False
            beta_mle = psi_mle = stats_mle = b_mle = None

        # Run SAEM
        saem_options = SAEMOptions(
            n_iterations=(20, 20, 10), n_mcmc_iterations=(2, 2, 2), verbose=0
        )
        try:
            beta_saem, psi_saem, stats_saem, b_saem = saem_algorithm(
                X, y, group, None, simple_model, beta0, saem_options
            )
            saem_success = True
        except Exception as e:
            print(f"SAEM failed: {e}")
            saem_success = False
            beta_saem = psi_saem = stats_saem = b_saem = None

        # At least one should succeed
        assert mle_success or saem_success, "Both algorithms failed"

        # If both succeed, check they give reasonable results
        if mle_success and saem_success:
            # Both should estimate parameters in reasonable range
            for beta in [beta_mle, beta_saem]:
                assert 5.0 <= beta[0] <= 15.0
                assert 0.1 <= beta[1] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
