"""
Tests for PyNLME data types and structures.
"""

import numpy as np
import pytest

from pynlme.data_types import ErrorModel, NLMEOptions, NLMEStats, SAEMOptions


class TestNLMEStats:
    """Test NLMEStats data structure."""

    def test_creation_with_defaults(self):
        """Test creating NLMEStats with default values."""
        stats = NLMEStats()
        assert stats.logl is None
        assert stats.aic is None
        assert stats.bic is None
        assert stats.dfe is None
        assert stats.mse is None
        assert stats.rmse is None
        assert stats.sse is None

    def test_creation_with_values(self):
        """Test creating NLMEStats with specific values."""
        stats = NLMEStats(
            logl=-100.5, aic=205.0, bic=210.2, dfe=50, mse=0.25, rmse=0.5, sse=12.5
        )
        assert stats.logl == -100.5
        assert stats.aic == 205.0
        assert stats.bic == 210.2
        assert stats.dfe == 50
        assert stats.mse == 0.25
        assert stats.rmse == 0.5
        assert stats.sse == 12.5

    def test_residuals_initialization(self):
        """Test residuals are properly initialized."""
        stats = NLMEStats()
        assert stats.residuals is None

        residuals = {"pres": np.array([1, 2, 3]), "ires": np.array([0.5, 1.0, 1.5])}
        stats = NLMEStats(residuals=residuals)
        assert "pres" in stats.residuals
        assert "ires" in stats.residuals
        np.testing.assert_array_equal(stats.residuals["pres"], [1, 2, 3])


class TestNLMEOptions:
    """Test NLMEOptions configuration."""

    def test_default_options(self):
        """Test default option values."""
        options = NLMEOptions()
        assert options.approximation_type == "LME"
        assert options.optim_fun == "lbfgs"
        assert options.error_model == "constant"
        assert options.cov_parametrization == "logm"
        assert options.max_iter == 200
        assert options.tol_fun == 1e-6
        assert options.tol_x == 1e-6
        assert options.verbose == 0
        assert options.compute_std_errors is True
        assert options.refine_beta0 is True

    def test_custom_options(self):
        """Test creating options with custom values."""
        options = NLMEOptions(
            approximation_type="FOCE",
            error_model="proportional",
            max_iter=500,
            verbose=2,
            random_state=42,
        )
        assert options.approximation_type == "FOCE"
        assert options.error_model == "proportional"
        assert options.max_iter == 500
        assert options.verbose == 2
        assert options.random_state == 42

    def test_param_transform_array(self):
        """Test parameter transformation array."""
        transform = np.array([0, 1, 0, 2])  # identity, log, identity, probit
        options = NLMEOptions(param_transform=transform)
        np.testing.assert_array_equal(options.param_transform, transform)


class TestSAEMOptions:
    """Test SAEM-specific options."""

    def test_default_saem_options(self):
        """Test default SAEM option values."""
        options = SAEMOptions()
        assert options.n_iterations == (150, 150, 100)
        assert options.n_mcmc_iterations == (2, 2, 2)
        assert options.n_burn_in == 5
        assert options.step_size_sequence == "auto"
        assert options.tol_ll == 1e-6
        assert options.tol_sa == 1e-4

    def test_custom_saem_options(self):
        """Test creating SAEM options with custom values."""
        options = SAEMOptions(
            n_iterations=(100, 100, 50),
            n_mcmc_iterations=(3, 3, 3),
            n_burn_in=10,
            tol_sa=1e-5,
        )
        assert options.n_iterations == (100, 100, 50)
        assert options.n_mcmc_iterations == (3, 3, 3)
        assert options.n_burn_in == 10
        assert options.tol_sa == 1e-5


class TestErrorModel:
    """Test ErrorModel functionality."""

    def test_constant_error_model(self):
        """Test constant error model evaluation."""
        model = ErrorModel("constant", np.array([0.1]))
        y_pred = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.2])

        variance = model.evaluate(y_pred, theta)
        expected = np.full_like(y_pred, 0.2**2)
        np.testing.assert_array_almost_equal(variance, expected)

    def test_proportional_error_model(self):
        """Test proportional error model evaluation."""
        model = ErrorModel("proportional", np.array([0.1]))
        y_pred = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.2])

        variance = model.evaluate(y_pred, theta)
        expected = (0.2 * np.abs(y_pred)) ** 2
        np.testing.assert_array_almost_equal(variance, expected)

    def test_combined_error_model(self):
        """Test combined error model evaluation."""
        model = ErrorModel("combined", np.array([0.1, 0.2]))
        y_pred = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.1, 0.2])

        variance = model.evaluate(y_pred, theta)
        expected = (0.1 + 0.2 * np.abs(y_pred)) ** 2
        np.testing.assert_array_almost_equal(variance, expected)

    def test_exponential_error_model(self):
        """Test exponential error model evaluation."""
        model = ErrorModel("exponential", np.array([0.1]))
        y_pred = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.5])

        variance = model.evaluate(y_pred, theta)
        expected = np.exp(2 * 0.5 * np.log(np.abs(y_pred)))
        np.testing.assert_array_almost_equal(variance, expected)

    def test_invalid_error_model(self):
        """Test that invalid error model raises error."""
        model = ErrorModel("invalid", np.array([0.1]))
        y_pred = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.1])

        with pytest.raises(ValueError, match="Unknown error model"):
            model.evaluate(y_pred, theta)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
