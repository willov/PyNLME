"""
Tests for parameter transformation functions in PyNLME.
"""

import numpy as np

from pynlme.utils import transform_parameters


class TestTransformParameters:
    """Test parameter transformation functionality."""

    def test_no_transform_none_codes(self):
        """Test that None transform codes return original parameters."""
        phi = np.array([1.0, 2.0, -1.0])
        result = transform_parameters(phi, None)

        np.testing.assert_array_equal(result, phi)
        # Ensure it's a copy, not the same object
        assert result is not phi

    def test_no_transform_zero_codes(self):
        """Test identity transformation (code 0)."""
        phi = np.array([1.0, 2.0, -1.0])
        codes = np.array([0, 0, 0])
        result = transform_parameters(phi, codes)

        np.testing.assert_array_equal(result, phi)

    def test_mixed_codes_with_identity(self):
        """Test mixed transformation codes including identity."""
        phi = np.array([1.0, 2.0, -1.0, 0.5])
        codes = np.array([0, 1, 0, 1])  # identity, exp, identity, exp
        result = transform_parameters(phi, codes)

        expected = np.array([1.0, np.exp(2.0), -1.0, np.exp(0.5)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_exponential_transform(self):
        """Test exponential transformation (code 1)."""
        phi = np.array([0.0, 1.0, -1.0, 2.0])
        codes = np.array([1, 1, 1, 1])
        result = transform_parameters(phi, codes)

        expected = np.exp(phi)
        np.testing.assert_array_almost_equal(result, expected)

    def test_exponential_transform_single_param(self):
        """Test exponential transformation for single parameter."""
        phi = np.array([2.0])
        codes = np.array([1])
        result = transform_parameters(phi, codes)

        expected = np.array([np.exp(2.0)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_probit_transform(self):
        """Test probit transformation (code 2)."""
        phi = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        codes = np.array([2, 2, 2, 2, 2])
        result = transform_parameters(phi, codes)

        # Expected: 0.5 * (1 + tanh(0.7978 * phi))
        expected = 0.5 * (1 + np.tanh(0.7978 * phi))
        np.testing.assert_array_almost_equal(result, expected)

    def test_probit_bounds(self):
        """Test that probit transformation stays within (0,1)."""
        phi = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        codes = np.array([2, 2, 2, 2, 2])
        result = transform_parameters(phi, codes)

        # All values should be between 0 and 1
        assert np.all(result > 0)
        assert np.all(result < 1)

        # Should be approximately 0.5 for phi=0
        assert abs(result[2] - 0.5) < 1e-10

    def test_logit_transform(self):
        """Test logit transformation (code 3)."""
        phi = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        codes = np.array([3, 3, 3, 3, 3])
        result = transform_parameters(phi, codes)

        # Expected: 1 / (1 + exp(-phi))
        expected = 1 / (1 + np.exp(-phi))
        np.testing.assert_array_almost_equal(result, expected)

    def test_logit_bounds(self):
        """Test that logit transformation stays within (0,1)."""
        phi = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        codes = np.array([3, 3, 3, 3, 3])
        result = transform_parameters(phi, codes)

        # All values should be between 0 and 1
        assert np.all(result > 0)
        assert np.all(result < 1)

        # Should be exactly 0.5 for phi=0
        assert abs(result[2] - 0.5) < 1e-10

    def test_logit_vs_probit(self):
        """Test difference between logit and probit transformations."""
        phi = np.array([0.0])
        logit_codes = np.array([3])
        probit_codes = np.array([2])

        logit_result = transform_parameters(phi, logit_codes)
        probit_result = transform_parameters(phi, probit_codes)

        # Both should give 0.5 for phi=0
        np.testing.assert_array_almost_equal(logit_result, [0.5])
        np.testing.assert_array_almost_equal(probit_result, [0.5])

    def test_mixed_transformations(self):
        """Test mixed transformation codes."""
        phi = np.array([1.0, 0.0, -1.0, 2.0])
        codes = np.array([0, 1, 2, 3])  # identity, exp, probit, logit
        result = transform_parameters(phi, codes)

        expected = np.array(
            [
                1.0,  # identity
                np.exp(0.0),  # exp(0) = 1
                0.5 * (1 + np.tanh(0.7978 * (-1.0))),  # probit
                1 / (1 + np.exp(-2.0)),  # logit
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_array_support(self):
        """Test that function works with 2D arrays."""
        phi = np.array([[1.0, 0.0], [-1.0, 2.0]])
        codes = np.array([1, 3])  # exp, logit
        result = transform_parameters(phi, codes)

        expected = np.array(
            [
                [np.exp(1.0), 1 / (1 + np.exp(-0.0))],
                [np.exp(-1.0), 1 / (1 + np.exp(-2.0))],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_edge_cases_exponential(self):
        """Test edge cases for exponential transformation."""
        # Very large positive value
        phi = np.array([100.0])
        codes = np.array([1])
        result = transform_parameters(phi, codes)
        assert np.isfinite(result[0]) or np.isinf(result[0])  # Should handle overflow

        # Very large negative value
        phi = np.array([-100.0])
        codes = np.array([1])
        result = transform_parameters(phi, codes)
        assert result[0] >= 0  # Should be very close to 0

    def test_edge_cases_logit_probit(self):
        """Test edge cases for logit and probit transformations."""
        # Very large positive values
        phi = np.array([100.0, 100.0])
        codes = np.array([2, 3])  # probit, logit
        result = transform_parameters(phi, codes)

        # Both should approach 1 (may reach exactly 1.0 due to floating-point limits)
        assert result[0] <= 1.0  # probit
        assert result[1] <= 1.0  # logit
        assert result[0] > 0.99  # probit close to 1
        assert result[1] > 0.99  # logit close to 1

        # Very large negative values
        phi = np.array([-100.0, -100.0])
        codes = np.array([2, 3])  # probit, logit
        result = transform_parameters(phi, codes)

        # Both should approach 0 (may reach exactly 0.0 due to floating-point limits)
        assert result[0] >= 0.0  # probit
        assert result[1] >= 0.0  # logit
        assert result[0] < 0.01  # probit close to 0
        assert result[1] < 0.01  # logit close to 0

    def test_invalid_transform_codes(self):
        """Test behavior with invalid transformation codes."""
        phi = np.array([1.0, 2.0])
        codes = np.array([0, 5])  # 5 is not a valid code

        # Should handle gracefully (only apply known transforms)
        result = transform_parameters(phi, codes)

        # First parameter should be unchanged (code 0)
        assert result[0] == 1.0
        # Second parameter should also be unchanged (unknown code)
        assert result[1] == 2.0

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        phi = np.array([])
        codes = np.array([])
        result = transform_parameters(phi, codes)

        assert len(result) == 0
        assert result.shape == phi.shape

    def test_single_parameter_different_codes(self):
        """Test all transformation codes on a single parameter value."""
        phi_val = 1.0
        phi = np.array([phi_val])

        # Test each transformation code
        for code in [0, 1, 2, 3]:
            codes = np.array([code])
            result = transform_parameters(phi, codes)

            if code == 0:  # identity
                assert result[0] == phi_val
            elif code == 1:  # exponential
                assert result[0] == np.exp(phi_val)
            elif code == 2:  # probit
                expected = 0.5 * (1 + np.tanh(0.7978 * phi_val))
                np.testing.assert_almost_equal(result[0], expected)
            elif code == 3:  # logit
                expected = 1 / (1 + np.exp(-phi_val))
                np.testing.assert_almost_equal(result[0], expected)

    def test_copy_behavior(self):
        """Test that original array is not modified."""
        phi = np.array([1.0, 2.0, 3.0])
        phi_original = phi.copy()
        codes = np.array([1, 1, 1])

        result = transform_parameters(phi, codes)

        # Original array should be unchanged
        np.testing.assert_array_equal(phi, phi_original)
        # Result should be different
        assert not np.array_equal(result, phi)


class TestTransformParametersIntegration:
    """Integration tests for parameter transformations."""

    def test_pharmacokinetic_example(self):
        """Test typical pharmacokinetic parameter transformations."""
        # Typical PK parameters in log space: log(CL), log(V), logit(F)
        phi = np.array([2.3, 4.1, 0.5])  # log(10), log(60), logit(0.62)
        codes = np.array([1, 1, 3])  # exp, exp, logit

        result = transform_parameters(phi, codes)

        # Should give reasonable PK values
        cl = result[0]  # clearance
        v = result[1]  # volume
        f = result[2]  # bioavailability

        assert cl > 0  # clearance must be positive
        assert v > 0  # volume must be positive
        assert 0 < f < 1  # bioavailability between 0 and 1

        # Check specific values
        np.testing.assert_almost_equal(cl, np.exp(2.3), decimal=5)
        np.testing.assert_almost_equal(v, np.exp(4.1), decimal=5)
        np.testing.assert_almost_equal(f, 1 / (1 + np.exp(-0.5)), decimal=5)

    def test_probability_transformations(self):
        """Test transformations for probability parameters."""
        # Parameters that should be constrained to (0,1)
        phi = np.array([-2.0, 0.0, 2.0])

        # Test both probit and logit
        probit_codes = np.array([2, 2, 2])
        logit_codes = np.array([3, 3, 3])

        probit_result = transform_parameters(phi, probit_codes)
        logit_result = transform_parameters(phi, logit_codes)

        # All results should be probabilities
        for result in [probit_result, logit_result]:
            assert np.all(result > 0)
            assert np.all(result < 1)
            # Middle value should be 0.5
            np.testing.assert_almost_equal(result[1], 0.5, decimal=5)
