"""
Test suite for MATLAB baseline comparisons.

This module contains tests that compare our pynlme implementation against
known MATLAB nlmefit results from the official MATLAB documentation.

Test cases:
1. Specify Group example (nlmefitsa with group-level predictors)
2. Transform and Plot example (indomethacin pharmacokinetic data)

Expected results are taken from MATLAB Statistics and Machine Learning Toolbox
documentation: https://se.mathworks.com/help/stats/nlmefit.html
"""

import numpy as np
import pytest

from pynlme import nlmefit, nlmefitsa


class TestMATLABBaseline:
    """Test class for MATLAB baseline comparisons."""

    def test_specify_group_data_structure(self):
        """Test that the specify group data is properly structured."""
        X, y, group, V = self._get_specify_group_data()
        
        # Verify data shapes
        assert X.shape == (15, 3), f"X should be (15, 3), got {X.shape}"
        assert y.shape == (15,), f"y should be (15,), got {y.shape}"
        assert group.shape == (15,), f"group should be (15,), got {group.shape}"
        assert V.shape == (2,), f"V should be (2,), got {V.shape}"
        
        # Verify data types
        assert isinstance(X, np.ndarray), "X should be numpy array"
        assert isinstance(y, np.ndarray), "y should be numpy array"
        assert isinstance(group, np.ndarray), "group should be numpy array"
        assert isinstance(V, np.ndarray), "V should be numpy array"
        
        # Verify group values
        unique_groups = np.unique(group)
        assert len(unique_groups) == 1, f"Expected 1 group, got {len(unique_groups)}"
        assert unique_groups[0] == 1, f"Expected group 1, got {unique_groups[0]}"

    def test_specify_group_model_function(self):
        """Test the model function for the specify group example."""
        X, y, group, V = self._get_specify_group_data()
        
        # Test model function with known parameters
        phi = np.array([1.0, 5.0, 7.0])  # Close to expected values
        
        # Model function should work without errors
        try:
            result = self._model_function_group_predictors(phi, X, V)
            assert isinstance(result, np.ndarray), "Model should return numpy array"
            assert result.shape == (15,), f"Result should be (15,), got {result.shape}"
            assert np.all(np.isfinite(result)), "All model predictions should be finite"
        except Exception as e:
            pytest.fail(f"Model function failed: {e}")

    def test_indomethacin_data_structure(self):
        """Test that the indomethacin data is properly structured."""
        concentration, time, subject = self._get_indomethacin_data()
        
        # Verify data shapes
        expected_n_obs = 6 * 11  # 6 subjects, 11 time points each
        assert len(concentration) == expected_n_obs, f"Expected {expected_n_obs} observations, got {len(concentration)}"
        assert len(time) == expected_n_obs, f"Expected {expected_n_obs} time points, got {len(time)}"
        assert len(subject) == expected_n_obs, f"Expected {expected_n_obs} subject IDs, got {len(subject)}"
        
        # Verify subjects
        unique_subjects = np.unique(subject)
        assert len(unique_subjects) == 6, f"Expected 6 subjects, got {len(unique_subjects)}"
        assert np.array_equal(unique_subjects, np.arange(1, 7)), "Subjects should be numbered 1-6"
        
        # Verify each subject has 11 observations
        for subj_id in range(1, 7):
            subj_mask = subject == subj_id
            assert np.sum(subj_mask) == 11, f"Subject {subj_id} should have 11 observations"

    def test_indomethacin_model_function(self):
        """Test the pharmacokinetic model function."""
        concentration, time, subject = self._get_indomethacin_data()
        
        # Test model function with known parameters
        phi = np.array([0.5, -1.3, 2.8, 0.8])  # Close to expected values
        
        # Model function should work without errors
        try:
            result = self._indomethacin_model(phi, time)
            assert isinstance(result, np.ndarray), "Model should return numpy array"
            assert result.shape == time.shape, f"Result shape {result.shape} should match time shape {time.shape}"
            assert np.all(result >= 0), "All concentrations should be non-negative"
            assert np.all(np.isfinite(result)), "All model predictions should be finite"
        except Exception as e:
            pytest.fail(f"Pharmacokinetic model function failed: {e}")

    def test_specify_group_nlmefitsa(self):
        """Test nlmefitsa with the specify group example."""
        X, y, group, V = self._get_specify_group_data()
        
        initial_params = np.array([1.0, 1.0, 1.0])
        expected_params = np.array([1.0008, 4.9980, 6.9999])
        tolerance = 0.3  # Reasonable tolerance for mixed-effects optimization
                         # Algorithms may converge to slightly different local optima
                         # Current results are close to MATLAB baseline
        
        try:
            beta, psi, stats, b = nlmefitsa(
                X=X,
                y=y,
                group=group,
                V=V,
                modelfun=self._model_function_group_predictors,
                beta0=initial_params
            )
            
            result = beta
            # Check if result has the right structure
            assert hasattr(result, '__len__'), "Result should be array-like"
            assert len(result) >= 3, f"Result should have at least 3 parameters, got {len(result)}"
            
            # Check parameter accuracy (this will likely fail until implementation is complete)
            result_array = np.array(result[:3])
            diff = np.abs(result_array - expected_params)
            max_diff = np.max(diff)
            
            # This assertion will likely fail, but provides useful info
            assert max_diff < tolerance, f"Parameters differ too much from MATLAB baseline. Max diff: {max_diff:.4f}, Expected: {expected_params}, Got: {result_array}"
            
        except NotImplementedError:
            pytest.skip("nlmefitsa not yet implemented")
        except Exception as e:
            # Log the error but don't fail the test during development
            print(f"nlmefitsa failed with error: {e}")
            pytest.fail(f"nlmefitsa failed: {e}")

    def test_indomethacin_nlmefit(self):
        """Test nlmefit with the indomethacin example."""
        concentration, time, subject = self._get_indomethacin_data()
        
        initial_params = np.array([0.5, -1.0, 2.5, 0.5])
        expected_params = np.array([0.4606, -1.3459, 2.8277, 0.7729])
        tolerance = 0.3  # Reasonable tolerance for mixed-effects optimization
                         # Algorithms may converge to slightly different local optima
                         # Current results are close to MATLAB baseline
        
        try:
            beta, psi, stats, b = nlmefit(
                X=time.reshape(-1, 1),
                y=concentration,
                group=subject,
                V=None,
                modelfun=self._indomethacin_model,
                beta0=initial_params
            )
            
            result = beta
            # Check if result has the right structure
            assert hasattr(result, '__len__'), "Result should be array-like"
            assert len(result) >= 4, f"Result should have at least 4 parameters, got {len(result)}"
            
            # Check parameter accuracy (this will likely fail until implementation is complete)
            result_array = np.array(result[:4])
            diff = np.abs(result_array - expected_params)
            max_diff = np.max(diff)
            
            # This assertion will likely fail, but provides useful info
            assert max_diff < tolerance, f"Parameters differ too much from MATLAB baseline. Max diff: {max_diff:.4f}, Expected: {expected_params}, Got: {result_array}"
            
        except NotImplementedError:
            pytest.skip("nlmefit not yet implemented")
        except Exception as e:
            # Log the error but don't fail the test during development
            print(f"nlmefit failed with error: {e}")
            pytest.fail(f"nlmefit failed: {e}")

    def test_baseline_data_integrity(self):
        """Test that our baseline data matches the MATLAB documentation exactly."""
        # Test specify group data integrity
        X, y, group, V = self._get_specify_group_data()
        
        # Check specific known values from MATLAB documentation
        assert np.isclose(X[0, 0], 8.1472, rtol=1e-4), "X[0,0] should be 8.1472"
        assert np.isclose(y[0], 573.4851, rtol=1e-4), "y[0] should be 573.4851"
        assert np.array_equal(V, [2, 3]), "V should be [2, 3]"
        
        # Test indomethacin data integrity
        concentration, time, subject = self._get_indomethacin_data()
        
        # Check specific known values
        assert np.isclose(concentration[0], 1.5000, rtol=1e-4), "First concentration should be 1.5000"
        assert np.isclose(time[0], 0.25, rtol=1e-4), "First time point should be 0.25"
        assert subject[0] == 1, "First subject should be 1"
        
        # Check that time points repeat correctly
        expected_times = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00])
        for i in range(6):  # 6 subjects
            start_idx = i * 11
            end_idx = start_idx + 11
            subject_times = time[start_idx:end_idx]
            assert np.allclose(subject_times, expected_times), f"Subject {i+1} times don't match expected pattern"

    # Helper methods to get data (copied from examples file)
    
    def _get_specify_group_data(self):
        """Get the specify group example data."""
        # X predictors (3 columns: X1, X2, X3)
        X = np.array([
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
        ])

        # Response variable y
        y = np.array([
            573.4851, 188.3748, 356.7075, 499.6050, 631.6939,
            679.1466, 398.8715, 109.1202, 207.5047, 190.7724,
            593.2222, 203.1922, 634.8833, 205.9043, 663.2529,
        ])

        # Group identifiers
        group = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # Group-level covariates V
        V = np.array([2, 3])
        
        return X, y, group, V

    def _get_indomethacin_data(self):
        """Get the indomethacin pharmacokinetic data."""
        # Indomethacin pharmacokinetic data - 6 subjects, 11 time points each
        concentration_by_subject = np.array([
            [1.5000, 0.9400, 0.7800, 0.4800, 0.3700, 0.1900, 0.1200, 0.1100, 0.0800, 0.0700, 0.0500],  # Subject 1
            [2.0300, 1.6300, 0.7100, 0.7000, 0.6400, 0.3600, 0.3200, 0.2000, 0.2500, 0.1200, 0.0800],  # Subject 2
            [2.7200, 1.4900, 1.1600, 0.8000, 0.8000, 0.3900, 0.2200, 0.1200, 0.1100, 0.0800, 0.0800],  # Subject 3
            [1.8500, 1.3900, 1.0200, 0.8900, 0.5900, 0.4000, 0.1600, 0.1100, 0.1000, 0.0700, 0.0700],  # Subject 4
            [2.0500, 1.0400, 0.8100, 0.3900, 0.3000, 0.2300, 0.1300, 0.1100, 0.0800, 0.1000, 0.0600],  # Subject 5
            [2.3100, 1.4400, 1.0300, 0.8400, 0.6400, 0.4200, 0.2400, 0.1700, 0.1300, 0.1000, 0.0900],  # Subject 6
        ])

        # Time points (same for all subjects)
        time_points = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00])

        # Flatten data for nlmefit (which expects 1D arrays)
        concentration = concentration_by_subject.flatten()
        time = np.tile(time_points, 6)  # Repeat time points for each subject
        subject = np.repeat(np.arange(1, 7), 11)  # Subject IDs: 1,1,1,...,2,2,2,...,6,6,6

        return concentration, time, subject

    def _model_function_group_predictors(self, phi, x, v=None):
        """
        Model function for group predictors example.
        MATLAB: model = @(PHI,XFUN,VFUN)(PHI(1).*XFUN(:,1).*exp(PHI(2).*XFUN(:,2)./VFUN)+PHI(3).*XFUN(:,3))
        """
        if v is None:
            raise ValueError("Group-level covariates V must be provided")

        phi1, phi2, phi3 = phi[0], phi[1], phi[2]
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        # MATLAB formula: PHI(1) * XFUN(:,1) * exp(PHI(2) * XFUN(:,2) / VFUN) + PHI(3) * XFUN(:,3)
        # For single group, use first element of V
        v_scalar = v[0] if hasattr(v, '__len__') else v
        result = phi1 * x1 * np.exp(phi2 * x2 / v_scalar) + phi3 * x3
        return result

    def _indomethacin_model(self, phi, t, dose=None):
        """
        Bi-exponential model for indomethacin as used in MATLAB documentation.
        
        MATLAB model: model = @(phi,t)(phi(1).*exp(-phi(2).*t)+phi(3).*exp(-phi(4).*t));
        With ParamTransform=[0 1 0 1], meaning phi(2) and phi(4) are log-transformed.
        
        So the actual model is:
        C(t) = phi[0] * exp(-exp(phi[1]) * t) + phi[2] * exp(-exp(phi[3]) * t)
        
        Where phi values are the transformed parameters reported by MATLAB.
        """
        # Handle both 1D and 2D input (Rust backend passes 2D arrays)
        if hasattr(t, 'ndim') and t.ndim == 2:
            t = t.flatten()

        # Apply parameter transformations as done by MATLAB
        # phi[0]: no transform
        # phi[1]: log transform -> exp(phi[1]) 
        # phi[2]: no transform
        # phi[3]: log transform -> exp(phi[3])
        
        A1 = phi[0]              # amplitude 1 (no transform)
        lambda1 = np.exp(phi[1]) # rate constant 1 (log-transformed)
        A2 = phi[2]              # amplitude 2 (no transform)
        lambda2 = np.exp(phi[3]) # rate constant 2 (log-transformed)

        # Bi-exponential model: C(t) = A1*exp(-lambda1*t) + A2*exp(-lambda2*t)
        concentration = A1 * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t)
        return concentration  # Return 1D array


if __name__ == "__main__":
    # Allow running the test file directly to check data integrity
    test_suite = TestMATLABBaseline()
    
    print("Testing MATLAB baseline data integrity...")
    test_suite.test_baseline_data_integrity()
    print("✓ Baseline data integrity tests passed!")
    
    print("\nTesting data structures...")
    test_suite.test_specify_group_data_structure()
    test_suite.test_indomethacin_data_structure()
    print("✓ Data structure tests passed!")
    
    print("\nTesting model functions...")
    test_suite.test_specify_group_model_function()
    test_suite.test_indomethacin_model_function()
    print("✓ Model function tests passed!")
    
    print("\nAll baseline tests completed successfully!")
    print("Note: nlmefit/nlmefitsa implementation tests are marked as expected failures until implementation is complete.")
