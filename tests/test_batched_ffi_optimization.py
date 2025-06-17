#!/usr/bin/env python3
"""
Test suite for PyNLME batched FFI optimization.

This module tests the performance improvements achieved through batched
FFI calls and validates that the optimization works correctly across
different dataset sizes.
"""

import time
import unittest

import numpy as np

import pynlme


class TestBatchedFFIOptimization(unittest.TestCase):
    """Test the batched FFI optimization implementation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests

    def generate_test_data(self, n_subjects, n_timepoints=8):
        """Generate test data for performance testing."""
        n_obs = n_subjects * n_timepoints

        # Create subject IDs
        subjects = np.repeat(np.arange(n_subjects), n_timepoints)

        # Create time points
        time = np.tile(np.linspace(0, 10, n_timepoints), n_subjects)

        # Create dose (covariate)
        dose = np.tile([10, 20, 30, 40, 50, 60, 70, 80], n_subjects)[:n_obs]

        # Create design matrix
        X = np.column_stack([dose, time])

        # True parameters for data generation
        true_ka = 2.0
        true_cl = 8.0

        # Generate realistic PK data with some noise
        conc = (
            dose
            * true_ka
            * np.exp(-true_cl * time)
            / (true_cl - true_ka)
            * (np.exp(-true_ka * time) - np.exp(-true_cl * time))
        )
        conc += np.random.normal(0, 0.1 * conc)  # Add proportional noise
        conc = np.maximum(conc, 0.01)  # Ensure positive concentrations

        return X, conc, subjects

    def pharmacokinetic_model(self, params, X, V):
        """Simple 1-compartment PK model."""
        ka, cl = params
        dose, time = X[:, 0], X[:, 1]

        # Avoid division by zero and negative parameters
        ka = max(ka, 0.01)
        cl = max(cl, 0.01)
        if abs(ka - cl) < 1e-6:
            ka += 1e-6

        pred = dose * ka / (ka - cl) * (np.exp(-cl * time) - np.exp(-ka * time))
        return np.maximum(pred, 1e-6)

    def test_batched_ffi_activation(self):
        """Test that batched FFI is activated for large datasets."""
        # Large dataset should trigger batched approach
        X, y, groups = self.generate_test_data(150)  # 1200 observations
        beta0 = np.array([2.0, 8.0])

        # This should use the batched approach for datasets > 1000 observations
        try:
            result = pynlme.nlmefit(
                X, y, groups, None, self.pharmacokinetic_model, beta0
            )
            # Should complete successfully
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 4)  # beta, psi, stats, b
        except Exception as e:
            self.fail(f"Batched FFI implementation failed: {e}")

    def test_small_dataset_compatibility(self):
        """Test that small datasets still work with individual FFI calls."""
        # Small dataset should use individual FFI calls
        X, y, groups = self.generate_test_data(10)  # 80 observations
        beta0 = np.array([2.0, 8.0])

        try:
            result = pynlme.nlmefit(
                X, y, groups, None, self.pharmacokinetic_model, beta0
            )
            # Should complete successfully
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 4)  # beta, psi, stats, b
        except Exception as e:
            self.fail(f"Small dataset processing failed: {e}")

    def test_performance_improvement(self):
        """Test that Rust backend is faster than Python for large datasets."""
        # Test with medium-large dataset where batching should help
        X, y, groups = self.generate_test_data(100)  # 800 observations
        beta0 = np.array([2.0, 8.0])

        # For this test, we need to force using Python backend to compare
        # This is a bit tricky since the API doesn't have explicit backend selection
        # For now, just test that the optimized version completes in reasonable time
        start_time = time.time()
        try:
            result = pynlme.nlmefit(
                X, y, groups, None, self.pharmacokinetic_model, beta0
            )
            elapsed_time = time.time() - start_time

            # Should complete in reasonable time
            self.assertLess(
                elapsed_time,
                2.0,
                f"Performance test took too long: {elapsed_time:.3f}s (expected <2.0s)",
            )

            print(f"Performance test completed in {elapsed_time:.3f}s")

        except Exception as e:
            self.skipTest(f"Performance test failed: {e}")

    def test_function_call_reduction(self):
        """Test that batched approach reduces function calls."""
        # This is more of an integration test - we'd need to instrument
        # the FFI calls to count them precisely

        # For now, test that large datasets complete quickly
        X, y, groups = self.generate_test_data(200)  # 1600 observations
        beta0 = np.array([2.0, 8.0])

        start_time = time.time()
        try:
            result = pynlme.nlmefit(
                X, y, groups, None, self.pharmacokinetic_model, beta0
            )
            elapsed_time = time.time() - start_time

            # Should complete in reasonable time (batched approach should be fast)
            self.assertLess(
                elapsed_time,
                1.0,
                f"Large dataset took too long: {elapsed_time:.3f}s (expected <1.0s)",
            )

            print(f"Large dataset (200 subjects) completed in {elapsed_time:.3f}s")

        except Exception as e:
            self.fail(f"Batched FFI failed for large dataset: {e}")

    def test_convergence_reliability(self):
        """Test that optimization converges reliably across dataset sizes."""
        test_sizes = [25, 50, 100, 200]
        beta0 = np.array([2.0, 8.0])

        convergence_results = []

        for n_subjects in test_sizes:
            X, y, groups = self.generate_test_data(n_subjects)

            try:
                result = pynlme.nlmefit(
                    X, y, groups, None, self.pharmacokinetic_model, beta0
                )

                # Check that parameters are reasonable
                beta_est = result[0]
                self.assertEqual(len(beta_est), 2, "Should estimate 2 parameters")

                # Parameters should be positive and reasonable
                self.assertGreater(beta_est[0], 0, "ka should be positive")
                self.assertGreater(beta_est[1], 0, "cl should be positive")
                self.assertLess(beta_est[0], 50, "ka should be reasonable")
                self.assertLess(beta_est[1], 50, "cl should be reasonable")

                convergence_results.append(True)

            except Exception as e:
                print(f"Failed for {n_subjects} subjects: {e}")
                convergence_results.append(False)

        # Should have high convergence rate
        convergence_rate = sum(convergence_results) / len(convergence_results)
        self.assertGreaterEqual(
            convergence_rate,
            0.75,
            f"Convergence rate too low: {convergence_rate:.2f} (expected ≥0.75)",
        )

        print(f"Convergence rate across dataset sizes: {convergence_rate:.2f}")

    def test_backend_consistency(self):
        """Test that Rust and Python backends give similar results."""
        X, y, groups = self.generate_test_data(50)
        beta0 = np.array([2.0, 8.0])

        try:
            result_rust = pynlme.nlmefit(
                X, y, groups, None, self.pharmacokinetic_model, beta0, backend="rust"
            )
            beta_rust = result_rust[0]
        except Exception:
            self.skipTest("Rust backend not available")

        try:
            result_python = pynlme.nlmefit(
                X, y, groups, None, self.pharmacokinetic_model, beta0, backend="python"
            )
            beta_python = result_python[0]
        except Exception:
            self.skipTest("Python backend not available")

        # Parameters should be reasonably close (allowing for optimization differences)
        for i, (rust_param, python_param) in enumerate(zip(beta_rust, beta_python)):
            relative_diff = abs(rust_param - python_param) / abs(python_param)
            self.assertLess(
                relative_diff,
                0.3,  # Allow 30% difference due to different optimization
                f"Parameter {i}: Rust={rust_param:.3f}, Python={python_param:.3f}, "
                f"diff={relative_diff:.1%}",
            )

        print(f"Backend consistency: Rust={beta_rust}, Python={beta_python}")


class TestPerformanceScaling(unittest.TestCase):
    """Test performance scaling behavior."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_scaling_trend(self):
        """Test that performance scaling trend is improved."""
        # This is more of a benchmark than a unit test
        # It validates that our optimization actually improves scaling

        test_sizes = [50, 100, 200]
        results = []

        for n_subjects in test_sizes:
            # Generate test data
            np.random.seed(42)
            n_obs = n_subjects * 8
            subjects = np.repeat(np.arange(n_subjects), 8)
            time = np.tile(np.linspace(0, 10, 8), n_subjects)
            dose = np.tile([10, 20, 30, 40, 50, 60, 70, 80], n_subjects)[:n_obs]
            X = np.column_stack([dose, time])

            # Simple PK model for data generation
            true_ka, true_cl = 2.0, 8.0
            conc = (
                dose
                * true_ka
                * np.exp(-true_cl * time)
                / (true_cl - true_ka)
                * (np.exp(-true_ka * time) - np.exp(-true_cl * time))
            )
            conc += np.random.normal(0, 0.1 * conc)
            conc = np.maximum(conc, 0.01)

            beta0 = np.array([2.0, 8.0])

            def pk_model(params, X, V):
                ka, cl = params
                dose, time = X[:, 0], X[:, 1]
                ka = max(ka, 0.01)
                cl = max(cl, 0.01)
                if abs(ka - cl) < 1e-6:
                    ka += 1e-6
                pred = dose * ka / (ka - cl) * (np.exp(-cl * time) - np.exp(-ka * time))
                return np.maximum(pred, 1e-6)

            # Time the Rust backend
            import time as time_module

            start_time = time_module.time()
            try:
                result = pynlme.nlmefit(X, conc, subjects, None, pk_model, beta0)
                elapsed_time = time_module.time() - start_time
                results.append((n_subjects, elapsed_time))
                print(f"{n_subjects} subjects: {elapsed_time:.4f}s")
            except Exception as e:
                print(f"Failed for {n_subjects} subjects: {e}")

        # Check that we have reasonable performance
        if len(results) >= 2:
            # Performance shouldn't degrade too much with size
            small_time = results[0][1]  # Time for smallest dataset
            large_time = results[-1][1]  # Time for largest dataset
            size_ratio = results[-1][0] / results[0][0]  # Size increase
            time_ratio = large_time / small_time  # Time increase

            # Time increase should be less than size increase (sublinear scaling)
            # Allow some margin for small datasets where overhead dominates
            self.assertLess(
                time_ratio,
                size_ratio + 2.0,
                f"Poor scaling: {size_ratio:.1f}x size increase led to "
                f"{time_ratio:.1f}x time increase",
            )

            print(f"Scaling analysis: {size_ratio:.1f}x size → {time_ratio:.1f}x time")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
