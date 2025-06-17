#!/usr/bin/env python3
"""
Test suite for PyNLME performance optimization features.

This test suite validates the batched FFI optimization and other performance
improvements, ensuring that the Rust backend achieves expected speedups
for medium to large datasets.
"""

import time

import numpy as np
import pandas as pd
import pytest

import pynlme


def generate_test_data(n_subjects, n_timepoints=8):
    """Generate synthetic PK data for testing"""
    np.random.seed(42)  # For reproducible results

    timepoints = np.linspace(0.5, 24, n_timepoints)
    data = []

    for subject in range(n_subjects):
        # Individual parameters
        cl = 2.0 * np.exp(np.random.normal(0, 0.3))
        v = 10.0 * np.exp(np.random.normal(0, 0.2))

        for timepoint in timepoints:
            dose = 100
            conc_true = (dose / v) * np.exp(-cl / v * timepoint)
            conc_obs = conc_true * (1 + np.random.normal(0, 0.1))

            data.append(
                {
                    "subject": subject,
                    "time": timepoint,
                    "dose": dose,
                    "concentration": max(conc_obs, 0.01),
                }
            )

    df = pd.DataFrame(data)
    x = df[["time", "dose"]].values
    y = df["concentration"].values
    groups = df["subject"].values

    return x, y, groups


def simple_pk_model(beta, x, v=None):
    """Simple one-compartment PK model"""
    time = x[:, 0]
    dose = x[:, 1]
    cl, v_param = np.maximum(beta, 1e-6)

    conc = (dose / v_param) * np.exp(-cl / v_param * time)
    return conc


class TestPerformanceOptimization:
    """Test performance optimization features"""

    def test_small_dataset_performance(self):
        """Test that small datasets still work correctly"""
        x, y, groups = generate_test_data(10)
        beta0 = np.array([2.5, 8.0])

        start_time = time.perf_counter()
        beta, psi, stats, b = pynlme.nlmefit(
            x, y, groups, None, simple_pk_model, beta0, verbose=0
        )
        end_time = time.perf_counter()

        # Should converge successfully
        assert stats.logl is not None, "Small dataset should converge"
        assert end_time - start_time < 30, "Small dataset should be fast"

        # Parameters should be reasonable
        assert 1.0 < beta[0] < 5.0, "CL estimate should be reasonable"
        assert 5.0 < beta[1] < 15.0, "V estimate should be reasonable"

    def test_medium_dataset_performance(self):
        """Test performance on medium datasets (where batching starts)"""
        x, y, groups = generate_test_data(100)
        beta0 = np.array([2.5, 8.0])

        start_time = time.perf_counter()
        beta, psi, stats, b = pynlme.nlmefit(
            x, y, groups, None, simple_pk_model, beta0, verbose=0
        )
        end_time = time.perf_counter()

        fit_time = end_time - start_time

        # Should converge successfully
        assert stats.logl is not None, "Medium dataset should converge"
        assert fit_time < 60, (
            f"Medium dataset should be reasonably fast, took {fit_time:.3f}s"
        )

        # Log-likelihood should be reasonable
        assert stats.logl < 0, "Log-likelihood should be negative"

        print(f"Medium dataset (100 subjects, 800 obs): {fit_time:.3f}s")

    def test_large_dataset_performance(self):
        """Test performance on large datasets (where batching is most beneficial)"""
        x, y, groups = generate_test_data(200)
        beta0 = np.array([2.5, 8.0])

        start_time = time.perf_counter()
        beta, psi, stats, b = pynlme.nlmefit(
            x, y, groups, None, simple_pk_model, beta0, verbose=0
        )
        end_time = time.perf_counter()

        fit_time = end_time - start_time

        # Should converge successfully
        assert stats.logl is not None, "Large dataset should converge"
        assert fit_time < 120, (
            f"Large dataset should be manageable, took {fit_time:.3f}s"
        )

        print(f"Large dataset (200 subjects, 1600 obs): {fit_time:.3f}s")

    def test_scaling_behavior(self):
        """Test that performance scales reasonably with dataset size"""
        dataset_sizes = [25, 50, 100, 150]
        times = []

        for n_subjects in dataset_sizes:
            x, y, groups = generate_test_data(n_subjects)
            beta0 = np.array([2.5, 8.0])

            start_time = time.perf_counter()
            beta, psi, stats, b = pynlme.nlmefit(
                x, y, groups, None, simple_pk_model, beta0, verbose=0
            )
            end_time = time.perf_counter()

            fit_time = end_time - start_time
            times.append(fit_time)

            # Should converge
            assert stats.logl is not None, (
                f"Dataset with {n_subjects} subjects should converge"
            )

            print(f"Dataset size {n_subjects} subjects: {fit_time:.3f}s")

        # Check that scaling is not worse than quadratic
        # (With batching, we expect much better scaling)
        time_ratios = []
        size_ratios = []

        for i in range(1, len(times)):
            time_ratio = times[i] / times[0]
            size_ratio = dataset_sizes[i] / dataset_sizes[0]
            time_ratios.append(time_ratio)
            size_ratios.append(size_ratio)

            # Time should not grow faster than O(n^2)
            assert time_ratio <= size_ratio**2 * 2, (
                f"Scaling is too poor: {time_ratio:.2f}x time for {size_ratio:.2f}x size"
            )

        print("Scaling analysis:")
        for time_r, size_r in zip(time_ratios, size_ratios, strict=True):
            print(f"  {size_r:.1f}x dataset -> {time_r:.2f}x time")

    def test_convergence_consistency(self):
        """Test that optimization converges to consistent results"""
        x, y, groups = generate_test_data(75)
        beta0 = np.array([2.5, 8.0])

        # Run multiple times with different random seeds
        results = []
        for seed in [42, 123, 456]:
            np.random.seed(seed)

            beta, psi, stats, b = pynlme.nlmefit(
                x, y, groups, None, simple_pk_model, beta0, verbose=0
            )

            if stats.logl is not None:
                results.append({"beta": beta, "logl": stats.logl, "psi": psi})

        # Should have converged in most cases
        assert len(results) >= 2, "Should converge in most random seeds"

        # Results should be reasonably consistent
        if len(results) >= 2:
            beta_diff = np.abs(results[0]["beta"] - results[1]["beta"])
            assert np.all(beta_diff < 0.5), "Parameter estimates should be consistent"

            logl_diff = abs(results[0]["logl"] - results[1]["logl"])
            assert logl_diff < 10, "Log-likelihood should be consistent"

    def test_memory_efficiency(self):
        """Test that memory usage is reasonable for large datasets"""
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Test with a large dataset
        x, y, groups = generate_test_data(150)
        beta0 = np.array([2.5, 8.0])

        beta, psi, stats, b = pynlme.nlmefit(
            x, y, groups, None, simple_pk_model, beta0, verbose=0
        )

        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - baseline_memory

        # Memory usage should be reasonable (less than 500MB for this size)
        assert mem_used < 500, f"Memory usage too high: {mem_used:.1f} MB"

        print(f"Memory usage for 150 subjects: {mem_used:.1f} MB")


class TestBatchedFFIOptimization:
    """Test specific batched FFI optimization features"""

    def test_batching_threshold(self):
        """Test that batching is activated for appropriate dataset sizes"""
        # This is somewhat indirect since we can't directly inspect the backend
        # But we can test that performance improves for larger datasets

        small_x, small_y, small_groups = generate_test_data(50)  # Below threshold
        large_x, large_y, large_groups = generate_test_data(150)  # Above threshold

        beta0 = np.array([2.5, 8.0])

        # Time small dataset
        start_time = time.perf_counter()
        pynlme.nlmefit(
            small_x, small_y, small_groups, None, simple_pk_model, beta0, verbose=0
        )
        small_time = time.perf_counter() - start_time

        # Time large dataset
        start_time = time.perf_counter()
        pynlme.nlmefit(
            large_x, large_y, large_groups, None, simple_pk_model, beta0, verbose=0
        )
        large_time = time.perf_counter() - start_time

        # Large dataset should not be proportionally slower
        # (with batching, scaling should be much better)
        size_ratio = 150 / 50  # 3x
        time_ratio = large_time / small_time

        print(f"Small dataset (50 subjects): {small_time:.3f}s")
        print(f"Large dataset (150 subjects): {large_time:.3f}s")
        print(f"Size ratio: {size_ratio:.1f}x, Time ratio: {time_ratio:.2f}x")

        # With batching, time ratio should be much better than quadratic scaling
        assert time_ratio < size_ratio**1.5, (
            f"Scaling suggests batching may not be working: {time_ratio:.2f}x time for {size_ratio:.1f}x size"
        )

    def test_numerical_stability(self):
        """Test that batched optimization maintains numerical stability"""
        x, y, groups = generate_test_data(100)
        beta0 = np.array([2.5, 8.0])

        # Run multiple times to check for stability
        results = []
        for _ in range(3):
            beta, psi, stats, b = pynlme.nlmefit(
                x, y, groups, None, simple_pk_model, beta0, verbose=0
            )

            if stats.logl is not None:
                results.append(beta)

        # Should converge consistently
        assert len(results) >= 2, "Should converge consistently"

        # Results should be numerically stable
        if len(results) >= 2:
            for i in range(1, len(results)):
                diff = np.abs(results[i] - results[0])
                assert np.all(diff < 1e-6), "Results should be numerically identical"


def test_performance_regression():
    """Integration test to ensure no performance regression"""
    # This test serves as a regression test for the optimization
    x, y, groups = generate_test_data(100)
    beta0 = np.array([2.5, 8.0])

    start_time = time.perf_counter()
    beta, psi, stats, b = pynlme.nlmefit(
        x, y, groups, None, simple_pk_model, beta0, verbose=0
    )
    end_time = time.perf_counter()

    fit_time = end_time - start_time

    # Should converge
    assert stats.logl is not None, "Should converge successfully"

    # Should be reasonably fast (this is a regression test threshold)
    assert fit_time < 45, (
        f"Performance regression detected: {fit_time:.3f}s for 100 subjects"
    )

    # Parameters should be reasonable
    assert 1.0 < beta[0] < 5.0, "CL estimate should be reasonable"
    assert 5.0 < beta[1] < 15.0, "V estimate should be reasonable"

    print(f"Performance regression test passed: {fit_time:.3f}s for 100 subjects")


if __name__ == "__main__":
    # Run a quick performance check
    print("🚀 PyNLME Performance Optimization Test Suite")
    print("=" * 50)

    # Quick test
    test_performance_regression()

    print("\n✅ Performance tests completed successfully!")
    print(
        "Run with pytest for full test suite: pytest tests/test_performance_optimization.py -v"
    )
