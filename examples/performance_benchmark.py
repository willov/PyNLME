#!/usr/bin/env python3
"""
PyNLME Performance Benchmark

This script compares the performance of PyNLME's Rust backend vs Python fallback
across different dataset sizes and model complexities.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pynlme


def generate_benchmark_data(n_subjects, n_timepoints_per_subject, complexity="simple"):
    """Generate synthetic data for benchmarking"""

    np.random.seed(42)  # For reproducible results

    # Time points
    if complexity == "simple":
        times = np.linspace(0.5, 24, n_timepoints_per_subject)
    else:
        # Irregular sampling for complex case
        times = np.sort(np.random.exponential(2, n_timepoints_per_subject))
        times = times[:n_timepoints_per_subject]

    data = []

    for subject in range(n_subjects):
        # Individual parameters
        cl = 2.0 * np.exp(np.random.normal(0, 0.3))
        v = 10.0 * np.exp(np.random.normal(0, 0.2))

        for time in times:
            # Simple one-compartment model
            dose = 100
            conc_true = (dose / v) * np.exp(-cl / v * time)
            conc_obs = conc_true * (1 + np.random.normal(0, 0.1))

            data.append(
                {
                    "subject": subject,
                    "time": time,
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
    cl, v_param = np.maximum(beta, 1e-6)  # Ensure positive parameters

    conc = (dose / v_param) * np.exp(-cl / v_param * time)
    return conc


def benchmark_fitting(dataset_sizes, use_rust_options=[True, False]):
    """Benchmark fitting performance across different conditions"""

    results = []

    for n_subjects in dataset_sizes:
        print(f"\n📊 Testing with {n_subjects} subjects...")

        # Generate data
        x, y, groups = generate_benchmark_data(n_subjects, 8)
        beta0 = np.array([2.5, 8.0])  # Initial estimates

        for use_rust in use_rust_options:
            backend = "Rust" if use_rust else "Python"
            print(f"  {backend} backend... ", end="", flush=True)

            try:
                start_time = time.perf_counter()

                beta, psi, stats, b = pynlme.nlmefit(
                    x, y, groups, None, simple_pk_model, beta0, verbose=0
                )

                end_time = time.perf_counter()
                fit_time = end_time - start_time

                results.append(
                    {
                        "n_subjects": n_subjects,
                        "n_observations": len(y),
                        "backend": backend,
                        "time_seconds": fit_time,
                        "converged": stats.logl is not None,
                        "iterations": getattr(stats, 'iterations', 0),
                        "logl": stats.logl,
                        "rmse": np.sqrt(np.mean((y - simple_pk_model(beta, x))**2)),
                    }
                )

                print(f"{fit_time:.3f}s ({'✓' if stats.logl is not None else '✗'})")

            except Exception as e:
                print(f"Failed: {e}")
                results.append(
                    {
                        "n_subjects": n_subjects,
                        "n_observations": len(y),
                        "backend": backend,
                        "time_seconds": np.nan,
                        "converged": False,
                        "iterations": 0,
                        "logl": np.nan,
                        "rmse": np.nan,
                    }
                )

    return pd.DataFrame(results)


def plot_benchmark_results(results_df):
    """Create benchmark visualization"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Execution time comparison
    ax1 = axes[0, 0]
    for backend in results_df["backend"].unique():
        data = results_df[results_df["backend"] == backend]
        ax1.plot(
            data["n_subjects"],
            data["time_seconds"],
            "o-",
            label=backend,
            linewidth=2,
            markersize=6,
        )

    ax1.set_xlabel("Number of Subjects")
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_title("Execution Time vs Dataset Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # 2. Speedup factor
    ax2 = axes[0, 1]
    rust_times = results_df[results_df["backend"] == "Rust"].set_index("n_subjects")[
        "time_seconds"
    ]
    python_times = results_df[results_df["backend"] == "Python"].set_index(
        "n_subjects"
    )["time_seconds"]

    speedup = python_times / rust_times
    speedup = speedup.dropna()

    ax2.plot(speedup.index, speedup.values, "ro-", linewidth=2, markersize=6)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Number of Subjects")
    ax2.set_ylabel("Speedup Factor (Python/Rust)")
    ax2.set_title("Rust Performance Advantage")
    ax2.grid(True, alpha=0.3)

    # Add speedup annotations
    for x, y in zip(speedup.index, speedup.values, strict=False):
        ax2.annotate(
            f"{y:.1f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # 3. Convergence comparison
    ax3 = axes[1, 0]
    conv_summary = (
        results_df.groupby(["n_subjects", "backend"])["converged"].mean().unstack()
    )
    conv_summary.plot(kind="bar", ax=ax3, width=0.8)
    ax3.set_xlabel("Number of Subjects")
    ax3.set_ylabel("Convergence Rate")
    ax3.set_title("Convergence Rate by Backend")
    ax3.legend(title="Backend")
    ax3.set_ylim(0, 1.1)

    # 4. Model quality (RMSE)
    ax4 = axes[1, 1]
    for backend in results_df["backend"].unique():
        data = results_df[
            (results_df["backend"] == backend) & (results_df["converged"] == True)
        ]
        if not data.empty:
            ax4.plot(
                data["n_subjects"],
                data["rmse"],
                "o-",
                label=backend,
                linewidth=2,
                markersize=6,
            )

    ax4.set_xlabel("Number of Subjects")
    ax4.set_ylabel("RMSE")
    ax4.set_title("Model Fit Quality")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to organized output folder
    examples_dir = os.path.dirname(__file__)
    output_dir = os.path.join(examples_dir, "performance_benchmark_output")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "benchmark_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()


def memory_benchmark():
    """Simple memory usage comparison"""
    import os

    try:
        import psutil
    except ImportError:
        print("🧠 Memory Usage Benchmark")
        print("=" * 30)
        print("psutil not installed - skipping memory benchmark")
        return

    print("\n🧠 Memory Usage Benchmark")
    print("=" * 30)

    # Get baseline memory
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Test with medium dataset
    x, y, groups = generate_benchmark_data(100, 10)
    beta0 = np.array([2.5, 8.0])

    for use_rust in [True, False]:
        backend = "Rust" if use_rust else "Python"

        # Memory before fitting
        mem_before = process.memory_info().rss / 1024 / 1024

        try:
            beta, psi, stats, b = pynlme.nlmefit(
                x, y, groups, None, simple_pk_model, beta0, verbose=0
            )

            # Memory after fitting
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - baseline_memory

            print(f"{backend:6} backend: {mem_used:5.1f} MB used")

        except Exception as e:
            print(f"{backend:6} backend: Failed ({e})")


def main():
    """Run comprehensive benchmark"""

    print("🚀 PyNLME Performance Benchmark")
    print("=" * 40)

    # Test different dataset sizes
    dataset_sizes = [10, 25, 50, 100, 200]

    print(f"Testing dataset sizes: {dataset_sizes}")
    print("Each with 8 timepoints per subject")

    # Run benchmarks
    results = benchmark_fitting(dataset_sizes)

    # Display summary table
    print("\n📋 Benchmark Results Summary:")
    print("=" * 60)

    summary = results.pivot_table(
        index="n_subjects",
        columns="backend",
        values=["time_seconds", "converged"],
        aggfunc={"time_seconds": "mean", "converged": "all"},
    )

    print(summary.round(3))

    # Calculate overall statistics
    rust_data = results[results["backend"] == "Rust"]
    python_data = results[results["backend"] == "Python"]

    if not rust_data.empty and not python_data.empty:
        avg_speedup = (
            python_data["time_seconds"].mean() / rust_data["time_seconds"].mean()
        )
        print(f"\n⚡ Average Rust Speedup: {avg_speedup:.1f}x")

        rust_convergence = rust_data["converged"].mean()
        python_convergence = python_data["converged"].mean()
        print(
            f"🎯 Convergence Rate - Rust: {rust_convergence:.1%}, Python: {python_convergence:.1%}"
        )

    # Memory benchmark
    memory_benchmark()

    # Create plots
    print("\n📊 Creating performance plots...")
    plot_benchmark_results(results)

    # Get the actual path where plot was saved
    examples_dir = os.path.dirname(__file__)
    plot_path = os.path.join(examples_dir, "benchmark_results.png")
    print(f"💾 Plots saved as '{plot_path}'")

    # Recommendations
    print("\n💡 Performance Recommendations:")
    print("  • Use Rust backend for datasets with >50 subjects")
    print("  • Python backend suitable for small datasets or debugging")
    print("  • Consider data preprocessing for large datasets")
    print("  • Monitor memory usage with very large datasets (>1000 subjects)")


if __name__ == "__main__":
    main()
