# PyNLME Examples

This directory contains practical examples demonstrating PyNLME functionality.

## 📋 Dependencies

All examples require the basic PyNLME installation. Some examples have additional requirements:

**Basic Requirements:**
- `numpy` (included with PyNLME)
- `scipy` (included with PyNLME)

**Optional Requirements for Enhanced Examples:**
- `matplotlib` - For plotting and visualization (all examples)
- `pandas` - For performance_benchmark.py data analysis
- `psutil` - For memory usage monitoring in performance_benchmark.py

Install optional dependencies:
```bash
# For plotting
uv add matplotlib

# For performance benchmarks  
uv add pandas psutil

# Or install all optional dependencies
uv add matplotlib pandas psutil
```

## 📋 Example Files

### [`basic_usage.py`](basic_usage.py)
Simple examples to get started with PyNLME:
- Exponential decay model
- Basic MLE and SAEM fitting
- Essential statistics and diagnostics

### [`multidimensional_input_demo.py`](multidimensional_input_demo.py)
**New!** Demonstrates the multi-dimensional input format feature:
- Traditional vs. new grouped data formats
- Pharmacokinetic modeling example
- Multiple predictor variables
- Automatic format detection and conversion

### [`grouped_vs_stacked_comparison.py`](grouped_vs_stacked_comparison.py)
**New!** Compares grouped vs stacked data formats using a concrete example
(3 subjects with 4 measurements each):
- Traditional stacked vs. new grouped format comparison
- Data format equivalence verification  
- Cleaner code examples

### [`advanced_usage.py`](advanced_usage.py)  
Complex modeling scenarios:
- Multiple error models
- Parameter transformations
- Multi-group analysis
- Custom convergence criteria

### [`matlab_comparison.py`](matlab_comparison.py)
MATLAB compatibility demonstrations:
- Side-by-side MATLAB vs PyNLME code
- Result validation and comparison
- Migration guides for MATLAB users

## 🚀 Running Examples

```bash
# Run all examples
cd examples/
python basic_usage.py
python advanced_usage.py
python matlab_comparison.py

# Or run from project root
uv run python examples/basic_usage.py
```

## 📊 Example Models

The examples demonstrate various nonlinear models commonly used in practice:

- **Exponential Decay**: `y = A * exp(-k * t)`
- **Michaelis-Menten**: `y = Vmax * x / (Km + x)`
- **Logistic Growth**: `y = A / (1 + exp(-(t - t0)/tau))`
- **Sigmoid Dose-Response**: `y = Emax * dose^n / (ED50^n + dose^n)`
- **Pharmacokinetic Models**: Various PK/PD models

## 🎯 Use Cases Covered

- **Pharmacokinetics**: Drug concentration over time
- **Pharmacodynamics**: Dose-response relationships  
- **Growth Models**: Population and individual growth
- **Bioassays**: Dose-response curves
- **Time Series**: Nonlinear time-dependent processes

## 📝 Example Template

Use this template for your own models:

```python
import numpy as np
from pynlme import nlmefit

def your_model(phi, X, V=None):
    """
    Your nonlinear model function.
    
    Parameters:
    -----------
    phi : ndarray
        Parameter vector [param1, param2, ...]
    X : ndarray
        Predictor variables
    V : ndarray, optional
        Group-level covariates
        
    Returns:
    --------
    ndarray
        Predicted values
    """
    # Your model equation here
    return predictions

# Your data preparation
X = your_predictors
y = your_responses  
group = your_grouping
beta0 = your_initial_estimates

# Fit the model
beta, psi, stats, b = nlmefit(X, y, group, None, your_model, beta0)

# Analyze results
print(f"Fixed effects: {beta}")
print(f"AIC: {stats.aic}")
# Add your analysis here
```

## 🔧 Customization Examples

Each example file includes variations showing:
- Different optimization options
- Error model specifications
- Convergence criteria
- Diagnostic plotting
- Result interpretation

## 📚 Additional Resources

- [API Reference](../docs/api_reference.md) - Complete function documentation
- [Implementation Details](../docs/implementation.md) - Technical details
- [Main Documentation](../docs/README.md) - Full feature overview

## 📂 Advanced Examples

In addition to the basic and advanced usage examples, this repository includes:

- **[`matlab_comparison.py`](matlab_comparison.py)** - MATLAB compatibility demonstration
  - Side-by-side comparison with MATLAB nlmefit
  - Parameter estimation accuracy verification
  - API compatibility examples

### Real-world Case Studies

- **[`warfarin_case_study.py`](warfarin_case_study.py)** - Realistic pharmacokinetic analysis
  - Multi-dose warfarin concentration modeling
  - Covariate effects (age, weight, gender)
  - Comprehensive model diagnostics and validation

- **[`performance_benchmark.py`](performance_benchmark.py)** - Performance comparison
  - Rust vs Python backend benchmarking
  - Scalability analysis across dataset sizes
  - Memory usage profiling
