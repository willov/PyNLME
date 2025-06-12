# PyNLME - Nonlinear Mixed-Effects Models for Python

A high-performance Python implementation of nonlinear mixed-effects (NLME) models with a MATLAB-compatible API and Rust backend.

## 🧪 Status: **FUNCTIONAL - PROOF OF CONCEPT** ⚠️

✅ **48/48 tests passing**  
✅ **Rust backend integrated and working**  
✅ **MATLAB-compatible API**  
⚠️ **Requires validation for production use**

## 🚀 Quick Start

```python
import numpy as np
from pynlme import nlmefit

# Define your nonlinear model
def exponential_decay(phi, t, v=None):
    return phi[0] * np.exp(-phi[1] * t.ravel())

# Prepare data
t = np.array([[1], [2], [3], [4]])  # Time points
y = np.array([10, 7, 5, 3])         # Observations
group = np.array([0, 0, 1, 1])      # Subject grouping
beta0 = np.array([10.0, 0.5])       # Initial estimates

# Fit model with Rust backend (automatic)
beta, psi, stats, b = nlmefit(t, y, group, None, exponential_decay, beta0)

print(f"Fixed effects: {beta}")
print(f"Log-likelihood: {stats.logl}")
print(f"AIC: {stats.aic}")
```

## 📦 Installation

```bash
# Install from source (recommended for development)
git clone https://github.com/willov/PyNLME
cd PyNLME
uv sync
uv run maturin develop

# For end users - install from releases (when available)
# Download wheel from https://github.com/willov/PyNLME/releases
# and install with: uv pip install downloaded_wheel.whl
```

## 🔧 Core Features

### **Algorithms**
- **`nlmefit()`** - Maximum Likelihood Estimation (MLE) 
- **`nlmefitsa()`** - Stochastic Approximation EM (SAEM)
- **Rust backend** - High-performance optimization (automatic fallback to Python)

### **MATLAB Compatibility**
- Same function names and parameter conventions
- Compatible return types: `(beta, psi, stats, b)`
- Similar options and configuration parameters

### **Robust Type System**
- `NLMEStats` - Statistics and diagnostics
- `NLMEOptions` - General fitting options  
- `SAEMOptions` - SAEM-specific parameters
- `ErrorModel` - Different error model types

### **Advanced Features**
- Multiple error models (constant, proportional, combined, exponential)
- Parameter transformations (identity, log, probit, logit)
- Flexible covariance structures
- Comprehensive residual diagnostics

## 🎯 Use Cases

**Pharmacokinetics/Pharmacodynamics**
```python
# PK model: Concentration = Dose * exp(-ke * t) / V
def pk_model(phi, t, dose):
    ke, V = phi
    return dose * np.exp(-ke * t) / V

beta, psi, stats, b = nlmefit(time, concentration, subject_id, dose, pk_model, [0.1, 50])
```

**Growth Models**
```python  
# Logistic growth: y = A / (1 + exp(-(t - t0)/tau))
def logistic_growth(phi, t, v=None):
    A, t0, tau = phi
    return A / (1 + np.exp(-(t - t0) / tau))

beta, psi, stats, b = nlmefitsa(time, size, organism_id, None, logistic_growth, [100, 10, 2])
```

**Dose-Response**
```python
# Sigmoid: y = Emax * dose^n / (ED50^n + dose^n)  
def sigmoid_model(phi, dose, v=None):
    Emax, ED50, n = phi
    return Emax * dose**n / (ED50**n + dose**n)

beta, psi, stats, b = nlmefit(dose, response, subject_id, None, sigmoid_model, [100, 10, 1])
```

## 📊 Performance

The Rust backend provides significant performance improvements:

- **Fast convergence**: Optimized numerical algorithms
- **Memory efficient**: Minimal memory footprint
- **Parallel ready**: Multi-threaded optimization (planned)
- **Automatic fallback**: Uses Python implementation if Rust fails

Example convergence output:
```
Iteration 1: logl = -12.454, beta_change = 9.041
Iteration 2: logl = -11.816, beta_change = 0.937
...
Converged after 22 iterations
```

## 🔍 Model Diagnostics

```python
# Comprehensive model statistics
print(f"Degrees of freedom: {stats.dfe}")
print(f"Log-likelihood: {stats.logl}")
print(f"AIC: {stats.aic}")  
print(f"BIC: {stats.bic}")
print(f"RMSE: {stats.rmse}")

# Residual analysis
residuals = stats.ires  # Individual residuals
pred_res = stats.pres   # Population residuals  
wt_res = stats.iwres    # Individual weighted residuals
```

## 🧪 Testing

```bash
# Run full test suite
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/test_algorithms.py -v  # Algorithm tests
uv run pytest tests/test_basic.py -v       # Integration tests
uv run pytest tests/test_data_types.py -v  # Type system tests
```

## 📚 Documentation

- **[API Reference](api_reference.md)** - Complete function documentation
- **[Tutorial](tutorial.md)** - Step-by-step learning guide
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Examples](../examples/)** - Usage examples and tutorials
- **[Implementation Details](implementation.md)** - Technical details
- **[Changelog](../CHANGELOG.md)** - Version history and release notes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `uv run pytest`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🏗️ Architecture

```
PyNLME Package
├── Python API Layer     # MATLAB-compatible interface
├── Algorithm Layer      # MLE/SAEM implementations  
├── Rust Backend        # High-performance optimization
└── Utilities           # Validation, diagnostics, etc.
```

The package is designed for:
- **Ease of use**: MATLAB-like interface
- **Performance**: Rust backend for speed
- **Reliability**: Comprehensive testing and error handling
- **Extensibility**: Modular architecture for new features

## Function Interface

PyNLME provides a clean, simple interface with just 3 essential functions:

### MATLAB-Compatible Functions

- **`nlmefit()`** - Maximum Likelihood Estimation (identical to MATLAB)
- **`nlmefitsa()`** - Stochastic Approximation EM (identical to MATLAB)

### Python-Style Unified Interface

- **`fit_nlme(method='ML'|'SAEM')`** - Unified interface with method parameter

This simplified design makes PyNLME easy to learn and use, whether you're coming from MATLAB or prefer Python-style APIs. All functions provide access to the same underlying algorithms and produce identical results for the same method.
