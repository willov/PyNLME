# Troubleshooting Guide

This guide helps resolve common issues when using PyNLME.

## ⚠️ Known Issues

### Rust Backend Optimization Bug (Current)

**Issue**: The Rust backend currently has an optimization bug that causes poor parameter estimation.

**Symptoms**:
- Parameters converge to incorrect values
- Poor model fits even with good data
- Log-likelihood values much worse than expected

**Current Fix**: 
PyNLME automatically uses the Python implementation instead of the Rust backend. You'll see this warning:
```
UserWarning: Rust backend temporarily disabled due to optimization bug. Using Python implementation.
```

**What to expect**:
- The Python implementation works correctly and gives good results
- Performance is slightly slower than the intended Rust backend
- All functionality remains available

**Action needed**: None - the package handles this automatically.

## Installation Issues

### Git Installation
```bash
# Install directly from GitHub (requires Rust toolchain)
pip install git+https://github.com/willov/PyNLME.git
```

### Rust Compilation Errors
If you encounter Rust compilation issues:

1. **Install Rust toolchain**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Update Rust** (if already installed):
   ```bash
   rustup update
   ```

3. **Clear cache** and reinstall:
   ```bash
   pip cache purge
   pip install --no-cache-dir git+https://github.com/willov/PyNLME.git
   ```

## Runtime Issues

### Convergence Problems

**Problem**: Model fails to converge
```python
NLMEError: Maximum iterations reached without convergence
```

**Solutions**:
1. **Adjust initial parameters**:
   ```python
   # Use better starting values
   beta0 = np.array([1.0, 0.5, 2.0])  # Domain-specific values
   result = nlmefit(model, beta0, x, y, groups)
   ```

2. **Increase iteration limits**:
   ```python
   options = {'max_iter': 1000, 'tol': 1e-4}
   result = nlmefit(model, beta0, x, y, groups, options=options)
   ```

3. **Try different algorithms**:
   ```python
   # Force Python backend for debugging
   result = nlmefit(model, beta0, x, y, groups, use_rust=False)
   ```

### Memory Issues

**Problem**: Out of memory errors with large datasets

**Solutions**:
1. **Process in chunks**:
   ```python
   # For datasets > 10,000 observations
   n_chunks = len(groups) // 1000
   # Process subsets and combine results
   ```

2. **Use data types efficiently**:
   ```python
   # Use float32 instead of float64 for large arrays
   x = x.astype(np.float32)
   y = y.astype(np.float32)
   ```

### Performance Issues

**Problem**: Slow fitting times

**Diagnostics**:
```python
import time
start_time = time.time()
result = nlmefit(model, beta0, x, y, groups)
print(f"Fitting took: {time.time() - start_time:.2f} seconds")
```

**Solutions**:
1. **Ensure Rust backend is used**:
   ```python
   # Check if Rust backend is available
   import pynlme
   print(f"Rust backend available: {pynlme.has_rust_backend()}")
   ```

2. **Optimize model function**:
   ```python
   # Vectorize operations
   def fast_model(x, beta):
       return beta[0] * np.exp(-beta[1] * x[:, 0]) + beta[2]
   ```

## Data Issues

### Input Validation Errors

**Problem**: Data format errors
```python
ValueError: Groups must be 1D array of integers
```

**Solutions**:
```python
# Ensure proper data types
groups = groups.astype(int)
x = np.asarray(x, dtype=float)
y = np.asarray(y, dtype=float)

# Check dimensions
assert x.ndim == 2, "x must be 2D array"
assert y.ndim == 1, "y must be 1D array"
assert len(y) == x.shape[0], "x and y must have same number of observations"
```

### Missing Data

**Problem**: NaN values in dataset

**Solutions**:
```python
# Remove missing observations
valid_idx = ~(np.isnan(y) | np.isnan(x).any(axis=1))
x_clean = x[valid_idx]
y_clean = y[valid_idx]
groups_clean = groups[valid_idx]
```

### Unbalanced Groups

**Problem**: Some groups have very few observations

**Diagnostics**:
```python
# Check group sizes
import pandas as pd
group_sizes = pd.Series(groups).value_counts()
print(f"Min group size: {group_sizes.min()}")
print(f"Groups with < 3 obs: {sum(group_sizes < 3)}")
```

**Solutions**:
```python
# Filter out small groups
min_size = 3
large_groups = group_sizes[group_sizes >= min_size].index
mask = np.isin(groups, large_groups)
x_filtered = x[mask]
y_filtered = y[mask]
groups_filtered = groups[mask]
```

## Model Specification Issues

### Ill-conditioned Problems

**Problem**: Singular matrix errors
```python
LinAlgError: Singular matrix
```

**Solutions**:
1. **Check parameter identifiability**:
   ```python
   # Ensure parameters are identifiable from data
   # Avoid overparameterization
   ```

2. **Add regularization**:
   ```python
   # Use parameter bounds
   bounds = [(0.1, 10), (0.01, 1), (0, 100)]
   result = nlmefit(model, beta0, x, y, groups, bounds=bounds)
   ```

### Poor Model Fit

**Diagnostics**:
```python
# Check residuals
residuals = result.residuals
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.scatter(result.fitted_values, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

plt.subplot(132)
plt.hist(residuals, bins=20)
plt.xlabel('Residuals')
plt.title('Residual Distribution')

plt.subplot(133)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.tight_layout()
plt.show()
```

## Environment Issues

### Python Version Compatibility

**Problem**: Package not working with Python version

**Check compatibility**:
```python
import sys
print(f"Python version: {sys.version}")
# PyNLME requires Python ≥3.10
```

### Dependency Conflicts

**Problem**: NumPy/SciPy version conflicts

**Solutions**:
```bash
# Check versions
pip list | grep -E "(numpy|scipy)"

# Upgrade dependencies
pip install --upgrade numpy scipy
```

## Getting Help

### Enable Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed error messages
result = nlmefit(model, beta0, x, y, groups)
```

### Report Issues
When reporting issues, please include:

1. **Python and package versions**:
   ```python
   import pynlme
   print(f"PyNLME version: {pynlme.__version__}")
   ```

2. **Minimal reproducible example**
3. **Full error traceback**
4. **System information** (OS, architecture)

### Performance Profiling
```python
import cProfile
import pstats

# Profile the fitting process
profiler = cProfile.Profile()
profiler.enable()
result = nlmefit(model, beta0, x, y, groups)
profiler.disable()

# Show results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

This should help resolve most common issues you might encounter with PyNLME!
