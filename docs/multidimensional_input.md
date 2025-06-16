# Multi-Dimensional Input Format

PyNLME now supports both traditional "stacked" data format (compatible with MATLAB's nlmefit) and a new "grouped" multi-dimensional format that makes working with structured data more intuitive.

## Overview

In the traditional MATLAB-style approach, all measurements must be arranged in a 1D stacked format where each row represents a single observation, and a separate group vector indicates which subject/group each observation belongs to.

With the new multi-dimensional format, you can provide data where each row represents a subject/group, making it much easier to work with structured datasets.

## Format Comparison

### Traditional Stacked Format

For 3 subjects with 4 measurements each:

```python
# X predictor data - all measurements stacked
X = np.array([[1], [2], [3], [4],  # Subject 1
              [1], [2], [3], [4],  # Subject 2  
              [1], [2], [3], [4]]) # Subject 3

# y response data - all measurements stacked  
y = np.array([10, 7, 5, 3,   # Subject 1 responses
              12, 8, 6, 4,   # Subject 2 responses
              11, 8, 6, 3])  # Subject 3 responses

# group indicators - must specify which subject each measurement belongs to
group = np.array([0, 0, 0, 0,  # Subject 1
                  1, 1, 1, 1,  # Subject 2
                  2, 2, 2, 2]) # Subject 3
```

### New Multi-Dimensional Format

The same data in the new format:

```python
# X predictor data - each row is a subject
X_grouped = np.array([
    [1, 2, 3, 4],  # Subject 1 measurements
    [1, 2, 3, 4],  # Subject 2 measurements
    [1, 2, 3, 4],  # Subject 3 measurements
])

# y response data - each row is a subject
y_grouped = np.array([
    [10, 7, 5, 3],  # Subject 1 responses
    [12, 8, 6, 4],  # Subject 2 responses
    [11, 8, 6, 3],  # Subject 3 responses
])

# No group parameter needed!
```

## Usage

### Automatic Detection

PyNLME automatically detects the input format and converts it internally:

```python
from pynlme import nlmefit

def exponential_model(phi, x, v=None):
    return phi[0] * np.exp(-phi[1] * x.ravel())

# Both approaches work identically
beta0 = np.array([10.0, 0.5])

# Traditional stacked format
beta1, psi1, stats1, b1 = nlmefit(X, y, group, None, exponential_model, beta0)

# New multi-dimensional format  
beta2, psi2, stats2, b2 = nlmefit(X_grouped, y_grouped, None, None, exponential_model, beta0)

# Results are identical
assert np.allclose(beta1, beta2)
```

### Manual Conversion

You can also manually convert between formats using utility functions:

```python
from pynlme import stack_grouped_data, detect_data_format

# Convert grouped to stacked format
X_stacked, y_stacked, group_stacked = stack_grouped_data(X_grouped, y_grouped)

# Detect data format
format_type = detect_data_format(X, y, group)  # Returns 'stacked' or 'grouped'
```

### Custom Group Identifiers

You can specify custom group identifiers when converting:

```python
# Use custom group labels
group_ids = ['Patient_A', 'Patient_B', 'Patient_C']
X_stacked, y_stacked, group_stacked = stack_grouped_data(
    X_grouped, y_grouped, group_ids
)
```

## Multiple Predictor Variables

The multi-dimensional format also supports multiple predictor variables:

```python
# 3 subjects, 4 time points, 2 predictors (time and dose)
X_multi = np.array([
    [[1, 100], [2, 100], [3, 100], [4, 100]],  # Subject 1: time, dose
    [[1, 150], [2, 150], [3, 150], [4, 150]],  # Subject 2: time, dose  
    [[1, 200], [2, 200], [3, 200], [4, 200]],  # Subject 3: time, dose
])

y_multi = np.array([
    [50, 40, 30, 20],  # Subject 1 responses
    [70, 55, 40, 25],  # Subject 2 responses
    [90, 70, 50, 30],  # Subject 3 responses
])

# Works seamlessly with nlmefit
beta, psi, stats, b = nlmefit(X_multi, y_multi, None, None, model_func, beta0)
```

## Benefits

1. **More Intuitive**: Data organized by subject/group is easier to understand
2. **Less Error-Prone**: No need to manually create group vectors
3. **Cleaner Code**: Eliminates manual stacking operations
4. **Backwards Compatible**: Existing stacked format still works
5. **Flexible**: Supports both single and multiple predictor variables

## Examples

See `examples/multidimensional_input_demo.py` for a complete working example and `tests/test_multidimensional_input.py` for comprehensive test cases.

## API Reference

### `stack_grouped_data(X_grouped, y_grouped, group_ids=None)`

Convert multi-dimensional grouped data to stacked format.

**Parameters:**
- `X_grouped`: array_like, shape (n_groups, n_obs_per_group, n_features) or (n_groups, n_obs_per_group)
- `y_grouped`: array_like, shape (n_groups, n_obs_per_group)  
- `group_ids`: array_like, optional - Custom group identifiers

**Returns:**
- `X_stacked`: ndarray, shape (n_groups * n_obs_per_group, n_features)
- `y_stacked`: ndarray, shape (n_groups * n_obs_per_group,)
- `group_stacked`: ndarray, shape (n_groups * n_obs_per_group,)

### `detect_data_format(X, y, group=None)`

Detect whether data is in stacked or grouped format.

**Parameters:**
- `X`: array_like - Predictor variables
- `y`: array_like - Response variables  
- `group`: array_like, optional - Group identifiers

**Returns:**
- `format`: str - Either 'stacked' or 'grouped'
