# PyNLME API Reference

## Core Functions

### `nlmefit(X, y, group, V, modelfun, beta0, **kwargs)`

Maximum Likelihood Estimation for nonlinear mixed-effects models.

**Parameters:**
- `X` (ndarray): Predictor variables, shape `(n_obs, n_predictors)`
- `y` (ndarray): Response variable, shape `(n_obs,)`
- `group` (ndarray): Grouping variable (0-based indices), shape `(n_obs,)`
- `V` (ndarray, optional): Group-level predictors, shape `(n_groups, n_covariates)`
- `modelfun` (callable): Model function with signature `f(phi, X, V=None)`
- `beta0` (ndarray): Initial parameter estimates, shape `(n_params,)`

**Keyword Arguments:**
- `max_iter` (int): Maximum iterations (default: 200)
- `tol_fun` (float): Function tolerance (default: 1e-6)
- `tol_x` (float): Parameter tolerance (default: 1e-6)
- `verbose` (int): Verbosity level (default: 0)
- `approximation_type` (str): "LME", "RELME", "FO", "FOCE" (default: "LME")
- `error_model` (str): "constant", "proportional", "combined", "exponential"
- `compute_std_errors` (bool): Whether to compute standard errors (default: True)

**Returns:**
- `beta` (ndarray): Fixed-effects estimates
- `psi` (ndarray): Random-effects covariance matrix  
- `stats` (NLMEStats): Fitting statistics and diagnostics
- `b` (ndarray): Individual random effects estimates

**Example:**
```python
def exponential_model(phi, x, v=None):
    return phi[0] * np.exp(-phi[1] * x.ravel())

beta, psi, stats, b = nlmefit(X, y, group, None, exponential_model, [10, 0.5])
```

---

### `nlmefitsa(X, y, group, V, modelfun, beta0, **kwargs)`

Stochastic Approximation EM for nonlinear mixed-effects models.

**Parameters:**
Same as `nlmefit()`.

**Additional Keyword Arguments:**
- `n_iterations` (tuple): SAEM phase iterations (burn-in, stochastic, smooth)
- `n_mcmc_iterations` (tuple): MCMC iterations per phase
- `n_burn_in` (int): Additional burn-in iterations
- `tol_sa` (float): Stochastic approximation tolerance (default: 1e-4)
- `step_size_sequence` (str): "auto" or custom sequence

**Returns:**
Same as `nlmefit()`.

**Example:**
```python
beta, psi, stats, b = nlmefitsa(X, y, group, None, exponential_model, [10, 0.5],
                                n_iterations=(100, 100, 50), verbose=1)
```

---

### `fit_nlme(X, y, group, V, modelfun, beta0, method="ML", **kwargs)`

Unified Python-style interface for nonlinear mixed-effects models.

**Parameters:**
- `X` (ndarray): Predictor variables, shape `(n_obs, n_predictors)`
- `y` (ndarray): Response variable, shape `(n_obs,)`
- `group` (ndarray): Grouping variable (0-based indices), shape `(n_obs,)`
- `V` (ndarray, optional): Group-level predictors, shape `(n_groups, n_covariates)`
- `modelfun` (callable): Model function with signature `f(phi, X, V=None)`
- `beta0` (ndarray): Initial parameter estimates, shape `(n_params,)`
- `method` (str): Fitting method - "ML" or "SAEM" (default: "ML")

**Keyword Arguments:**
Same as `nlmefit()` and `nlmefitsa()` depending on the method selected.

**Returns:**
Same as `nlmefit()` and `nlmefitsa()`.

**Example:**
```python
# Maximum Likelihood Estimation
beta, psi, stats, b = fit_nlme(X, y, group, None, exponential_model, [10, 0.5], method="ML")

# Stochastic Approximation EM  
beta, psi, stats, b = fit_nlme(X, y, group, None, exponential_model, [10, 0.5], method="SAEM")
```

---

## Data Types

### `NLMEStats`

Container for model fitting statistics and diagnostics.

**Attributes:**
- `dfe` (int): Degrees of freedom for error
- `logl` (float): Log-likelihood value
- `aic` (float): Akaike Information Criterion
- `bic` (float): Bayesian Information Criterion  
- `rmse` (float): Root mean squared error
- `mse` (float): Mean squared error
- `sse` (float): Sum of squared errors
- `sebeta` (ndarray): Standard errors of fixed effects
- `covb` (ndarray): Covariance matrix of fixed effects
- `ires` (ndarray): Individual residuals
- `pres` (ndarray): Population residuals
- `iwres` (ndarray): Individual weighted residuals
- `pwres` (ndarray): Population weighted residuals
- `cwres` (ndarray): Conditional weighted residuals

---

### `NLMEOptions`

General options for NLME fitting.

**Attributes:**
- `approximation_type` (str): Approximation method
- `optim_fun` (str): Optimization algorithm
- `param_transform` (ndarray): Parameter transformations
- `error_model` (str): Error model specification
- `cov_pattern` (ndarray): Covariance pattern matrix
- `cov_parametrization` (str): "logm" or "chol"
- `vectorization` (str): "SinglePhi", "SingleGroup", "Full"
- `compute_std_errors` (bool): Compute standard errors
- `refine_beta0` (bool): Refine initial estimates
- `max_iter` (int): Maximum iterations
- `tol_fun` (float): Function tolerance
- `tol_x` (float): Parameter tolerance
- `verbose` (int): Verbosity level
- `random_state` (int): Random seed

---

### `SAEMOptions`

SAEM-specific algorithm options.

**Attributes:**
- `n_iterations` (tuple): Phase iterations (burn-in, stochastic, smooth)
- `n_mcmc_iterations` (tuple): MCMC iterations per phase
- `n_burn_in` (int): Additional burn-in
- `step_size_sequence` (str): Step size schedule
- `tol_ll` (float): Log-likelihood tolerance
- `tol_sa` (float): Stochastic approximation tolerance
- `verbose` (int): Verbosity level
- `random_state` (int): Random seed
- `algorithm` (str): Algorithm identifier

---

### `ErrorModel`

Error model specification and evaluation.

**Attributes:**
- `model_type` (str): "constant", "proportional", "combined", "exponential"
- `parameters` (ndarray): Error model parameters

**Methods:**
- `evaluate(y_pred, theta)`: Evaluate error variance given predictions

**Error Model Types:**
- **Constant**: `σ² = θ₁²`
- **Proportional**: `σ² = (θ₁ × |y_pred|)²`
- **Combined**: `σ² = (θ₁ + θ₂ × |y_pred|)²`
- **Exponential**: `σ² = exp(2 × θ₁ × log(|y_pred|))`

---

## Model Function Interface

Your model function should follow this signature:

```python
def your_model(phi, X, V=None):
    """
    Nonlinear model function.
    
    Parameters:
    -----------
    phi : ndarray
        Parameter vector for this group/individual
    X : ndarray  
        Predictor variables for this group
    V : ndarray, optional
        Group-level covariates
        
    Returns:
    --------
    ndarray
        Predicted values, same length as observations
    """
    # Your model implementation
    return predictions
```

**Examples:**

```python
# Exponential decay
def exponential_decay(phi, X, V=None):
    A, k = phi
    return A * np.exp(-k * X.ravel())

# Michaelis-Menten kinetics  
def michaelis_menten(phi, X, V=None):
    Vmax, Km = phi
    return Vmax * X.ravel() / (Km + X.ravel())

# Logistic growth
def logistic_growth(phi, X, V=None):
    A, t0, tau = phi
    t = X.ravel()
    return A / (1 + np.exp(-(t - t0) / tau))

# Sigmoid dose-response
def sigmoid_dose_response(phi, X, V=None):
    Emax, ED50, hill = phi
    dose = X.ravel()
    return Emax * dose**hill / (ED50**hill + dose**hill)
```

---

## Error Handling

The package provides comprehensive error handling:

**Common Exceptions:**
- `ValueError`: Invalid input dimensions or parameters
- `TypeError`: Incorrect parameter types
- `RuntimeError`: Optimization failures
- `ConvergenceWarning`: Algorithm didn't converge

**Example Error Handling:**
```python
try:
    beta, psi, stats, b = nlmefit(X, y, group, None, model, beta0)
except ValueError as e:
    print(f"Input validation error: {e}")
except RuntimeError as e:
    print(f"Optimization failed: {e}")
    # Perhaps try different initial values or SAEM
    beta, psi, stats, b = nlmefitsa(X, y, group, None, model, beta0)
```

---

## Advanced Usage

### Parameter Transformations

```python
# Log transformation for positive parameters
options = NLMEOptions(param_transform=np.array([1, 1]))  # Log transform both params
beta, psi, stats, b = nlmefit(X, y, group, None, model, beta0, **options.__dict__)
```

### Custom Error Models

```python
# Proportional error model
beta, psi, stats, b = nlmefit(X, y, group, None, model, beta0, 
                              error_model="proportional")
```

### Covariance Patterns

```python
# Diagonal covariance (no correlations)
cov_pattern = np.eye(2)  # 2x2 diagonal
options = NLMEOptions(cov_pattern=cov_pattern)
beta, psi, stats, b = nlmefit(X, y, group, None, model, beta0, **options.__dict__)
```
