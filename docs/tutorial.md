# Tutorial: Getting Started with PyNLME

This tutorial provides a step-by-step introduction to nonlinear mixed-effects modeling using PyNLME.

## PyNLME Function Interface

PyNLME is designed with simplicity in mind. We provide **just 3 essential functions**:

### MATLAB Users
- **`nlmefit()`** - Maximum Likelihood Estimation (identical to MATLAB)
- **`nlmefitsa()`** - Stochastic Approximation EM (identical to MATLAB)

### Python Users
- **`fit_nlme(method='ML'|'SAEM')`** - Unified interface with method parameter

This clean design makes PyNLME easy to learn and use, regardless of your background. All functions provide access to the same underlying algorithms and produce identical results for the same method.

## What are Nonlinear Mixed-Effects Models?

Nonlinear mixed-effects (NLME) models are used to analyze data where:
- The relationship between variables is **nonlinear**
- Data comes from **multiple groups** (subjects, batches, etc.)
- There's both **population-level** and **individual-specific** variation

Common applications include:
- **Pharmacokinetics**: Drug concentration over time
- **Growth curves**: Height/weight development
- **Dose-response**: Biological responses to treatments

## Basic Concepts

### Model Structure
An NLME model has the form:

```
y_ij = f(x_ij, β_i) + ε_ij
β_i = β + b_i
```

Where:
- `y_ij`: Observation j from individual i
- `f()`: Nonlinear function
- `β_i`: Individual-specific parameters
- `β`: Population (fixed) effects
- `b_i`: Random effects for individual i
- `ε_ij`: Residual error

### Key Components
1. **Fixed Effects (β)**: Population-average parameters
2. **Random Effects (b_i)**: Individual deviations from population
3. **Error Model**: Describes residual variation

## Tutorial 1: One-Compartment Pharmacokinetic Model

Let's fit a simple PK model: `C(t) = Dose/V * exp(-CL/V * t)`

### Step 1: Import and Setup
```python
import numpy as np
import matplotlib.pyplot as plt
import pynlme

# Set random seed for reproducibility
np.random.seed(42)
```

### Step 2: Generate Synthetic Data
```python
def generate_pk_data(n_subjects=20, n_timepoints=8):
    """Generate synthetic PK data"""
    
    # Population parameters
    pop_cl = 2.0    # Clearance (L/h)
    pop_v = 10.0    # Volume (L)
    
    # Between-subject variability (CV%)
    cv_cl = 0.3
    cv_v = 0.2
    
    # Residual error (proportional)
    prop_error = 0.1
    
    # Time points
    times = np.array([0.5, 1, 2, 4, 8, 12, 24, 48])
    
    data = []
    for subject in range(n_subjects):
        # Individual parameters (log-normal distribution)
        cl_i = pop_cl * np.exp(np.random.normal(0, cv_cl))
        v_i = pop_v * np.exp(np.random.normal(0, cv_v))
        
        # Dose (assume 100 mg)
        dose = 100
        
        for time in times:
            # True concentration
            conc_true = (dose / v_i) * np.exp(-cl_i / v_i * time)
            
            # Add proportional error
            conc_obs = conc_true * (1 + np.random.normal(0, prop_error))
            
            data.append({
                'subject': subject,
                'time': time,
                'dose': dose,
                'concentration': conc_obs,
                'cl_true': cl_i,
                'v_true': v_i
            })
    
    return pd.DataFrame(data)

# Generate data
import pandas as pd
df = generate_pk_data(n_subjects=20, n_timepoints=8)
print(f"Generated {len(df)} observations from {df['subject'].nunique()} subjects")
```

### Step 3: Prepare Data for PyNLME
```python
# Extract arrays for PyNLME
x = df[['time', 'dose']].values  # Covariates (time, dose)
y = df['concentration'].values   # Observations
groups = df['subject'].values    # Subject IDs

print(f"Data shape: x={x.shape}, y={y.shape}, groups={groups.shape}")
```

### Step 4: Define the Model Function
```python
def pk_model(x, beta):
    """
    One-compartment PK model
    x[:, 0] = time
    x[:, 1] = dose
    beta[0] = CL (clearance)
    beta[1] = V (volume)
    """
    time = x[:, 0]
    dose = x[:, 1]
    cl, v = beta[0], beta[1]
    
    # Avoid division by zero and negative parameters
    cl = np.maximum(cl, 1e-6)
    v = np.maximum(v, 1e-6)
    
    # One-compartment model: C = (Dose/V) * exp(-CL/V * t)
    conc = (dose / v) * np.exp(-cl / v * time)
    return conc
```

### Step 5: Set Initial Parameter Estimates
```python
# Initial guesses for population parameters
beta0 = np.array([
    2.5,   # CL initial guess
    8.0    # V initial guess
])

print(f"Initial parameters: CL={beta0[0]}, V={beta0[1]}")
```

### Step 6: Fit the Model

PyNLME provides multiple function interfaces. Choose the one that matches your background:

#### Option A: MATLAB-style interface
```python
# Fit using MATLAB-compatible function
print("Fitting NLME model using nlmefit()...")
beta, psi, stats, b = pynlme.nlmefit(
    X=x,
    y=y,
    group=groups,
    V=None,
    modelfun=pk_model,
    beta0=beta0,
    max_iter=100,
    verbose=1
)

print(f"Convergence: {'Yes' if stats.converged else 'No'}")
print(f"Final log-likelihood: {stats.logl:.2f}")
```

#### Option B: Python-style unified interface
```python
# Fit using Python-style unified function
print("Fitting NLME model using fit_nlme()...")
beta, psi, stats, b = pynlme.fit_nlme(
    X=x,
    y=y,
    group=groups,
    V=None,
    modelfun=pk_model,
    beta0=beta0,
    method="ML",  # or "SAEM" for stochastic EM
    max_iter=100,
    verbose=1
)

print(f"Convergence: {'Yes' if stats.converged else 'No'}")
print(f"Final log-likelihood: {stats.logl:.2f}")
```

### Step 7: Examine Results
```python
# Population parameters
print("\nPopulation Parameters:")
print(f"CL = {beta[0]:.2f} L/h (true: 2.0)")
print(f"V  = {beta[1]:.2f} L   (true: 10.0)")

# Random effects variance
print(f"\nRandom Effects Covariance:")
print(psi)

# Model fit statistics
print(f"\nModel Fit:")
print(f"AIC: {stats.aic:.1f}")
print(f"BIC: {stats.bic:.1f}")
print(f"RMSE: {stats.rmse:.3f}")
```

### Step 8: Visualize Results
```python
# Plot population fit
plt.figure(figsize=(15, 10))

# Individual profiles
plt.subplot(2, 3, 1)
for subject in df['subject'].unique()[:6]:  # Show first 6 subjects
    subj_data = df[df['subject'] == subject]
    plt.plot(subj_data['time'], subj_data['concentration'], 'o-', alpha=0.7, 
             label=f'Subject {subject}')

plt.xlabel('Time (h)')
plt.ylabel('Concentration (mg/L)')
plt.title('Individual Concentration Profiles')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Population prediction
plt.subplot(2, 3, 2)
time_pred = np.linspace(0.5, 48, 100)
x_pred = np.column_stack([time_pred, np.full_like(time_pred, 100)])
pop_pred = pk_model(x_pred, result.beta)
plt.plot(time_pred, pop_pred, 'r-', linewidth=2, label='Population')
plt.scatter(df['time'], df['concentration'], alpha=0.5, label='Observed')
plt.xlabel('Time (h)')
plt.ylabel('Concentration (mg/L)')
plt.title('Population Fit')
plt.legend()

# Residuals vs fitted
plt.subplot(2, 3, 3)
fitted = pk_model(x, result.beta)
residuals = y - fitted
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# Residual distribution
plt.subplot(2, 3, 4)
plt.hist(residuals, bins=20, density=True, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Residual Distribution')

# Q-Q plot
plt.subplot(2, 3, 5)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

# Parameter correlation
plt.subplot(2, 3, 6)
if hasattr(result, 'random_effects') and result.random_effects is not None:
    re = result.random_effects
    plt.scatter(re[:, 0], re[:, 1], alpha=0.7)
    plt.xlabel('CL Random Effect')
    plt.ylabel('V Random Effect')
    plt.title('Random Effects Correlation')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

## Tutorial 2: Comparing True vs Estimated Parameters

```python
# Extract true and estimated individual parameters
true_params = df.groupby('subject')[['cl_true', 'v_true']].first()
est_params = result.beta + result.random_effects  # Population + individual effects

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CL comparison
axes[0].scatter(true_params['cl_true'], est_params[:, 0], alpha=0.7)
axes[0].plot([0, 5], [0, 5], 'r--', label='Perfect agreement')
axes[0].set_xlabel('True CL')
axes[0].set_ylabel('Estimated CL')
axes[0].set_title('Clearance')
axes[0].legend()

# V comparison  
axes[1].scatter(true_params['v_true'], est_params[:, 1], alpha=0.7)
axes[1].plot([5, 15], [5, 15], 'r--', label='Perfect agreement')
axes[1].set_xlabel('True V')
axes[1].set_ylabel('Estimated V')
axes[1].set_title('Volume')
axes[1].legend()

plt.tight_layout()
plt.show()

# Calculate prediction accuracy
cl_corr = np.corrcoef(true_params['cl_true'], est_params[:, 0])[0, 1]
v_corr = np.corrcoef(true_params['v_true'], est_params[:, 1])[0, 1]

print(f"Parameter Recovery:")
print(f"CL correlation: {cl_corr:.3f}")
print(f"V correlation:  {v_corr:.3f}")
```

## Key Takeaways

1. **Data Preparation**: Ensure proper array shapes and data types
2. **Model Function**: Define vectorized functions that handle parameters safely
3. **Initial Values**: Good starting values improve convergence
4. **Diagnostics**: Always check residuals and model fit
5. **Validation**: Compare results with known truth when possible

## Next Steps

- **Tutorial 2**: Multi-compartment models
- **Tutorial 3**: Covariate effects
- **Tutorial 4**: Different error models
- **Tutorial 5**: Model selection and validation

Ready to explore more complex models? Check out the advanced examples!
