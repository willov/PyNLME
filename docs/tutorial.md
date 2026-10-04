# Tutorial

A walk through the main features, for users coming from MATLAB. The code runs from the
repository root (the datasets are in `validation/datasets/`).

## 1. A MATLAB example, line by line

MATLAB's `nlmefit` documentation fits a bi-exponential model to the indomethacin data:

```matlab
load indomethacin
model = @(phi,t)(phi(1).*exp(-phi(2).*t)+phi(3).*exp(-phi(4).*t));
[beta,psi,stats,b] = nlmefit(time,concentration,subject,[],model,[1 1 1 1], ...
                             ParamTransform=[0 1 0 1]);
```

In pynlme:

```python
import numpy as np
from pynlme import nlmefit

data = np.genfromtxt("validation/datasets/indometh.csv", delimiter=",", names=True)
subject, time, concentration = data["subject"], data["time"], data["conc"]

def model(phi, t):
    return phi[0] * np.exp(-phi[1] * t) + phi[2] * np.exp(-phi[3] * t)

beta, psi, stats, b = nlmefit(time, concentration, subject, None, model, [1, 1, 1, 1],
                              ParamTransform=[0, 1, 0, 1])
print(beta)            # MATLAB: 0.4606 -1.3459 2.8277 0.7729 (possibly with the two terms swapped)
print(np.diag(psi))    # MATLAB: 0.0124 0.0000 0.3264 0.0250
print(stats.logl, stats.rmse, stats.sebeta)
```

`b` is r x m (one column per subject) and `psi` r x r, as in MATLAB. The start
`[1 1 1 1]` is symmetric in the two exponential terms, so which term ends up first is
decided by rounding; the fit is the same.

## 2. Group-level predictors, covariates and random-effects selection

Theophylline: an oral one-compartment model where the dose differs by subject (`V`), and
only absorption and clearance vary between subjects.

```python
data = np.genfromtxt("validation/datasets/theoph.csv", delimiter=",", names=True)
subject, time, conc = data["subject"], data["time"], data["conc"]
_, first = np.unique(subject, return_index=True)
dose = data["dose"][first]                      # one value per subject (V)

def oral(phi, t, dose):
    ke, ka, cl = phi
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

beta, psi, stats, b = nlmefit(time, conc, subject, dose, oral, [-2.5, 0.5, -3.2],
                              ParamTransform=[1, 1, 1],   # estimate log ke, log ka, log cl
                              REParamsSelect=[2, 3])      # random effects on ka and cl (1-based)
```

A weight effect on log clearance uses a group-specific fixed-effects design, as in
MATLAB:

```python
wt = data["wt"][first] - 70
A = np.zeros((3, 4, len(wt)))       # p x f x m
A[0, 0] = A[1, 1] = A[2, 2] = 1
A[2, 3] = wt                         # log cl = beta3 + beta4 * (wt - 70)
beta, psi, stats, b = nlmefit(time, conc, subject, dose, oral, [-2.5, 0.5, -3.2, 0.0],
                              ParamTransform=[1, 1, 1], REParamsSelect=[2, 3], FEGroupDesign=A)
```

## 3. Approximations, error models and SAEM

```python
fit = dict(ParamTransform=[1, 1, 1], REParamsSelect=[2, 3])
nlmefit(time, conc, subject, dose, oral, [-2.5, 0.5, -3.2], ApproximationType="FOCE", **fit)
nlmefit(time, conc, subject, dose, oral, [-2.5, 0.5, -3.2], ErrorModel="combined", **fit)

from pynlme import nlmefitsa
beta, psi, stats, b = nlmefitsa(time, conc, subject, dose, oral, [-2.5, 0.5, -3.2],
                                random_state=1, LogLikMethod="is", **fit)
```

SAEM is stochastic: set `random_state` for reproducible results. It maximises the exact
likelihood rather than a linearised one.

## 4. Faster fits: `Vectorization="Full"`

If the model can evaluate many parameter rows at once, pass `Vectorization="Full"`; the
model then receives one parameter row per observation. Index the last axis so the same
function also works for a single vector:

```python
def oral_v(phi, t, dose):
    ke, ka, cl = phi[..., 0], phi[..., 1], phi[..., 2]
    t, dose = t[:, 0], dose[..., 0]
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

nlmefit(time, conc, subject, dose, oral_v, [-2.5, 0.5, -3.2], Vectorization="Full", **fit)
```

## 5. Grouped data

With one row per subject, `group` can be left out:

```python
X = np.tile([0.5, 1, 2, 4, 8], (6, 1))          # 6 subjects x 5 times
Y = 10 * np.exp(-0.3 * X) * np.exp(np.random.default_rng(0).normal(0, 0.1, (6, 1)))
nlmefit(X, Y, None, None, lambda phi, t: phi[0] * np.exp(-phi[1] * t[:, 0]), [8, 0.2],
        REParamsSelect=[1])
```

The layout follows from the shape of `y` (a vector is stacked; a matrix with more than
one row and column is grouped) or is set with `DataFormat="stacked"`/`"grouped"`.

## 6. ODE models

See `examples/sund_ode_model.py` for a model compiled with SUND, and the API reference
for `ode_model` (SciPy).

## 7. R conventions

`compat="r"` follows R's nlme: its iteration and default tolerances for `"LME"` and
`"RELME"`, and its standard errors and BIC; `nlmefitsa` defaults to one chain like
saemix. On well-behaved problems the estimates are those of the default mode to the
convergence tolerance.
