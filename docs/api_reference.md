# API reference

```python
from pynlme import nlmefit, nlmefitsa, fit_nlme, ode_model, sund_model, with_jacobian
```

## `nlmefit(X, y, group, V, modelfun, beta0, **options)`

Maximum-likelihood fit, as MATLAB's `nlmefit`. The model for group i is

```
y_ij = f(phi_i, x_ij[, v_i]) + error
phi_i = g^-1(A_i beta + B_i b_i),   b_i ~ N(0, PSI)
```

where `g` is the parameter transform (`ParamTransform`), and `A_i`, `B_i` are the fixed-
and random-effects designs (by default identity: every parameter has a fixed and a
random effect).

| Argument | Description |
|---|---|
| `X` | Predictors, one row per observation (n x h). A vector is one column. |
| `y` | Responses. Stacked: a vector. Grouped: one row per group (see `DataFormat`). |
| `group` | Group of each observation (any labels; groups are ordered by sorted label). Required for stacked data; optional row labels for grouped data. |
| `V` | Group-level predictors, one row per group (m x g), or `None`. A vector is one value per group. |
| `modelfun` | `modelfun(phi, x)` if `V` is `None`, else `modelfun(phi, x, v)`. `phi` is the parameter vector of one group, `x` that group's rows of `X` (n_i x h), `v` its row of `V`. Returns n_i predictions. With `Vectorization="Full"`, see below. |
| `beta0` | Starting values of the fixed effects. |

**Returns** `beta, psi, stats, b`:

| Output | Shape | Contents |
|---|---|---|
| `beta` | (f,) | Fixed effects (on the transformed scale) |
| `psi` | (r, r) | Random-effects covariance |
| `stats` | `NLMEStats` | See below |
| `b` | (r, m) | Random effects, one column per group (MATLAB layout) |

### Options

Pass MATLAB name-value arguments as keyword arguments (`ParamTransform=[0, 1]`), or the
snake_case names (`param_transform=[0, 1]`). **Indices given under MATLAB names are
1-based, as in MATLAB; under Python names they are 0-based.** Logical masks work under
both.

| MATLAB name | Python name | Default | Notes |
|---|---|---|---|
| `ApproximationType` | `approximation_type` | `"LME"` | `"LME"`, `"RELME"`, `"FO"`, `"FOCE"` |
| `ErrorModel` | `error_model` | `"constant"` | `"constant"`, `"proportional"`, `"combined"`, `"exponential"` |
| `ErrorParameters` | `error_parameters` | MATLAB's | Starting value(s); `[a, b]` for combined |
| `ParamTransform` | `param_transform` | none | Per parameter: 0 identity, 1 log, 2 probit, 3 logit |
| `CovPattern` | `cov_pattern` | diagonal | Diagonal or full (`np.ones((r, r))`) |
| `FEParamsSelect`, `FEConstDesign`, `FEGroupDesign` | `fe_params_select`, ... | all parameters | `FEGroupDesign` is p x f x m (covariate models) |
| `REParamsSelect`, `REConstDesign`, `REGroupDesign` | `re_params_select`, ... | all parameters | |
| `RefineBeta0` | `refine_beta0` | `"on"` | Pooled fit of `beta0` first |
| `Vectorization` | `vectorization` | `"SinglePhi"` | `"SinglePhi"`, `"SingleGroup"`, `"Full"` |
| `Options` | | | dict with `MaxIter` (200), `TolFun` and `TolX` (1e-6; MATLAB's default is 1e-4), `Display` (`"off"`, `"final"`, `"iter"`) |
| `CovParameterization`, `OptimFun` | | | Accepted for compatibility; no effect on the result |
| `DataFormat` | `data_format` | from `y` | `"stacked"` or `"grouped"` |
| | `compat` | `"matlab"` | `"matlab"` or `"r"` reporting conventions (below) |

Options that are not implemented (`FEObsDesign`, `REObsDesign`, block `CovPattern`)
raise `NotImplementedError`; unknown options raise `TypeError`. Nothing is silently
ignored.

### `stats`

| Field | MATLAB definition |
|---|---|
| `dfe` | N - (fixed effects + covariance parameters + error parameters) |
| `logl`, `aic`, `bic` | Maximised (approximate) log-likelihood; `aic = -2 logl + 2 k`, `bic = -2 logl + log(m) k` |
| `rmse` | `sqrt(sum(ires^2) / dfe)` |
| `mse` | Estimated error variance |
| `errorparam` | Error-model parameters: `a` (constant, exponential), `b` (proportional), `[a, b]` (combined) |
| `sebeta`, `covb` | Standard errors and covariance of `beta` |
| `ires`, `pres` | `y - f(beta, b)`, `y - f(beta, 0)` |
| `iwres`, `pwres`, `cwres` | Weighted residuals as MATLAB's `nlmeres` (error SD at the population prediction); `None` when an error variance is zero |

For the exponential error model the model is fitted on the log scale and residuals are
on that scale, as in MATLAB.

### `compat`: MATLAB or R conventions

Estimates are the same; defaults and reported statistics differ.

| | `compat="matlab"` | `compat="r"` (nlme) |
|---|---|---|
| `RefineBeta0` default | on | off |
| ML standard errors | unadjusted | x sqrt(N / (N - f)) |
| BIC | log(groups) | log(observations) (N - f for REML) |
| RELME error variance, `psi`, `covb` | RSS / N | RSS / (N - f) |
| FO/FOCE exponential-model log-likelihood | without `-sum(log y)` (as MATLAB) | full likelihood |

Explicit options override the convention set.

## `nlmefitsa(X, y, group, V, modelfun, beta0, **options)`

Stochastic Approximation EM, as MATLAB's `nlmefitsa`: same arguments and outputs.
Supports `ParamTransform`, `CovPattern`, `ErrorModel`, `ErrorParameters`,
`FEParamsSelect`/`FEConstDesign`/`FEGroupDesign`, `REParamsSelect`, `Vectorization`,
`DataFormat`, `compat`, and:

| MATLAB name | Python name | Default |
|---|---|---|
| `NIterations` | `n_iterations` | `[150, 150, 100]` (annealing, full step, decreasing step) |
| `NMCMCIterations` | `n_mcmc_iterations` | `[2, 2, 2]` (prior draws, single-coordinate, full-vector random walk) |
| `NBurnIn` | `n_burn_in` | 5 |
| `NChains` | `n_chains` | `max(1, round(100 / m))` (MATLAB's); 1 with `compat="r"` (saemix's) |
| `LogLikMethod` | `loglik_method` | `"none"`; `"is"` (importance sampling) or `"lin"` (linearisation) |
| `ComputeStdErrors` | `compute_std_errors` | `False` |
| | `random_state` | seed for reproducible results |

With `LogLikMethod="none"`, `stats.logl`, `aic` and `bic` are `None`, as in MATLAB. `b`
holds the conditional means of the random effects.

## `fit(model, data, response, group, start, predictors=(), group_predictors=(), transform=None, random=None, method="LME", **options)`

Named parameters, data as a dict of arrays or a dataframe (Polars, or anything indexed
by column name), results as a plain dict. Runs `nlmefit`/`nlmefitsa` with
`Vectorization="Full"`.

```python
import numpy as np
from pynlme import fit

def model(p, x, v):  # dicts of arrays, one entry per observation
    ke, ka, cl, t = p["ke"], p["ka"], p["cl"], x["time"]
    return v["dose"] * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

result = fit(model, data, response="conc", group="subject",
             predictors=["time"], group_predictors=["dose"],
             start={"ke": 0.08, "ka": 1.6, "cl": 0.04},          # natural scale
             transform={"ke": "log", "ka": "log", "cl": "log"},
             random=["ka", "cl"], method="LME")
result["typical"]["cl"]   # 0.040 (natural scale)
```

- `start`: initial values on the natural scale; its key order is the parameter order.
- `transform`: `"log"`, `"logit"`, `"probit"` or `"identity"` (default) per parameter.
- `random`: parameters with random effects (default all).
- `covariates`: group-level covariate models, `{"cl": ["wt"]}` makes the transformed
  `cl` equal `beta_cl + beta_cl.wt * wt`; the coefficient `"cl.wt"` starts at 0 unless
  given in `start` (transformed scale).
- `group_predictors`: columns constant within a group, passed to the model as `v`
  (an error is raised if they vary within a group).
- `method`: `"LME"`, `"RELME"`, `"FO"`, `"FOCE"` or `"SAEM"`.
- `pynlme.predict(result, data=None, level=1)`: predictions for new rows (level 1
  individual, NaN for unknown groups; level 0 population).
- `**options`: any `nlmefit`/`nlmefitsa` option by its Python name
  (`error_model="proportional"`, `cov_pattern=...`, `random_state=1`, `compat="r"`).
- A `model.jacobian(p, x[, v])` returning d f/d phi (columns in parameter order) is used
  if present.

The result:

| Key | Contents |
|---|---|
| `fixed` | `{name: estimate}` on the transformed scale |
| `typical` | the same on the natural scale |
| `se` | `{name: standard error}` of `fixed` |
| `psi` | `{"names": [...], "matrix": array, "variances": {name: variance}}` |
| `random_effects` | `{group label: {name: b}}` |
| `error` | `{"model": ..., "parameters": ...}` |
| `stats` | all `stats` fields of `nlmefit` as a dict |
| `transform`, `method` | as used |

## R-style: `pynlme.r`

```python
from pynlme.r import nlme, pdDiag, pdSymm, summary, fixef, ranef, VarCorr, vcov, logLik, AIC, BIC, fitted, resid
fit = nlme("conc ~ SSfol(dose, time, lKe, lKa, lCl)", data=theo,
           fixed=["lKe ~ 1", "lKa ~ 1", "lCl ~ wtc"], random=pdDiag("lKa + lCl ~ 1"),
           groups="~subject", start=[-2.5, 0.5, -3.2, 0])
```

`nlme(model, data, fixed, random=None, groups=None, start, method="ML", **options)`:

- `model`: `"response ~ expression"`. The expression may use data columns, the
  parameters, `exp`, `log`, `sqrt`, `^` and other R math functions, R's `SSfol`,
  `SSlogis`, `SSbiexp`, `SSasymp`, `SSmicmen` (without self-starting), and any Python
  function visible where `nlme` is called. It is evaluated on all rows at once.
- `fixed`: `"a + b ~ 1"` or a list `["a ~ 1", "b ~ wt"]` (group-level covariates by
  column name; coefficients named `b.(Intercept)`, `b.wt` as in R).
- `random`: `"a + b ~ 1"` (general covariance, R's default `pdSymm`), `pdDiag(...)`,
  `pdSymm(...)`; optionally `"... ~ 1 | subject"`. Default: all parameters.
- `groups`: `"~subject"` (or via `| subject` in `random`).
- `start`: dict by coefficient name or list in R's coefficient order; required.
- `method`: `"ML"` (R's default) or `"REML"` (true REML, unlike R's nlme).
- `**options`: `nlmefit` options by Python name, e.g. `error_model="proportional"`.

Runs `fit` with `compat="r"`. The result is `fit`'s dict plus `fixef`, `se_fixef`,
`ranef` (`{group: {name: b}}`), `sigma`, `logLik`, `AIC`, `BIC`, `vcov`, `fitted`,
`residuals` and `call`, read by the R-named functions. `summary(fit)` returns R's
printout (DF = observations - groups - fixed effects + 1, as nlme reports).
`predict(fit, newdata=None, level=1)`: individual predictions (NaN for groups not in the
fit, R's NA) or `level=0` population predictions.

### saemix: `saemixData`, `saemixModel`, `saemix`

- `saemixData(data, name_group, name_predictors, name_response, name_covariates=())`.
- `saemixModel(model, psi0, description="", transform_par=None, covariate_model=None,
  covariance_model=None, error_model="constant")`. `model(psi, id, xidep)` as in R
  (`psi[id, k]`, `xidep[:, j]`; `id` is 0-based). `psi0` is a dict
  `{"ka": 1.0, ...}`; a list `{"CL": [0.5, 0]}` adds starting values for covariate effects
  (R's extra psi0 rows). `covariate_model`: R's 0/1 matrix (covariates x parameters) or
  `{"CL": ["wt"]}`. `covariance_model`: diagonal, or full among the random parameters.
  `error_model`: constant, proportional, combined, exponential. `omega_init` and
  `fixed_estim` are not supported (they raise).
- `saemix(model, data, control)`. `control` uses saemix's names: `seed`,
  `nbiter.saemix`, `nbiter.sa`, `nb.chains` (default 1), `nbiter.mcmc`, `nbiter.burn`,
  `ll.is`, `fim`; display/save options are accepted and ignored, others raise.

The result is `fit`'s dict plus `population` (natural scale) and `se_population`
(delta method), `covariate_effects` (`"beta_wt(CL)"`, transformed scale),
`se_covariate_effects`, `omega`, `respar` (`a.1`, `b.1`), `logLik`, `AIC`, `BIC`.
Functions: `summary`, `coef` (fixed, population, individual `map.psi`/`cond.mean.psi`),
`psi`/`phi`/`eta` (`type="mode"`, the default as in saemix, computed by maximising
each subject's posterior; or `"mean"`), `predict(fit, newdata, type=)` with `"ipred"`
(modes), `"icpred"` (conditional means), `"ppred"` (population parameters), `"ypred"`
(mean over the random effects, Monte Carlo), `logLik`, `AIC`, `BIC`.

## `fit_nlme(X, y, group, V, modelfun, beta0, method="MLE", **options)`

`method="MLE"` (or `"ML"`) calls `nlmefit`, `method="SAEM"` calls `nlmefitsa`.
`fit_mle` and `fit_saem` are aliases.

## `Vectorization="Full"`

The model is called once for many groups: `phi` has one row per observation (n x p),
`x` is n x h and `v` n x g. Write the model by indexing the last axis so that it works
for a single parameter vector too:

```python
def model(phi, t, dose):
    ke, ka, cl = phi[..., 0], phi[..., 1], phi[..., 2]
    t, dose = t[:, 0], dose[..., 0]
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
```

Results are identical to `"SinglePhi"`; fits are several times faster.

## Analytic derivatives: `with_jacobian(modelfun, jacobian)`

`jacobian` takes the same arguments as the model and returns d f / d phi (one row per
observation, one column per parameter, phi on the model's scale). All methods then use
it instead of finite differences; the chain rule through `ParamTransform` is applied by
pynlme. Equivalently, set a `jacobian` attribute on the model function.

## ODE models

### `sund_model(model, feature, parameters, v_states=(), v_parameters=(), ...)`

A compiled [SUND](https://isbgroup.eu/sund-toolbox) model as a model function
(`pip install "pynlme[sund]"`). `model` is the name of an installed SUND model (or a
`sund.Simulation`); `parameters` are the SUND parameters set from `phi`, in order; the
columns of `V` set the initial states `v_states`, then the parameters `v_parameters`.
With `Vectorization="Full"` it simulates `batch_size` (16) parameter sets per SUND call
using containers. Integrator tolerances default to rel 1e-10, abs 1e-12, because the
model is differentiated numerically. See `examples/sund_ode_model.py`.

### `ode_model(rhs, y0, output, t0=0, method="DOP853", rtol=1e-10, atol=1e-12)`

A SciPy ODE model: `rhs(t, y, phi, v)` vectorised over K systems (y is K x states),
`y0(phi, v)`, `output(y, phi, v)`. With `Vectorization="Full"` all subjects are solved
in one `solve_ivp` call.

## Data helpers

- `stack_grouped_data(X_grouped, y_grouped, group_ids=None)`: grouped to stacked arrays.
- `transform_parameters(phi, codes)`: apply `ParamTransform` codes.
- `detect_data_format(X, y, group)`: a heuristic guess, not used by the fitting functions.
