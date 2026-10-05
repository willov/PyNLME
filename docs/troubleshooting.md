# Troubleshooting

## Warnings and errors

**`nlmefit did not converge in N iterations`**: the result is the best state found.
Try better starting values, `Options={"MaxIter": 500}`, or a simpler random-effects
structure (`REParamsSelect`). The Lindstrom-Bates iteration can cycle between states on
weakly identified models; FOCE (`ApproximationType="FOCE"`) or `nlmefitsa` may then help.

**`NotImplementedError`**: the option is not implemented yet (e.g. `FEObsDesign`, a
block-structured `CovPattern`, `LogLikMethod="gq"`). pynlme never ignores an option.

**`TypeError: ... unexpected keyword argument`**: misspelt option. MATLAB names are
CamelCase (`ParamTransform`), Python names snake_case (`param_transform`).

**`group is required for stacked data`**: `y` is a vector, so each observation needs a
group. For one row per subject, pass `y` as a matrix or set `DataFormat="grouped"`.

**`modelfun returned non-finite values`**: the model produced NaN or inf, e.g. from a
parameter outside its valid range. Use `ParamTransform` (log for positive parameters,
logit/probit for probabilities) instead of constraining inside the model.

**`modelfun returned k values for a group with n observations`**: the model must return
one value per row of `x`. With the default `Vectorization="SinglePhi"`, `phi` is a
vector and `x` an n x h matrix (a column for a single predictor: use `x[:, 0]` or let
NumPy broadcast against `phi[k]`).

## Results differ from MATLAB

- **Parameter indices**: under MATLAB names (`REParamsSelect=[2, 3]`) they are 1-based,
  under Python names (`re_params_select=[1, 2]`) 0-based.
- **Same options, same result**: `nlmefit` (LME/RELME) follows MATLAB's iteration and
  default options (`TolX = TolFun = 1e-4`), so results agree with MATLAB's to ~1e-10 on
  most data. Pass the same `Options` as in MATLAB.
- **Within MATLAB's tolerance**: MATLAB's LME step runs `fminsearch` to 1e-2, and where it
  stops can flip with perturbations at the 1e-9 level (even MATLAB's own result moves if
  its start is rounded). Such fits agree to about MATLAB's `TolX` (1e-4). Tighter
  `TolX`/`TolFun` make both converge to the same optimum.
- **Warning "A random-effects variance was estimated as zero during the iterations"**:
  the likelihood is flat along that variance, and where the iteration stops there is
  decided by rounding, in MATLAB too (its result changes with the number of BLAS
  threads). pynlme's fit then has the same or a slightly different likelihood. Remove
  the random effect (`REParamsSelect`) or check the fit with `nlmefitsa`.
- **SAEM** is stochastic; compare several seeds, not single runs.
- **FO**: MATLAB's FO often collapses all variances to zero; pynlme finds better optima.
- **FO/FOCE standard errors**: MATLAB reuses those of its initial LME stage; pynlme
  computes them at the FO/FOCE estimates.
- **Label switching**: symmetric starting values (e.g. two identical exponential terms)
  can give the same fit with parameters in a different order.

## Results differ from R

Use `compat="r"` (or `pynlme.r.nlme`): it follows nlme's iteration and defaults
(`maxIter=50`, `tolerance=1e-5`, `pnlsTol=1e-3`), so the fit is R's to ~1e-6 with the same
controls, and the reported statistics follow R (ML standard errors scaled by
sqrt(N/(N-f)), BIC with log(observations)). Without it, pynlme runs the PNLS step to
convergence; usually that reaches R's fit too, but where the model has several local
solutions (seen on ~0.1% of simulated oral-absorption datasets) it can reach a
different one. R's `nlme(method="REML")` is not a restricted-likelihood fit for nonlinear
models (its PNLS step updates the covariance by ML); `compat="r"` reproduces it, the
default RELME does not.

## Speed

1. Write the model for `Vectorization="Full"` (see the API reference): one model call
   for many groups instead of one per group.
2. For ODE models use a compiled SUND model via `sund_model` (batched with containers),
   or `ode_model` with a vectorised right-hand side.
3. Supply analytic derivatives with `with_jacobian` when the model is expensive.
4. `nlmefitsa`: fewer chains (`NChains=1`) or iterations trade precision for speed.
