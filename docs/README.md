# pynlme documentation

pynlme fits nonlinear mixed-effects models in Python with the call syntax of MATLAB's
`nlmefit` and `nlmefitsa`, validated against MATLAB R2023b, R `nlme` and R `saemix`.

Where to start:

- **MATLAB users**: [tutorial.md](tutorial.md) follows MATLAB's own documentation
  examples; options and outputs carry MATLAB's names and definitions.
- **Python users**: [`fit`](api_reference.md#fitmodel-data-response-group-start-predictors-group_predictors-transformnone-randomnone-methodlme-options)
  with named parameters, dict/Polars data and dict results; or
  [api_reference.md](api_reference.md) for snake_case options, 0-based indices, `fit_nlme`, `Vectorization="Full"`, ODE models).
- **R (nlme/saemix) users**: `pynlme.r.nlme` with R's formulas, `summary()`, `fixef()`,
  ... ([API reference](api_reference.md#r-style-pynlmer)); `compat="r"` and the translation table in the
  [main README](../README.md#r-users-nlme-saemix-who-would-rather-not-use-r);
  differences from R in [troubleshooting.md](troubleshooting.md#results-differ-from-r).

| Page | Contents |
|---|---|
| [tutorial.md](tutorial.md) | From a MATLAB example to covariates, error models, SAEM, faster fits and ODE models |
| [api_reference.md](api_reference.md) | Functions, options (MATLAB and Python names), outputs, MATLAB vs R conventions |
| [multidimensional_input.md](multidimensional_input.md) | Stacked and grouped data layouts |
| [troubleshooting.md](troubleshooting.md) | Warnings, differences from MATLAB, speed |
| [implementation.md](implementation.md) | Algorithms and code structure, for developers |
| [../validation/README.md](../validation/README.md) | How the package is validated, and what was found |
| [sund-wishlist.md](sund-wishlist.md) | Features requested from the SUND toolbox |
| [release-workflow.md](release-workflow.md) | Making a release |

## What is implemented

- `nlmefit`: `ApproximationType` `"LME"`, `"RELME"`, `"FO"`, `"FOCE"`; error models
  `"constant"`, `"proportional"`, `"combined"`, `"exponential"`; parameter transforms,
  fixed- and random-effects designs (including group-specific covariate designs),
  diagonal or full covariance; all `stats` fields including weighted residuals.
- `nlmefitsa`: SAEM with MATLAB's defaults (iterations, MCMC kernels, chains), all error
  models, importance-sampling and linearised log-likelihoods, standard errors.
- `Vectorization="Full"` for batched model evaluation; analytic derivatives via
  `with_jacobian`; ODE models via SUND (`sund_model`) or SciPy (`ode_model`).
- `compat="matlab"` (default) or `compat="r"` reporting conventions.

Not implemented: `FEObsDesign`/`REObsDesign`, block-structured `CovPattern`,
`LogLikMethod="gq"`, `Replicates`, `OutputFcn`. These raise errors rather than being
ignored.
