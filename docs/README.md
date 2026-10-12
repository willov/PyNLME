# pynlme documentation

pynlme fits nonlinear mixed-effects models in Python with the call syntax of MATLAB's
`nlmefit` and `nlmefitsa`, validated against MATLAB R2023b, R `nlme` and R `saemix`.

Where to start:

- **MATLAB users**: [tutorial.md](tutorial.md) follows MATLAB's own documentation
  examples; options and outputs carry MATLAB's names and definitions.
- **Python users**: `fit_nlme` with snake_case options and 0-based indices
  ([api_reference.md](api_reference.md): `fit_nlme`, `Vectorization="Full"`, ODE
  models). `fit` (named parameters, dict/Polars data and results) is provisional.
- **R (nlme/saemix) users**: `pynlme.r.nlme` with R's formulas, and
  `saemixData`/`saemixModel`/`saemix`, with `summary()`, `fixef()`, `predict()`,
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
| [Speed](../README.md#speed) | Benchmarks against MATLAB, R and nlmixr2 on the validation cases (same machine, same data, start values and defaults) |
| [../validation/accuracy/README.md](../validation/accuracy/README.md) | Accuracy benchmark: bias, RMSE and confidence-interval coverage over 100 simulated datasets, against MATLAB, R and nlmixr2 |
| [../validation/scaling/README.md](../validation/scaling/README.md) | Scaling benchmark: 10, 100 and 1000 subjects, closed-form and ODE models, with each program's log-likelihood as a fit check |
| [nlmixr2 comparison](../validation/README.md#comparison-with-nlmixr2) | FOCE and SAEM against nlmixr2, scored by the same exact log-likelihood |
| [release-workflow.md](release-workflow.md) | Making a release |

## What is implemented

- `nlmefit`: `ApproximationType` `"LME"`, `"RELME"`, `"FO"`, `"FOCE"`; error models
  `"constant"`, `"proportional"`, `"combined"`, `"exponential"`; parameter transforms,
  fixed- and random-effects designs (including group-specific covariate designs),
  diagonal or full covariance; all `stats` fields including weighted residuals.
- `nlmefitsa`: SAEM with MATLAB's defaults (iterations, MCMC kernels, chains), all error
  models, importance-sampling and linearised log-likelihoods, standard errors.
- `Vectorization="Full"` for batched model evaluation; analytic derivatives via
  `with_jacobian`; ODE models via SUND (`sund_model`) or SciPy (`ode_model`), including
  several observed features per subject (stacked data with a feature column) and doses
  and other inputs from SUND activities (one set per experiment).
- `compat="matlab"` (default): `nlmefit` LME/RELME follow MATLAB's iteration and default
  options, giving MATLAB's results (~1e-10 on most data). `compat="r"`: they follow
  nlme's iteration and `nlmeControl` defaults, giving R's (including nlme's REML), and
  the reporting conventions are R's.
- `pynlme.r`: R's `nlme()` with string formulas and `saemixData`/`saemixModel`/`saemix`,
  with `summary`, `fixef`, `ranef`, `VarCorr`, `predict`, ...
- `fit` and `predict`: named parameters, dict/Polars data and results (provisional).

Not implemented: `FEObsDesign`/`REObsDesign`, block-structured `CovPattern`,
`LogLikMethod="gq"`, `Replicates`, `OutputFcn`. These raise errors rather than being
ignored.
