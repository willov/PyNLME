# Validation

How pynlme is checked. There are three kinds of test in
[tests/validation/](../tests/validation/):

1. **Fixed reference cases**: real datasets fitted by MATLAB R2023b
   (`test_matlab_references.py`), R `nlme` (`test_nlmefit_references.py`) and R
   `saemix` plus MATLAB `nlmefitsa` (`test_saem_references.py`), stored as JSON
   here. These need neither R nor MATLAB to run.
2. **Randomized comparison with R** (`test_random_vs_r.py`): every run simulates
   new datasets (new true parameters, group counts and sampling times) and fits
   them with both R `nlme` and pynlme. Skipped when R with `nlme` and `jsonlite`
   is not installed. `PYNLME_RANDOM_SEED` reproduces a run; `PYNLME_RANDOM_N`
   sets the number of datasets per model family (default 5).
3. **Simulation studies** (`test_simulation_recovery.py`, run with `--runslow`):
   hundreds of datasets simulated from known parameters, checking bias and
   confidence-interval coverage. No reference software involved.

The simulated model families (oral PK with dose as `V`, logistic growth, linear
with full covariance, Emax) are defined in
[tests/validation/simulate.py](../tests/validation/simulate.py) and mirrored in
[r/fit_manifest.R](r/fit_manifest.R).

## Layout

| Path | Contents |
|---|---|
| `datasets/` | Data as CSV: Indometh, Theoph, Orange (from R's `datasets`) and `linear_sim` (simulated, see `r/datasets.R`). Indometh is the same data as MATLAB's `indomethacin.mat`. |
| `reference/r_nlme_*.json` | R `nlme` fits, from `r/nlme_references.R` |
| `reference/matlab_docs.json` | Values printed in the MATLAB nlmefit/nlmefitsa documentation |
| `reference/r_saemix_*.json` | R `saemix` fits (10 seeds per case), from `r/saemix_references.R` |
| `reference/matlab_*.json` | MATLAB R2023b fits, from `matlab/run_references.m`: `nlmefit` with LME, RELME, FO and FOCE (default and tight tolerances) and `nlmefitsa` (5 seeds) |
| `r/fit_manifest.R` | Fits simulated datasets for the randomized comparison |
| `r/benchmark.R` | R `nlme`/`saemix` timings (best of repeated runs) for `benchmark.py` |
| `benchmark.py` | Times nlmefit/nlmefitsa against R and MATLAB on the validation cases |

The case definitions (model, start values, options) are in
[tests/validation/cases.py](../tests/validation/cases.py), mirrored in the R and
MATLAB scripts.

## Reference cases

| Case | Model | What it exercises |
|---|---|---|
| `indometh_lme` | Bi-exponential, `ParamTransform=[0 1 0 1]` | MATLAB doc example; a variance estimated at zero |
| `theoph_lme` | One-compartment oral PK, dose as `V` | Group-level predictors, `REParamsSelect=[2 3]` |
| `theoph_wt_lme` | As above, weight on log clearance | `FEGroupDesign` (covariate model) |
| `orange_lme` | Logistic growth | Single random effect, large parameter scales |
| `linear_lme`, `linear_relme` | Linear, full covariance | Exact LME/REML answer; `CovPattern=ones(2)` |

## Findings worth knowing

- **pynlme `nlmefit` matches MATLAB R2023b** to ~1e-6 on all outputs for LME and
  RELME on the five datasets. MATLAB conventions that pynlme follows:
  `rmse = sqrt(sum(ires^2) / dfe)` (not the error SD, which is `errorparam`);
  `dfe = N - (fixed effects + covariance parameters + error parameters)`;
  `bic` uses log(number of groups); for RELME the error variance is reported as
  RSS/N (rescaling psi and covb accordingly) and BIC uses log(m - f); standard
  errors are unadjusted (no small-sample factor).
- **MATLAB's FO collapses every random-effects variance to zero** on all five
  datasets (log-likelihood far below the true FO optimum), and on the linear
  model it ran for 1.5 hours without returning a log-likelihood. pynlme's FO is
  tested to reach MATLAB's log-likelihood or a better one. FO and FOCE coincide
  when the random effects enter the model linearly, as they must.
- **MATLAB's FO/FOCE standard errors** come from the LME stage it runs first,
  rescaled by the final error SD (nlmefit.m computes `covb` from that stage),
  not from the FO/FOCE estimates. pynlme computes them at its own estimates
  (a few percent different); a test pins down MATLAB's behaviour.
- **Error models** (nlmefit.m, nlmeres.m): proportional/combined fits use
  iteratively reweighted least squares with weights 1/g(f) refreshed after each
  LME step and `sum(log(weights))` added to the log-likelihood; combined (a, b)
  are refitted by Nelder-Mead; exponential fits log(y) = log(f) + a e with the
  log-likelihood corrected by -sum(log y). All weighted residuals use the error
  SD at the population prediction. pynlme follows all of this.
- **MATLAB nlmefitsa runs max(1, round(100/m)) MCMC chains** by default (an
  undocumented `NChains` option); pynlme does the same (saemix uses 1).
- **R's BIC conventions differ by package**: nlme uses log(number of
  observations), saemix log(number of subjects) like MATLAB. `compat="r"`
  follows nlme for `nlmefit` and saemix for `nlmefitsa`.
- **Theophylline at time 0**: the oral model predicts exactly 0 while some
  observed concentrations are positive, so proportional and exponential error
  models are ill-posed with those rows (MATLAB returns error SDs of ~1e7 and
  ~110). The error-model references drop time 0.
- **MATLAB R2023b reproduces the current (2026) documentation's values exactly**,
  so a newer MATLAB is not needed for references.
- **SAEM on small datasets**: with one MCMC chain the covariance estimate can
  collapse to a singular matrix (correlation 1) during the full-step phase; this
  is inherent to stochastic EM with few subjects (it happens even with exact
  posterior draws). pynlme limits how fast eigenvalues may shrink in that phase
  and, by default, runs ceil(50/m) chains per subject (saemix's
  recommendation). With that, run-to-run variation is at or below saemix's and
  below MATLAB's.
- **MATLAB's indomethacin `nlmefitsa` example is not reproducible**: across seeds
  MATLAB itself gives estimates with standard deviation ~0.85, and saemix ~0.5-2.5.
  It is not used as a SAEM reference.

- **R `nlme` is a valid stand-in for MATLAB `nlmefit` (`"LME"`)**, if its inner
  tolerance is tightened. With `pnlsTol = 1e-6` it reproduces the MATLAB
  documentation's indomethacin estimates to all four printed decimals. With the
  default `pnlsTol = 1e-3` the estimates are off by about 2e-3.
- **R `nlme(method = "REML")` is not a restricted-likelihood fit.** For nonlinear
  models it returns the ML estimates with sigma² scaled by N/(N−p). This shows
  even on a linear model, where `lme` REML gives a different psi. So `"RELME"` is
  only checked against `lme` REML on the linear case; the nonlinear cases need
  MATLAB results.
- **Standard errors**: R's `summary()` multiplies ML standard errors by
  sqrt(N/(N−p)). pynlme reports the unadjusted asymptotic values, and the
  tests apply R's factor before comparing. Which convention MATLAB uses still
  needs a MATLAB run.
- **Label switching**: the MATLAB indomethacin example starts at `[1 1 1 1]`, which
  is symmetric in the two exponential terms. pynlme reaches the same optimum
  but may label the terms in the other order.
- **Randomized comparison (640 datasets, 20 seeds)**: pynlme reaches R's
  solution to ~1e-6 on 594 of the 598 datasets where R converged, a better one on
  3, and one 0.003 lower in log-likelihood. R failed to converge on 42 datasets;
  pynlme returned an estimate for all of them. The Lindstrom-Bates iteration can
  have several fixed points, so small differences are expected.
- **Robustness fixes found by the randomized comparison**: the LME step is
  computed with QR decompositions (cross-products cancel catastrophically for
  large random-effects scales), uses a log-Cholesky parameterisation, restarts
  variance components that collapse to zero, starts with an LME step at the
  start values as nlme does, and damps cycling iterations.
- **`RefineBeta0`** (MATLAB's default pooled pre-fit) never gave a better fit by
  more than 0.01 in log-likelihood over the 640 datasets and gave a worse one in
  3. It stays on by default for MATLAB parity; the randomized comparison
  switches it off because R has no such step.
- **Simulation studies (30 groups)**: for the linear model, where the LME
  approximation is exact, fixed effects are unbiased and 95% intervals cover
  93-95.5%. For the nonlinear models the linearisation causes small biases
  (largest: -13% for log(ka) in the oral model, 90% coverage; others within
  1%), shared with R `nlme`. Variance components are ~3-11% low, as expected
  for ML with 30 groups.

## Regenerating references

R (needs `nlme`, `jsonlite`, `saemix`; `saemix` needs a Fortran compiler to
install):

```bash
Rscript validation/r/datasets.R          # only when adding datasets
Rscript validation/r/nlme_references.R
Rscript validation/r/saemix_references.R
```

MATLAB (Statistics and Machine Learning Toolbox), from the repository root:

```matlab
addpath validation/matlab
run_references
```

This fits every case with nlmefit (`LME`, `RELME`, `FO`, `FOCE`; MATLAB default and
tight tolerances) and nlmefitsa (five seeds). It writes
`reference/matlab_*.json` and exports `orangetreedata.mat` and
`nonlineardata.mat` to `datasets/` so the remaining documentation examples can
be tested.

Some documentation datasets (`orangetreedata.mat`, `nonlineardata.mat`) are not
part of every MATLAB installation; the script skips them if missing, and skips
fits whose JSON already exists (delete a file to regenerate it).

On Arch Linux, MATLAB's license service may fail to start with a
`libpangoft2 ... FcConfigSetDefaultSubstitute` error; starting MATLAB with
`LD_PRELOAD=/usr/lib/libfontconfig.so.1` works around it.

## Not covered yet

- Non-constant error models with FO/FOCE (MATLAB references exist).
- The MATLAB documentation examples that need `orangetreedata.mat` and
  `nonlineardata.mat`.
