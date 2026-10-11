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
3. **Randomized comparison with MATLAB** (`test_random_vs_matlab.py`): as 2, against
   MATLAB's `nlmefit` (LME and RELME). Skipped unless `PYNLME_MATLAB` gives the command
   that starts MATLAB, e.g.
   `PYNLME_MATLAB="LD_PRELOAD=/usr/lib/libfontconfig.so.1 /usr/local/MATLAB/R2023b/bin/matlab"`.
4. **Simulation studies** (`test_simulation_recovery.py`, run with `--runslow`):
   hundreds of datasets simulated from known parameters, checking bias and
   confidence-interval coverage. No reference software involved.

CI runs the fast tests (`uv run pytest`) on every push to `main` and on pull requests.
The slow checks are run by hand, when the algorithms change:

```bash
uv run pytest --runslow                              # + simulation studies, R if installed
PYNLME_RANDOM_N=10 uv run pytest tests/validation    # more random datasets
```

or in CI via Actions -> CI/CD Pipeline -> Run workflow (the `validation` job, R only).

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
| `datasets/mm_pk.csv`, `make_mm_dataset.py`, `models/mm_pk_sund.txt` | Michaelis-Menten PK data simulated from known parameters, and the model in SUND format |
| `r/fit_manifest.R` | Fits simulated datasets for the randomized comparison |
| `r/benchmark.R` | R `nlme`/`saemix` timings (best of repeated runs) for `benchmark.py`, written to `reference/r_timings.json` |
| `matlab/benchmark.m` | MATLAB `nlmefit`/`nlmefitsa` timings (best of repeated runs after a warm-up, `SinglePhi` and `Full`), written to `reference/matlab_timings.json` |
| `benchmark.py` | Times pynlme on the benchmark cases against those timings; `--markdown` prints the README's tables and the machine's specifications |
| `scaling/` | Scaling benchmark: 10, 100 and 1000 simulated subjects, closed-form and ODE models, pynlme against MATLAB, R and nlmixr2 (see `scaling/README.md`) |
| `r/nlmixr2_benchmark.R`, `nlmixr2_compare.py` | nlmixr2 (FOCEi, SAEM) on the benchmark cases, and pynlme's and MATLAB's FOCE and SAEM scored with the same exact log-likelihood (see below) |

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
- **MATLAB's FO/FOCE log-likelihood for the exponential error model omits
  the -sum(log y) term** that its LME/RELME include (theophylline: FOCE +1.6,
  LME -182.6 for nearly the same fit). `compat="matlab"` reproduces this;
  `compat="r"` reports the full likelihood for every method.
- **MATLAB nlmefitsa runs max(1, round(100/m)) MCMC chains** by default (an
  undocumented `NChains` option); pynlme does the same (saemix uses 1).
- **R's BIC conventions differ by package**: nlme uses log(number of
  observations), saemix log(number of subjects) like MATLAB. `compat="r"`
  follows nlme for `nlmefit` and saemix for `nlmefitsa`.
- **Theophylline at time 0**: the oral model predicts exactly 0 while some
  observed concentrations are positive, so proportional and exponential error
  models are ill-posed with those rows (MATLAB returns error SDs of ~1e7 and
  ~110). The error-model references drop time 0.
- **ODE models**: on the Michaelis-Menten case (no closed form) `nlmefit` with a
  SciPy ODE, the batched `ode_model` or a compiled SUND model reproduces
  MATLAB's ode45-based fit to ~1e-4 (solver noise), and Theophylline solved as
  an ODE reproduces its closed-form fit. With ODE models the Lindstrom-Bates
  iteration stops when changes only jitter at the solver's noise level.
- **SAEM for parameters without random effects**: MATLAB and saemix sample them
  during the annealing phase with a pseudo-variance shrinking by 10^-3 over
  that phase, then fix and optimise them. pynlme does the same; optimising
  them from the start (as pynlme first did) got stuck on the Vmax/Km ridge of
  the Michaelis-Menten model.
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
- **R `nlme(method = "REML")` is not a restricted-likelihood fit.** Its PNLS step
  updates the covariance by ML EM iterations and the reported covariance is that
  update, so even on a linear model `lme` REML gives a different psi. pynlme's
  default `"RELME"` is checked against `lme` REML (linear case) and MATLAB;
  `compat="r"` reproduces `nlme(method = "REML")` (`*_nlme_reml` references, ~1e-8).
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
- **Several fixed points: pynlme now takes R's path** (2026-10-04). Two simulated
  oral-PK datasets (seed 657630818 `oral_0`; seed 1007 `oral_7`) ended at a fixed point
  0.9-1.5 lower in exact log-likelihood than R's, although R's point was also a fixed
  point of pynlme's iteration. Tracing nlme (`verbose = TRUE`, and `pnlsMaxIter = k`
  for the inner iterations) showed why: nlme's PNLS step is not a converged penalised
  least-squares fit with fixed covariance but at most 7 Gauss-Newton steps, each
  preceded by an EM + nlm update of the covariance (keeping nlm's second-to-last
  iterate). `compat="r"` now follows this iteration (see `docs/implementation.md`);
  the traces agree to the printed digits, and the same mechanism makes nlme's REML
  differ from true REML. With the strict criterion (same log-likelihood to 1e-3 and
  estimates to 1e-4), 1598 of 1600 datasets (seeds 1001-1020 and 2001-2020,
  `PYNLME_RANDOM_N=10`; and 2399 of 2400 with seeds 3001-3020 added and the final
  code) gave R's fit, including both former failures and five cases
  where the old iteration had found a different (in three cases better) fixed point.
  The other two have a variance near zero that is not identified (same likelihood
  within 5e-4, e.g. 0.013 vs 0.019); where nlme's optimizers stop along it depends
  on rounding in R's numerical derivatives. Further random runs found one more kind
  of difference (seed 1770464656, `oral_1`): with a variance collapsing to zero, R's
  LME step (nlminb, not reimplemented) stops on the nearly flat likelihood after one
  iteration while pynlme's continues, ending 0.0017 higher in log-likelihood (likewise
  `lme` on a linear dataset with a zero intercept variance, seed 2009 `linear_7`,
  0.00015), or elsewhere on the ridge (seed 1005 `oral_9`, 0.0007 lower, 1 of 2400
  datasets). The test accepts a deviation only when pynlme's likelihood is higher, or
  when R's fit has a variance collapsed to zero and the likelihoods agree to 1e-3. The
  LME step in `compat="r"` keeps pynlme's restart of collapsed variances (kept only if
  it improves the likelihood): without it L-BFGS-B stalled at a zero variance on one
  linear dataset (seed 1502491072, 18.6 below `lme`), where nlminb does not. The default (MATLAB) mode keeps its own
  iteration and can still reach a different fixed point than R on such data.
- **MATLAB's iteration replicated** (2026-10-05). `compat="matlab"` follows nlmefit's
  alternating algorithm step by step (`matlab_lb.py`). On the reference cases it
  reproduces MATLAB's results at MATLAB's default options to ~1e-10, and its iterates
  agree with MATLAB's `OutputFcn` trace. The randomized comparison
  (`test_random_vs_matlab.py`, needs `PYNLME_MATLAB`; seeds 1001-1020 and 2001-2020,
  `PYNLME_RANDOM_N=10`, LME and RELME, 3200 fits): of the 2715 fits whose iteration never
  had a variance at zero, 2699 agree with MATLAB to <1e-8 in the fixed effects, 13 to
  1e-6 and 3 to MATLAB's TolX (1e-4). The latter are MATLAB's own sensitivity: its LME
  step runs fminsearch to 1e-2, and where it stops can flip with 1e-9 perturbations
  (rounding MATLAB's starting theta of seed 1007 `logistic_6` to 8 digits moves MATLAB's
  own estimates by 2e-5, onto pynlme's). In the other ~480 fits a variance reached zero
  (16% of simulated fits end there; pynlme warns, as the estimates are not well
  determined). There the likelihood is flat and rounding decides whether either program
  leaves the plateau: MATLAB's own result changes with the number of BLAS threads (seed
  1001 `logistic_7` RELME: -308.544 with one thread, as pynlme; -310.691 with four), and
  pynlme ended up to 2.1 higher or 1.2 lower than MATLAB in 13 of them.
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

## Comparison with nlmixr2

nlmixr2 is not a replicate target (its FOCEi is NONMEM's method, not MATLAB's), but it
is widely used for PK/PD, so `nlmixr2_compare.py` compares FOCE and SAEM with it on the
benchmark cases (nlmixr2 7.0.1, nlmixr2est 7.1.0, rxode2 5.1.7.1; same machine as the
README's speed tables; best of 3 runs after a warm-up). The programs report different
likelihood approximations, so all estimates are scored with the same exact
log-likelihood (importance sampling, `tests/validation/likelihood.py`; higher is
better, Monte Carlo error ~0.01). With an additive error model FOCEi is FOCE.

| Case | Method | Program | Time (ms) | Exact log-likelihood |
|---|---|---|---|---|
| indometh | FOCE | pynlme | 695 | 56.114 |
| indometh | FOCE | MATLAB `nlmefit` | 1447 | 56.115 |
| indometh | FOCE | nlmixr2 FOCEi | 1308 | 56.055 |
| theoph | FOCE | pynlme | 612 | -177.750 |
| theoph | FOCE | MATLAB `nlmefit` | 1722 | -178.446 |
| theoph | FOCE | nlmixr2 FOCEi | 323 | -177.756 |
| theoph | SAEM | pynlme `nlmefitsa` | 163 | -177.761 |
| theoph | SAEM | nlmixr2 SAEM | 1372 | -178.138 |
| orange | FOCE | pynlme | 95 | -131.575 |
| orange | FOCE | MATLAB `nlmefit` | 98 | -131.575 |
| orange | FOCE | nlmixr2 FOCEi | 281 | -131.576 |
| orange | SAEM | pynlme `nlmefitsa` | 109 | -131.593 |
| orange | SAEM | nlmixr2 SAEM | 614 | -158.405 |
| linear | FOCE | pynlme | 66 | -67.484 |
| linear | FOCE | MATLAB `nlmefit` | 137 | -67.484 |
| linear | FOCE | nlmixr2 FOCEi | 198 | -67.481 |
| linear | SAEM | pynlme `nlmefitsa` | 84 | -67.559 |
| linear | SAEM | nlmixr2 SAEM | – | failed: chol(): decomposition failed |

pynlme with its Rust kernels (the default).

- **FOCE**: the three programs reach the same optimum (within 0.06 in log-likelihood;
  MATLAB 0.7 lower on theoph). pynlme is the fastest on indometh, orange and linear
  (1.9-3 times faster than nlmixr2's FOCEi), nlmixr2 on theoph (1.9 times faster than
  pynlme).
- **SAEM**: on theoph (all parameters log-transformed) pynlme's SAEM is 8 times
  faster than nlmixr2's and ends higher. In our setup nlmixr2's SAEM left the parameters without
  a transform (indometh `tp1`/`tp3`, orange `tasym`, linear `ta`/`tb`) at their start
  values with their variances at zero, and failed on the linear case. We are not
  nlmixr2 experts: this may be a matter of how we specified the models (start values
  of the variances and the error are our guesses, see the script).

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

- The MATLAB documentation examples that need `orangetreedata.mat` and
  `nonlineardata.mat`.
