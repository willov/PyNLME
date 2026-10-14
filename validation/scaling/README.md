# Scaling benchmark

How the fitting time grows with the number of subjects (10, 100, 1000), for a
closed-form model and three ODE models (non-stiff, multiple doses, stiff), in pynlme,
MATLAB R2023b, R (`nlme`, `saemix`) and nlmixr2.

- **Data**: simulated by `make_data.py` (fixed seeds), the same files for every
  program. *oral*: one-compartment model with first-order absorption (as
  Theophylline), random effects on log ka and log CL. *mm*: oral dose with
  Michaelis-Menten elimination, an ODE without a closed form, random effects on log V
  and log Vmax. *mmd*: the mm model with doses at 0, 12 and 24 h. *tmdd*: target-mediated
  drug disposition (`validation/models/tmdd_sund.txt`), stiff (binding about 1000 times
  faster than elimination), log free drug observed, random effects on log kel and log
  kint. Additive error, the same start values everywhere.
- **Options**: every program with its defaults (nlmefit/nlmefitsa's, `nlmeControl()`,
  saemix's, nlmixr2's; pynlme's MATLAB mode follows MATLAB's defaults, its R mode
  nlme's).
- **ODE solvers**: every program at the same tolerance, relative 1e-8 and absolute
  1e-10, with its usual method for the model: for mm and mmd the non-stiff one (SUND
  Adams and SciPy DOP853 in pynlme, `ode45` in MATLAB, deSolve's `lsoda` in `nlme`,
  rxode2 in nlmixr2), for tmdd the stiff one (SUND BDF, SciPy LSODA, `ode15s`,
  `lsoda`, rxode2). At this setting the solvers are about equally accurate (table
  below). One rule for all programs: a fit that fails or ends at a wrong optimum at
  the common tolerance is run at a tighter one and marked † with its relative
  tolerance. This applies to pynlme's MATLAB mode and FOCE with SUND (at 1e-8 SUND's
  error, the largest on mm, spoils their forward-difference derivatives, and the mm
  fit ends 11 lower in log-likelihood), run at 1e-10, and R `nlme` on tmdd, whose
  PNLS step fails near the optimum at 1e-6 to 1e-10, run at 1e-12. At 1e-10 for all,
  `nlme` instead took a wrong path on mm, so no single tolerance suits every program.
- **Model calls**: the closed-form model is called for all subjects at once in pynlme
  and MATLAB (`Vectorization="Full"`); pynlme batches the ODE models over subjects
  (SUND containers, or one SciPy system).
- **Timing**: one machine (AMD Ryzen 7 7800X3D, 8 cores, 32 GB RAM, Linux; pynlme
  0.5.1 with its Rust kernels, Python 3.13, SciPy 1.18, SUND 2.5.3; MATLAB R2023b
  Update 1; R 4.6.1, nlme 3.1-169, saemix 3.5, nlmixr2 7.0.1, deSolve 1.42), idle
  during the runs (load logged). Runs under 5 s are the best of 3, the first run of
  each method is an untimed warm-up. Once a run takes over 60 s, the larger sizes are
  skipped ("skipped": they would take ten times longer); for SAEM only from 100
  subjects on, as on 10 subjects nlmefitsa runs several chains by default. MATLAB's
  FOCE on tmdd was stopped after 3.3 hours at 10 subjects, and on mmd not run.

## Solver accuracy

`solver_accuracy.py` (with `.R` and `.m`) solves 20 parameter sets of mm and tmdd
with each program's solver at the common tolerance and compares with a reference at
1e-13 (mm: largest error relative to the largest concentration; tmdd: largest error
in log free drug):

| Solver | mm (non-stiff) | tmdd (stiff) |
|---|---|---|
| SciPy (DOP853 / LSODA), pynlme | 1.9e-9 | 6.0e-7 |
| SUND (Adams / BDF), pynlme | 2.6e-8 | 1.2e-6 |
| deSolve `lsoda`, R `nlme` | 1.7e-8 | 7.2e-7 |
| rxode2, nlmixr2 | 1.7e-8 | 6.0e-7 |
| MATLAB (`ode45` / `ode15s`) | 2.1e-9 | 1.8e-6 |

## Results

Closed-form oral PK model (3 parameters, 2 random effects, 10 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode | 0.007 | 0.030 | 0.288 |
| LME | MATLAB `nlmefit` | 0.012 | 0.081 | 5.58 |
| LME | pynlme, R mode | 0.004 | 0.019 | 0.154 |
| LME | R `nlme` | 0.009 | 0.026 | 0.200 |
| FOCE | pynlme | 0.415 | 1.16 | 8.50 |
| FOCE | MATLAB `nlmefit` | 1.48 | 10 | 599 |
| FOCE | nlmixr2 (FOCEi) | 0.388 | 0.415 | 1.41 |
| SAEM | pynlme `nlmefitsa` | 0.274 | 0.281 | 1.22 |
| SAEM | MATLAB `nlmefitsa` | 0.600 | 0.948 | 3.02 |
| SAEM | pynlme `pynlme.r.saemix` | 0.265 | 0.607 | 4.06 |
| SAEM | R `saemix` | 1.26 | 2.81 | 24 |
| SAEM | nlmixr2 | 1.16 | 8.19 | 104 |

Log-likelihood each program reports, as a check that the fits are equally good (LME and FOCE; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode | -88.67 | -1040.13 | -10337.75 |
| LME | MATLAB `nlmefit` | -88.67 | -1040.13 | -10337.75 |
| LME | pynlme, R mode | -88.67 | -1040.13 | -10337.73 |
| LME | R `nlme` | -88.67 | -1040.13 | -10337.73 |
| FOCE | pynlme | -88.66 | -1040.04 | -10336.82 |
| FOCE | MATLAB `nlmefit` | -89.10 | -1047.07 | -10411.14 |
| FOCE | nlmixr2 (FOCEi) | -88.66 | -1040.04 | -10336.83 |

Michaelis-Menten ODE model (4 parameters, 2 random effects, 8 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 2.56 †1e-10 | 1.28 †1e-10 | 75 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 0.457 | 0.737 | 4.82 |
| LME | MATLAB `nlmefit` (ode45) | 0.390 | 3.49 | 47 |
| LME | pynlme, R mode (SUND) | 0.029 | 0.238 | 2.53 |
| LME | pynlme, R mode (SciPy) | 0.199 | 0.537 | 4.73 |
| LME | R `nlme` (deSolve) | 1.35 | 13 | 210 |
| FOCE | pynlme (SUND) | 19 †1e-10 | 177 †1e-10 | skipped |
| FOCE | pynlme (SciPy) | 119 | skipped | skipped |
| FOCE | MATLAB `nlmefit` (ode45) | 426 | skipped | skipped |
| FOCE | nlmixr2 (FOCEi, rxode2) | 0.881 | 1.86 | 14 |
| SAEM | pynlme `nlmefitsa` (SUND) | 8.92 | 12 | 100 |
| SAEM | pynlme `nlmefitsa` (SciPy) | 25 | 26 | 74 |
| SAEM | MATLAB `nlmefitsa` (ode45) | 135 | 140 | skipped |
| SAEM | nlmixr2 (rxode2) | 5.17 | 37 | 383 |

ODE tolerance relative 1e-8, absolute 1e-10 in every program; † marks a fit run at a tighter relative tolerance (shown), because it fails or misses the optimum at the common one.

Log-likelihood each program reports, as a check that the fits are equally good (LME and FOCE; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 1.11 †1e-10 | -33.91 †1e-10 | -360.26 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 1.10 | -33.91 | -360.28 |
| LME | MATLAB `nlmefit` (ode45) | 1.10 | -33.91 | -360.28 |
| LME | pynlme, R mode (SUND) | 1.10 | -33.90 | -360.27 |
| LME | pynlme, R mode (SciPy) | 1.10 | -33.91 | -360.27 |
| LME | R `nlme` (deSolve) | 1.11 | -33.91 | -360.23 |
| FOCE | pynlme (SUND) | 0.71 †1e-10 | -33.76 †1e-10 | – |
| FOCE | pynlme (SciPy) | 1.15 | – | – |
| FOCE | MATLAB `nlmefit` (ode45) | 1.03 | – | – |
| FOCE | nlmixr2 (FOCEi, rxode2) | 1.14 | -33.32 | -353.27 |

Michaelis-Menten ODE model with three doses (0, 12 and 24 h; SUND activities, nlmixr2 event table, ode45/deSolve from dose to dose; 15 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | failed | 22 †1e-10 | 511 †1e-10 |
| LME | MATLAB `nlmefit` (ode45) | failed | 31 | 217 |
| LME | pynlme, R mode (SUND) | 0.114 | 0.638 | 6.67 |
| LME | R `nlme` (deSolve) | 5.52 | 55 | 697 |
| FOCE | pynlme (SUND) | failed | 403 †1e-10 | skipped |
| FOCE | MATLAB `nlmefit` (ode45) | not run | not run | not run |
| FOCE | nlmixr2 (FOCEi, rxode2) | 1.40 | 5.09 | 36 |
| SAEM | pynlme `nlmefitsa` (SUND) | 29 | 30 | 293 |
| SAEM | MATLAB `nlmefitsa` (ode45) | 700 | 731 | skipped |
| SAEM | nlmixr2 (rxode2) | 13 | 111 | skipped |

ODE tolerance relative 1e-8, absolute 1e-10 in every program; † marks a fit run at a tighter relative tolerance (shown), because it fails or misses the optimum at the common one.

Log-likelihood each program reports, as a check that the fits are equally good (LME and FOCE; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | – | 271.09 †1e-10 | 2571.67 †1e-10 |
| LME | MATLAB `nlmefit` (ode45) | – | 271.09 | 2571.61 |
| LME | pynlme, R mode (SUND) | 29.87 | 270.97 | 2571.81 |
| LME | R `nlme` (deSolve) | 29.87 | 271.08 | 2571.60 |
| FOCE | pynlme (SUND) | – | 271.67 †1e-10 | – |
| FOCE | nlmixr2 (FOCEi, rxode2) | 29.92 | 271.67 | 2576.12 |

Stiff target-mediated drug disposition model (3 parameters, 2 random effects, 3 states, log free drug observed; each program's stiff solver: SUND BDF, SciPy LSODA, ode15s, lsoda; 10 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 0.305 †1e-10 | 8.88 †1e-10 | 331 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 3.43 | 29 | 453 |
| LME | MATLAB `nlmefit` (ode15s) | 8.27 | 426 | skipped |
| LME | pynlme, R mode (SUND) | 0.131 | 6.20 | 51 |
| LME | pynlme, R mode (SciPy) | 1.11 | 6.85 | 62 |
| LME | R `nlme` (deSolve) | 7.28 †1e-12 | 84 †1e-12 | skipped |
| FOCE | pynlme (SUND) | 67 †1e-10 | skipped | skipped |
| FOCE | pynlme (SciPy) | 1142 | skipped | skipped |
| FOCE | MATLAB `nlmefit` (ode15s) | > 3 h, stopped | not run | not run |
| FOCE | nlmixr2 (FOCEi, rxode2) | 2.12 | 11 | 122 |
| SAEM | pynlme `nlmefitsa` (SUND) | 107 | 136 | skipped |
| SAEM | pynlme `nlmefitsa` (SciPy) | 282 | 383 | skipped |
| SAEM | MATLAB `nlmefitsa` (ode15s) | 1739 | 1746 | skipped |
| SAEM | nlmixr2 (rxode2) | 36 | 228 | skipped |

ODE tolerance relative 1e-8, absolute 1e-10 in every program; † marks a fit run at a tighter relative tolerance (shown), because it fails or misses the optimum at the common one.

Log-likelihood each program reports, as a check that the fits are equally good (LME and FOCE; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 57.58 †1e-10 | 482.69 †1e-10 | 4966.71 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 57.58 | 482.68 | 4966.59 |
| LME | MATLAB `nlmefit` (ode15s) | 57.58 | 481.82 | – |
| LME | pynlme, R mode (SUND) | 57.58 | 482.59 | 4966.41 |
| LME | pynlme, R mode (SciPy) | 57.58 | 482.68 | 4966.70 |
| LME | R `nlme` (deSolve) | 57.58 †1e-12 | 482.68 †1e-12 | – |
| FOCE | pynlme (SUND) | 57.57 †1e-10 | – | – |
| FOCE | pynlme (SciPy) | 57.58 | – | – |
| FOCE | nlmixr2 (FOCEi, rxode2) | 53.72 | 482.31 | 4890.82 |

## Findings

At the largest size each program completed:

- **Closed-form model**: for LME pynlme's MATLAB mode is 19 times faster than MATLAB
  and its R mode 1.3 times faster than R `nlme`; for SAEM pynlme is the fastest (2.5
  times faster than MATLAB, `pynlme.r.saemix` 6 times faster than saemix, 85 times
  faster than nlmixr2). For FOCE pynlme is 70 times faster than MATLAB, but nlmixr2's
  FOCEi is 6 times faster than pynlme (same optimum).
- **ODE models, LME**: pynlme's R mode is the fastest on all three: 2.5 s on mm at
  1000 subjects (R `nlme` 210 s, MATLAB 47 s), 6.7 s on mmd (`nlme` 697 s, MATLAB 217
  s), 6.2 s on tmdd at 100 (`nlme` 84 s at 1e-12, MATLAB 426 s). The MATLAB mode is
  fast with SciPy (4.8 s on mm, 10 times MATLAB), but with SUND, which it needs at
  1e-10, it is slower than MATLAB on mm (75 s) and mmd (511 s against 217 s). On the
  stiff tmdd model it is 48 times faster than MATLAB at 100 subjects (8.9 s).
- **ODE models, SAEM**: from 100 subjects on pynlme is the fastest (mm 74 s at 1000
  against 383 s for nlmixr2; mmd 30 s against 111 s at 100; tmdd 136 s against 228 s
  at 100) and 12 to 24 times faster than MATLAB. At 10 subjects nlmixr2 is faster (5 s
  against 9 s on mm, 36 s against 107 s on tmdd): with few subjects nlmefitsa's
  default runs several chains.
- **ODE models, FOCE**: nlmixr2 is far ahead (14 s on mm, 36 s on mmd, 122 s on tmdd,
  all at 1000 subjects). It computes the model's derivatives from sensitivity
  equations and its outer gradient analytically; pynlme and MATLAB use finite
  differences, which re-solve every subject's conditional modes for each parameter.
  This is pynlme's main gap (SUND sensitivities will allow the same). MATLAB's FOCE
  took 426 s on mm at 10 subjects (pynlme 19 s) and did not finish tmdd at 10
  subjects in 3.3 hours (pynlme 67 s); we did not investigate why.
- **SUND or SciPy**: SUND is faster on the stiff model (LME 0.13 s against 1.1 s at
  10 subjects, SAEM 107 s against 282 s; the gap narrows to 1.2-1.4 times at 1000 subjects), and
  it handles dosing events (mmd). On the
  non-stiff mm model SciPy is faster for SAEM (74 s against 100 s at 1000) and the
  MATLAB mode (where SUND needs the tighter tolerance), SUND for the R mode (2.5 s
  against 4.7 s).
- **mmd at 10 subjects**: MATLAB's pre-fit drives Km to 0, where the model is not
  finite, so MATLAB and pynlme's MATLAB mode both stop there ("MYFUN has returned Inf
  or NaN"); the R mode fits.
- **Fit check**: within a method all programs reach the same fit (log-likelihood
  tables), except MATLAB's FOCE, which ends lower, and nlmixr2's FOCEi on tmdd, which
  reports a lower value (its FOCEi likelihood is a different approximation).

We are not experts in tuning MATLAB, R or nlmixr2: they run as a typical user would
run them, and could likely be made faster.

## Running

```bash
uv run python validation/scaling/make_data.py      # the datasets (data/, not in git)
uv run python validation/scaling/run_pynlme.py     # results/pynlme.json
Rscript validation/scaling/run_r.R                 # results/r.json (nlme, saemix, nlmixr2, deSolve)
# in MATLAB, from the repository root: addpath validation/scaling; run_matlab
uv run python validation/scaling/report.py         # the tables above
# solver accuracy: solver_accuracy.py write, solver_accuracy.R, solver_accuracy.m, solver_accuracy.py
```

`SCALING_SIZES`, `SCALING_MODELS` (R) and `SCALING_METHODS` (MATLAB) select parts.
pynlme and R take about two hours together; MATLAB about three hours without its FOCE
on mmd and tmdd, which takes many hours.
