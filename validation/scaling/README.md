# Scaling benchmark

How the fitting time grows with the number of subjects (10, 100, 1000), for a
closed-form model and an ODE model, in pynlme, MATLAB R2023b, R (`nlme`, `saemix`)
and nlmixr2.

- **Data**: simulated by `make_data.py` (fixed seeds), the same files for every
  program. *oral*: one-compartment model with first-order absorption (as
  Theophylline), random effects on log ka and log CL. *mm*: oral dose with
  Michaelis-Menten elimination, an ODE without a closed form, random effects on log V
  and log Vmax. Additive error, the same start values everywhere.
- **Options**: every program with its defaults (nlmefit/nlmefitsa's, `nlmeControl()`,
  saemix's, nlmixr2's; pynlme's MATLAB mode follows MATLAB's defaults, its R mode
  nlme's). The ODE is solved at a common tolerance (relative 1e-8, absolute 1e-10): by
  `ode45` per subject in MATLAB, deSolve's `lsoda` per subject in `nlme`, rxode2 in
  nlmixr2, SciPy in pynlme; SUND keeps its tighter defaults (1e-10, 1e-12), because at
  1e-8 its solver error spoiled the MATLAB mode's forward-difference derivatives (the
  fit at 1000 subjects ended 115 lower in log-likelihood). This is conservative
  against pynlme.
- **Model calls**: the closed-form model is called for all subjects at once in pynlme
  and MATLAB (`Vectorization="Full"`); pynlme batches the SUND model over subjects.
- **Timing**: one machine (AMD Ryzen 7 7800X3D, 8 cores, 32 GB RAM, Linux; pynlme
  0.5.0 with its Rust kernels, Python 3.13; MATLAB R2023b Update 1; R 4.6.1, nlme
  3.1-169, saemix 3.5, nlmixr2 7.0.1, deSolve). Runs under 5 s are the best of 3, the
  first run of each method is an untimed warm-up. Once a run takes over 60 s, the
  larger sizes are skipped ("skipped": they would take ten times longer).

Closed-form oral PK model (3 parameters, 2 random effects, 10 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode | 0.008 | 0.042 | 0.417 |
| LME | MATLAB `nlmefit` | 0.012 | 0.081 | 5.58 |
| LME | pynlme, R mode | 0.006 | 0.031 | 0.277 |
| LME | R `nlme` | 0.009 | 0.026 | 0.200 |
| FOCE | pynlme | 0.404 | 1.47 | 9.96 |
| FOCE | MATLAB `nlmefit` | 1.48 | 10 | 599 |
| FOCE | nlmixr2 (FOCEi) | 0.388 | 0.415 | 1.41 |
| SAEM | pynlme `nlmefitsa` | 0.298 | 0.319 | 1.79 |
| SAEM | MATLAB `nlmefitsa` | 0.600 | 0.948 | 3.02 |
| SAEM | pynlme `pynlme.r.saemix` | 0.248 | 0.636 | 4.58 |
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
| LME | pynlme, MATLAB mode (SUND) | 0.249 | 1.67 | 42 |
| LME | pynlme, MATLAB mode (SciPy) | 1.06 | 6.78 | 66 |
| LME | MATLAB `nlmefit` (ode45) | 0.388 | 3.49 | 49 |
| LME | pynlme, R mode (SUND) | 0.121 | 1.12 | 11 |
| LME | R `nlme` (deSolve) | 1.34 | 12 | 206 |
| FOCE | pynlme (SUND) | 33 | 437 | skipped |
| FOCE | MATLAB `nlmefit` (ode45) | 431 | skipped | skipped |
| FOCE | nlmixr2 (FOCEi, rxode2) | 0.847 | 1.86 | 14 |
| SAEM | pynlme `nlmefitsa` (SUND) | 30 | 34 | 319 |
| SAEM | MATLAB `nlmefitsa` (ode45) | 137 | skipped | skipped |
| SAEM | nlmixr2 (rxode2) | 5.04 | 36 | 379 |

Log-likelihood each program reports, as a check that the fits are equally good (LME and FOCE; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 1.11 | -33.91 | -360.24 |
| LME | pynlme, MATLAB mode (SciPy) | 1.10 | -33.91 | -360.28 |
| LME | MATLAB `nlmefit` (ode45) | 1.10 | -33.91 | -360.28 |
| LME | pynlme, R mode (SUND) | 1.10 | -33.91 | -360.27 |
| LME | R `nlme` (deSolve) | 1.11 | -33.91 | -360.23 |
| FOCE | pynlme (SUND) | 1.15 | -33.32 | – |
| FOCE | MATLAB `nlmefit` (ode45) | 1.03 | – | – |
| FOCE | nlmixr2 (FOCEi, rxode2) | 1.14 | -33.32 | -353.27 |

## Findings

- **Closed-form model**: for LME pynlme's MATLAB mode is 13 times faster than MATLAB
  at 1000 subjects, while R `nlme` is slightly faster than pynlme's R mode (0.20 vs
  0.28 s). For SAEM pynlme is the fastest (1.7 times faster
  than MATLAB, 5 times faster than saemix, 58 times faster than nlmixr2 at 1000).
  For FOCE pynlme is 60 times faster than MATLAB at 1000 subjects, but nlmixr2's
  FOCEi is 7 times faster than pynlme (same optimum).
- **ODE model, LME**: pynlme with SUND is the fastest: its R mode is 18 times faster
  than R `nlme` with deSolve at 1000 subjects, its MATLAB mode 1.2-2 times faster than
  MATLAB with `ode45` (with SciPy's solver pynlme is slower than MATLAB).
- **ODE model, FOCE**: nlmixr2 is far ahead (14 s at 1000 subjects; pynlme needs 437
  s at 100, MATLAB 431 s at 10). nlmixr2 computes the model's derivatives from
  sensitivity equations and its outer gradient analytically; pynlme and MATLAB use
  finite differences, which re-solve every subject's conditional modes for each
  parameter. This is pynlme's main gap.
- **ODE model, SAEM**: pynlme is 4.5 times faster than MATLAB at 10 subjects; against
  nlmixr2 it is 6 times slower at 10 subjects (with few subjects nlmefitsa's default
  runs several chains), equal at 100 and 1.2 times faster at 1000.
- All programs reach the same fit within a method (log-likelihood tables), except
  MATLAB's FOCE, which ends lower.

We are not experts in tuning MATLAB, R or nlmixr2: they run as a typical user would
run them, and could likely be made faster.

## Running

```bash
uv run python validation/scaling/make_data.py      # the datasets (data/, not in git)
uv run python validation/scaling/run_pynlme.py     # results/pynlme.json
Rscript validation/scaling/run_r.R                 # results/r.json (nlme, saemix, nlmixr2, deSolve)
# in MATLAB, from the repository root: addpath validation/scaling; run_matlab
uv run python validation/scaling/report.py         # the tables above
```

The whole run takes about two hours, most of it the ODE fits at 1000 subjects.
