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
| LME | pynlme, MATLAB mode (SUND) | 0.218 | 1.20 | 35 |
| LME | pynlme, MATLAB mode (SciPy) | 0.441 | 0.733 | 4.83 |
| LME | MATLAB `nlmefit` (ode45) | 0.388 | 3.49 | 49 |
| LME | pynlme, R mode (SUND) | 0.106 | 0.717 | 7.54 |
| LME | R `nlme` (deSolve) | 1.34 | 12 | 206 |
| FOCE | pynlme (SUND) | 33 | 385 | skipped |
| FOCE | MATLAB `nlmefit` (ode45) | 431 | skipped | skipped |
| FOCE | nlmixr2 (FOCEi, rxode2) | 0.847 | 1.86 | 14 |
| SAEM | pynlme `nlmefitsa` (SUND) | 30 | 30 | 285 |
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

At 1000 subjects:

- **Closed-form model**: pynlme is the fastest for LME (its MATLAB mode 19 times
  faster than MATLAB, its R mode 1.3 times faster than R `nlme`) and for SAEM (2.5
  times faster than MATLAB, `pynlme.r.saemix` 6 times faster than saemix, 85 times
  faster than nlmixr2). For FOCE pynlme is 70 times faster than MATLAB, but nlmixr2's
  FOCEi is 6 times faster than pynlme (same optimum).
- **ODE model, LME**: pynlme is the fastest: its R mode (SUND) 27 times faster than R
  `nlme` with deSolve; its MATLAB mode 10 times faster than MATLAB with `ode45` when
  using SciPy's batched solver (`ode_model`, all subjects as one vectorised system),
  1.4 times with SUND. For many subjects `ode_model` beats SUND's batches of 16; for
  10 subjects SUND is faster.
- **ODE model, FOCE**: nlmixr2 is far ahead (14 s at 1000 subjects; pynlme needs 385
  s at 100, MATLAB 431 s at 10). nlmixr2 computes the model's derivatives from
  sensitivity equations and its outer gradient analytically; pynlme and MATLAB use
  finite differences, which re-solve every subject's conditional modes for each
  parameter. This is pynlme's main gap (SUND sensitivities will allow the same).
- **ODE model, SAEM**: pynlme is 4.5 times faster than MATLAB at 10 subjects; against
  nlmixr2 it is 6 times slower at 10 subjects (with few subjects nlmefitsa's default
  runs several chains) and 1.2-1.3 times faster at 100 and 1000.
- All programs reach the same fit within a method (log-likelihood tables), except
  MATLAB's FOCE, which ends lower.
- Not yet measured: SciPy's solver for FOCE, SAEM and the R mode at 100 and 1000
  subjects (a run was stopped because the machine was busy). FOCE with SciPy took 134
  s at 10 subjects, so SUND remains the better choice there.

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
