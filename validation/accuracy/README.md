# Accuracy benchmark

How close each program's estimates come to the true values, and whether its standard
errors are right: per scenario 100 datasets simulated from the same known parameters,
each fitted
by pynlme, MATLAB R2023b (`nlmefit`, `nlmefitsa`), R `nlme`, `saemix` and nlmixr2.

- **Scenarios** (the one-compartment oral model: log ke, log ka, log CL):
  *basic*: random effects on log ka and log CL (variances 0.16 and 0.04), additive
  error (SD 0.5), 50 subjects with 10 observations. *hard*: random effects on all
  three parameters with a full covariance (correlations 0.3 and -0.2), log(wt/70) on
  log CL (effect 0.75), proportional error (0.15), 60 subjects with 10 observations.
- **Options**: every program with its defaults and the same start values, except
  that standard errors are requested from `nlmefitsa` (`ComputeStdErrors`; saemix and
  nlmixr2 compute them by default). The covariate effect starts at 0.5 everywhere:
  from 0, nlmixr2's FOCEi left it at 0. nlme's proportional error is
  `varPower(fixed = 1)`; pynlme's R mode follows nlme's iteration only for constant
  and exponential errors, so in the hard scenario it is pynlme's own LME. Different
  seeds per replicate for the stochastic methods.
- **Measures**: for each fixed effect the bias and RMSE over the replicates, and the
  coverage of the 95% confidence interval (estimate +- 1.96 SE: about 95% if the
  standard errors are right; with 100 replicates, 91-99% is within chance); for the
  variances and the error parameter the relative bias.

**Basic: random effects on log ka and log CL, additive error** (100 replicates of 50 subjects). Fixed effects: bias / RMSE / coverage of the 95% CI; variances and error parameter: relative bias.

| Method | Program | Fits | log ke | log ka | log CL | omega2 ka | omega2 CL | error SD |
|---|---|---|---|---|---|---|---|---|
| LME | pynlme, MATLAB mode | 100 | +0.006 / 0.019 / 95% | +0.004 / 0.051 / 95% | +0.004 / 0.030 / 97% | -4% | -3% | -0.1% |
| LME | MATLAB `nlmefit` | 100 | +0.006 / 0.019 / 95% | +0.004 / 0.051 / 95% | +0.004 / 0.030 / 97% | -4% | -3% | -0.1% |
| LME | pynlme, R mode | 100 | +0.006 / 0.019 / 95% | +0.004 / 0.051 / 95% | +0.004 / 0.030 / 97% | -4% | -3% | -0.1% |
| LME | R `nlme` | 100 | +0.006 / 0.019 / 95% | +0.004 / 0.051 / 95% | +0.004 / 0.030 / 97% | -4% | -3% | -0.1% |
| FOCE | pynlme | 100 | +0.001 / 0.018 / 98% | +0.010 / 0.052 / 95% | +0.003 / 0.030 / 99% | -3% | -2% | -0.2% |
| FOCE | MATLAB `nlmefit` | 100 | +0.001 / 0.018 / 98% | +0.002 / 0.053 / 94% | +0.082 / 0.088 / 26% | -4% | +15% | -0.1% |
| FOCE | nlmixr2 (FOCEi) | 100 | +0.001 / 0.018 / 95% | +0.010 / 0.052 / 96% | +0.003 / 0.030 / 97% | -3% | -2% | -0.2% |
| SAEM | pynlme, MATLAB mode | 100 | +0.003 / 0.019 / 94% | +0.010 / 0.053 / 97% | +0.004 / 0.031 / 97% | -3% | -2% | -0.1% |
| SAEM | MATLAB `nlmefitsa` | 100 | +0.000 / 0.020 / 93% | +0.010 / 0.053 / 95% | +0.002 / 0.031 / 97% | -4% | -4% | -0.5% |
| SAEM | pynlme, saemix mode | 100 | +0.005 / 0.021 / 91% | +0.009 / 0.053 / 96% | +0.006 / 0.032 / 93% | -3% | -2% | -0.1% |
| SAEM | R `saemix` | 100 | +0.004 / 0.021 / 93% | +0.009 / 0.055 / 95% | +0.005 / 0.031 / 96% | -3% | -2% | -0.1% |
| SAEM | nlmixr2 | 100 | -0.047 / 0.053 / 36% | +0.043 / 0.064 / 92% | -0.028 / 0.039 / 85% | -3% | -3% | +1.2% |

**Hard: random effects on all three parameters with a full covariance, log(wt/70) on log CL, proportional error** (100 replicates of 60 subjects). Fixed effects: bias / RMSE / coverage of the 95% CI; variances and error parameter: relative bias.

| Method | Program | Fits | log ke | log ka | log CL | wt on CL | omega2 ke | omega2 ka | omega2 CL | prop. error |
|---|---|---|---|---|---|---|---|---|---|---|
| LME | pynlme, MATLAB mode | 100 | +0.019 / 0.033 / 86% | -0.033 / 0.070 / 89% | +0.017 / 0.035 / 94% | +0.013 / 0.157 / 94% | -6% | -6% | -5% | +1.3% |
| LME | MATLAB `nlmefit` | 100 | +0.019 / 0.033 / 86% | -0.033 / 0.070 / 89% | +0.017 / 0.035 / 94% | +0.014 / 0.157 / 94% | -6% | -7% | -5% | +1.2% |
| LME | pynlme, R mode | 100 | +0.019 / 0.033 / 87% | -0.033 / 0.069 / 89% | +0.017 / 0.035 / 94% | +0.012 / 0.152 / 96% | -4% | -8% | -5% | +1.1% |
| LME | R `nlme` | 100 | +0.019 / 0.033 / 87% | -0.033 / 0.069 / 89% | +0.017 / 0.035 / 94% | +0.013 / 0.152 / 96% | -3% | -7% | -4% | +0.9% |
| FOCE | pynlme | 100 | +0.013 / 0.030 / 91% | -0.027 / 0.067 / 92% | +0.020 / 0.037 / 93% | +0.012 / 0.152 / 96% | -9% | -11% | -5% | +2.5% |
| FOCE | MATLAB `nlmefit` | 100 | -0.038 / 0.061 / 62% | -0.190 / 0.218 / 24% | +0.313 / 0.322 / 0% | +0.009 / 0.168 / 93% | +9% | +13% | +144% | +1.6% |
| FOCE | nlmixr2 (FOCEi) | 100 | +0.014 / 0.030 / 93% | -0.027 / 0.067 / 90% | +0.020 / 0.037 / 93% | +0.015 / 0.150 / 96% | -2% | -6% | -4% | +0.2% |
| SAEM | pynlme, MATLAB mode | 100 | +0.001 / 0.027 / 97% | -0.008 / 0.063 / 92% | +0.005 / 0.032 / 95% | +0.015 / 0.150 / 97% | -1% | -4% | -4% | +0.1% |
| SAEM | MATLAB `nlmefitsa` | 100 | +0.002 / 0.028 / 95% | -0.009 / 0.064 / 92% | +0.006 / 0.032 / 95% | +0.016 / 0.156 / 97% | -2% | -5% | -4% | +0.1% |
| SAEM | pynlme, saemix mode | 100 | +0.001 / 0.027 / 97% | -0.007 / 0.064 / 93% | +0.006 / 0.032 / 95% | +0.015 / 0.151 / 94% | -2% | -4% | -4% | +0.3% |
| SAEM | R `saemix` | 100 | +0.002 / 0.028 / 96% | -0.009 / 0.062 / 94% | +0.006 / 0.032 / 95% | +0.012 / 0.151 / 95% | -2% | -5% | -4% | +0.3% |
| SAEM | nlmixr2 | 100 | +0.002 / 0.027 / 97% | -0.008 / 0.063 / 90% | +0.005 / 0.032 / 95% | +0.014 / 0.151 / 95% | -1% | -4% | -4% | +0.1% |

## Findings

- **LME**: pynlme's MATLAB mode gives MATLAB's estimates and standard errors, its R
  mode R `nlme`'s (in the hard scenario too, though there it is not nlme's
  iteration). In the basic scenario the bias is small and coverage 95-97%; in the
  hard one all LME fits share the linearisation's bias (log ke +0.02, coverage
  86-89% for log ke and log ka): a limitation of the method, not of a program.
- **FOCE**: pynlme and nlmixr2's FOCEi agree, with coverage 90-99%. MATLAB's FOCE is
  biased: for log CL by +0.08 (coverage 26%) in the basic scenario, and by +0.31
  (coverage 0%, CL variance +144%) in the hard one, consistent with its lower
  log-likelihoods in the other benchmarks.
- **SAEM**: the most accurate method. pynlme (both modes), MATLAB and saemix agree in
  both scenarios (coverage 91-97%). nlmixr2's SAEM agrees in the hard scenario but is
  biased for log ke in the basic one (coverage 36%), where log ke is the only fixed
  effect without a random effect; this may be how we specified the model (we are not
  nlmixr2 experts).

## Running

```bash
uv run python validation/accuracy/simulate.py     # both scenarios (data/, not in git)
uv run python validation/accuracy/run_pynlme.py   # results/<scenario>/pynlme.json
Rscript validation/accuracy/run_r.R               # results/<scenario>/r.json (nlme, saemix, nlmixr2)
# in MATLAB, from the repository root: addpath validation/accuracy; run_matlab
uv run python validation/accuracy/report.py       # the table above
```

Scripts take scenario names as arguments (default both). The hard scenario takes
about an hour per program.
