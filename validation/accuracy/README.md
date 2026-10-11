# Accuracy benchmark

How close each program's estimates come to the true values, and whether its standard
errors are right: 100 datasets simulated from the same known parameters, each fitted
by pynlme, MATLAB R2023b (`nlmefit`, `nlmefitsa`), R `nlme`, `saemix` and nlmixr2.

- **Model**: the closed-form oral model of the scaling benchmark (log ke, log ka, log
  CL; random effects on log ka and log CL, variances 0.16 and 0.04; additive error, SD
  0.5), 50 subjects with 10 observations each.
- **Options**: every program with its defaults and the same start values, except
  that standard errors are requested from `nlmefitsa` (`ComputeStdErrors`; saemix and
  nlmixr2 compute them by default). Different seeds per replicate for the stochastic
  methods.
- **Measures**: for each fixed effect the bias and RMSE over the replicates, and the
  coverage of the 95% confidence interval (estimate +- 1.96 SE: about 95% if the
  standard errors are right; with 100 replicates, 91-99% is within chance); for the
  variances and the error SD the relative bias.

100 replicates of 50 subjects. Fixed effects: bias / RMSE / coverage of the 95% CI; variances and error SD: relative bias.

| Method | Program | Fits | log ke | log ka | log CL | omega2 ka | omega2 CL | sigma |
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

## Findings

- **LME**: pynlme (both modes), MATLAB and R `nlme` give the same estimates and
  standard errors (the replicate modes follow these programs step by step); small
  bias, coverage 95-97%. The variances are 3-4% low, the usual downward bias of
  maximum likelihood.
- **FOCE**: pynlme and nlmixr2's FOCEi agree, with coverage 95-99%. MATLAB's FOCE is
  biased for log CL (+0.08, coverage 26%), consistent with its lower log-likelihoods
  in the other benchmarks.
- **SAEM**: pynlme (both modes), MATLAB and saemix agree, coverage 91-97%. nlmixr2's
  SAEM is biased for log ke (-0.05, coverage 36%), the one fixed effect without a
  random effect; this may be how we specified the model (we are not nlmixr2
  experts), as its FOCEi on the same model is unbiased.

## Running

```bash
uv run python validation/accuracy/simulate.py     # 100 datasets (data/, not in git)
uv run python validation/accuracy/run_pynlme.py   # results/pynlme.json
Rscript validation/accuracy/run_r.R               # results/r.json (nlme, saemix, nlmixr2)
# in MATLAB, from the repository root: addpath validation/accuracy; run_matlab
uv run python validation/accuracy/report.py       # the table above
```

The fits take about ten minutes for pynlme, half an hour for R and MATLAB each.
