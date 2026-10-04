# pynlme: nonlinear mixed-effects models in Python

Fit nonlinear mixed-effects models (population PK/PD, growth curves, systems-biology
models) in Python. The algorithms are those of MATLAB's `nlmefit`/`nlmefitsa` and R's
`nlme`/`saemix`, and the results are validated against all of them.

## AI Disclaimer

I (willov) asked Claude Sonnet 4 in GitHub Copilot agent mode to "create a Python NLME implementation with a rust backend with a similar call syntax to MATLAB's nlmefitsa". Please use the package with caution, it is very likely that the AI have hallucinated quirks not tested. If you find it useful, please assist in creating tests to test the reliability of the package.

## Pick your starting point

### Coming from MATLAB

Same functions, same arguments, same outputs, same option names:

```python
import numpy as np
from pynlme import nlmefit, nlmefitsa

# MATLAB: model = @(phi,t)(phi(1).*exp(-phi(2).*t)+phi(3).*exp(-phi(4).*t));
def model(phi, t):
    return phi[0] * np.exp(-phi[1] * t) + phi[2] * np.exp(-phi[3] * t)

beta, psi, stats, b = nlmefit(time, concentration, subject, [], model, [1, 1, 1, 1],
                              ParamTransform=[0, 1, 0, 1])
```

Indices in MATLAB-named options are 1-based (`REParamsSelect=[2, 3]`); `b` is r-by-m;
`stats` has MATLAB's fields and definitions. Results match MATLAB R2023b to ~1e-6.
Start with the [tutorial](docs/tutorial.md).

### Python users

Named parameters, data as a dict of arrays or a Polars DataFrame, results as a dict:

```python
from pynlme import fit

def model(p, x, v):
    ke, ka, cl, t = p["ke"], p["ka"], p["cl"], x["time"]
    return v["dose"] * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

result = fit(model, data, response="conc", group="subject", predictors=["time"],
             group_predictors=["dose"], start={"ke": 0.08, "ka": 1.6, "cl": 0.04},
             transform={"ke": "log", "ka": "log", "cl": "log"}, random=["ka", "cl"])
result["typical"]["cl"], result["psi"]["variances"], result["stats"]["aic"]
```

The model receives one value per observation, so it is evaluated for all subjects at
once. For arrays and positional parameters, `fit_nlme` takes snake_case options with
0-based indices. See the [API reference](docs/api_reference.md).

### R users (nlme, saemix) who would rather not use R

`pynlme.r` takes R's `nlme()` call almost verbatim, with formulas as strings, and
prints R's `summary()`:

```python
import polars as pl
from pynlme.r import nlme, pdDiag, summary, fixef, ranef, VarCorr

theo = pl.read_csv("theoph.csv").with_columns(wtc=pl.col("wt") - 70)
fit = nlme("conc ~ SSfol(dose, time, lKe, lKa, lCl)", data=theo,
           fixed=["lKe ~ 1", "lKa ~ 1", "lCl ~ wtc"],
           random=pdDiag("lKa + lCl ~ 1"), groups="~subject",
           start=[-2.5, 0.5, -3.2, 0])
print(summary(fit))      # same layout and numbers as R
fixef(fit)["lCl.wtc"]; ranef(fit)[1]; VarCorr(fit)["Residual"]["StdDev"]
```

`fixef`, `ranef`, `VarCorr`, `vcov`, `logLik`, `AIC`, `BIC`, `fitted`, `resid`,
`predict(fit, newdata, level=)` work as in R; the fit itself is a dict. R's defaults are kept (`method="ML"`, `pdSymm` for a
plain `random` formula), and the results match R `nlme` to ~1e-4 or closer. Not yet
supported: `weights=` (use `error_model=`), random effects with covariates, and
transformed covariates in `fixed` (compute a column instead). See
[the API reference](docs/api_reference.md#r-style-pynlmer).

saemix users write the same three calls as in R, with the model in R's
`psi[id, k]` form:

```python
from pynlme.r import saemixData, saemixModel, saemix, summary, coef, psi, predict

def model(psi, id, xidep):
    dose, t = xidep[:, 0], xidep[:, 1]
    ka, V, CL = psi[id, 0], psi[id, 1], psi[id, 2]
    k = CL / V
    return dose * ka / (V * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))

data = saemixData(theo, name_group="subject", name_predictors=["dose", "time"],
                  name_response="conc", name_covariates=["wt"])
smodel = saemixModel(model, psi0={"ka": [1, 0], "V": [20, 0], "CL": [0.5, 0]},
                     transform_par=[1, 1, 1], covariate_model=[[0, 0, 1]],
                     covariance_model=np.diag([1, 0, 1]))
fit = saemix(smodel, data, {"seed": 632545})
print(summary(fit))            # saemix's layout: estimates, SE, CV(%), omega2, -2LL/AIC/BIC
psi(fit)[1]; predict(fit, type="ppred")
```

saemix's defaults are kept (one chain, `nbiter.saemix = c(300, 100)`, importance
sampling, standard errors); results agree with R saemix within Monte Carlo error.

Lower-level equivalents of R's arguments:

| R | pynlme |
|---|---|
| `nlme(..., method="ML")` / `"REML"` | `nlmefit(..., ApproximationType="LME")` / `"RELME"`, `compat="r"` |
| `fixed = list(a ~ 1, b ~ wt)` | `fit(..., covariates={"b": ["wt"]})` or `FEGroupDesign` |
| `random = pdDiag(a ~ 1)` / `pdSymm(...)` | `REParamsSelect`, `CovPattern` diagonal / full |
| `saemix`: `transform.par`, `covariance.model` | `ParamTransform` (same codes), `CovPattern` |
| `saemix(...)` | `nlmefitsa(..., compat="r")` (one chain, as saemix) |

`compat="r"` follows nlme's iteration step by step, so pynlme returns R's own fit even
where the model has several local solutions, and R's `method="REML"` (which for
nonlinear models is not a restricted-likelihood fit). The default `compat="matlab"`
RELME is a true REML fit. See [docs/troubleshooting.md](docs/troubleshooting.md).

## What is validated

Every claim here is a test in [tests/validation/](tests/validation/); the details and
findings are in [validation/README.md](validation/README.md).

- **`nlmefit`**: matches MATLAB R2023b on every output (estimates, covariance,
  log-likelihood, AIC/BIC, standard errors, error parameters, all residual types,
  random effects) for LME and RELME with all four error models. FOCE reaches MATLAB's
  optimum or a better one; FO reaches far better optima than MATLAB's (whose FO
  collapses the variances to zero). Matches R `nlme` on fixed datasets, and on 640
  freshly simulated datasets reaches R's solution or a better one in all but one case.
- **`nlmefitsa`** (SAEM): agrees with MATLAB and R `saemix` within Monte Carlo error;
  reaches the exact maximum likelihood on a linear model; removes the linearisation
  bias of `nlmefit` in simulation studies.
- **ODE models** (SciPy, or compiled with SUND): reproduce MATLAB's `ode45`-based fits.

Supported: `ApproximationType` LME/RELME/FO/FOCE; error models constant, proportional,
combined, exponential; `ParamTransform`; fixed- and random-effects designs including
covariate models; diagonal or full covariance; SAEM options and log-likelihoods. Not
yet: `FEObsDesign`/`REObsDesign`, block `CovPattern`, `LogLikMethod="gq"`,
`Replicates`. Unsupported options raise errors; none are silently ignored.

## Speed

Pure Python/NumPy. With `Vectorization="Full"` (`uv run python validation/benchmark.py`):

| | pynlme | R | MATLAB R2023b |
|---|---|---|---|
| `nlmefit` LME | 0.006-0.056 s | `nlme` 0.007-0.044 s | 0.02-0.23 s |
| `nlmefit` FOCE | 0.4-3.7 s | – | 14-49 s |
| `nlmefitsa`, MATLAB's chains | 0.23-0.57 s | – | 1.3-3.7 s |
| `nlmefitsa`, 1 chain | 0.15-0.34 s | `saemix` 0.45-1.32 s | – |

With the default `Vectorization="SinglePhi"` the model is called once per subject:
same results, several times slower.

## ODE models

- **SUND** (recommended): compile the model with the
  [SUND toolbox](https://isbgroup.eu/sund-toolbox) and use
  `pynlme.sund_model("model_name", feature, parameters, v_states=[...])`
  (`pip install "pynlme[sund]"`); see [examples/sund_ode_model.py](examples/sund_ode_model.py).
- **SciPy**: `pynlme.ode_model(rhs, y0, output)` with a vectorised right-hand side.

On the Michaelis-Menten validation model: `nlmefit` takes 1.0 s with SUND (batched with
containers), 3.6 s with batched SciPy, 3-5 s in MATLAB with `ode45`. Analytic
derivatives (`pynlme.with_jacobian`) replace finite differences when available.

## Installation

```bash
pip install git+https://github.com/willov/PyNLME          # or a wheel from Releases
pip install "pynlme[sund] @ git+https://github.com/willov/PyNLME"   # with SUND models
```

For development: `git clone`, then `uv sync --dev` and `uv run pytest`
(`--runslow` adds the simulation studies; the randomized comparison needs R with
`nlme` and `jsonlite`).

## Documentation

[Tutorial](docs/tutorial.md) · [API reference](docs/api_reference.md) ·
[Troubleshooting](docs/troubleshooting.md) · [Implementation](docs/implementation.md) ·
[Validation](validation/README.md) · [Changelog](CHANGELOG.md)

## License

MIT License, see [LICENSE](LICENSE).

## References

1. Lindstrom, M. J., and D. M. Bates. "Nonlinear mixed-effects models for repeated measures data." Biometrics 46 (1990): 673-687.
2. Pinheiro, J. C., and D. M. Bates. Mixed-Effects Models in S and S-PLUS. Springer, 2000.
3. Kuhn, E., and M. Lavielle. "Maximum likelihood estimation in nonlinear mixed effects models." Computational Statistics & Data Analysis 49 (2005): 1020-1038.
4. Delyon, B., M. Lavielle, and E. Moulines. "Convergence of a stochastic approximation version of the EM algorithm." Annals of Statistics 27 (1999): 94-128.
5. Comets, E., A. Lavenu, and M. Lavielle. "Parameter estimation in nonlinear mixed effect models using saemix, an R implementation of the SAEM algorithm." Journal of Statistical Software 80 (2017).
