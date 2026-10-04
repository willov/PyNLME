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

Snake_case options with 0-based indices, data stacked (a group per observation) or
grouped (one row per subject):

```python
from pynlme import fit_nlme

beta, psi, stats, b = fit_nlme(t, y, group, None, model, beta0,
                               param_transform=[0, 1, 0, 1], re_params_select=[1, 3],
                               method="SAEM", random_state=1)
```

The model is a plain function `f(phi, x)` (or `f(phi, x, v)` with group-level
predictors). Write it for `vectorization="full"` to have it evaluated for many subjects
at once. See the [API reference](docs/api_reference.md).

### R users (nlme, saemix) who would rather not use R

The same algorithms, with R's reporting conventions via `compat="r"`:

| R | pynlme |
|---|---|
| `nlme(..., method="ML")` / `"REML"` | `nlmefit(..., ApproximationType="LME")` / `"RELME"`, `compat="r"` |
| `fixed = a + b ~ 1`, `start = c(...)` | `beta0` (one per fixed effect) |
| `fixed = list(a ~ 1, b ~ wt)` | `FEGroupDesign` (covariates per group) |
| `random = pdDiag(a ~ 1)` / `pdSymm(...)` | `REParamsSelect`, `CovPattern` diagonal / full |
| `saemix`: `transform.par`, `covariance.model` | `ParamTransform` (same codes), `CovPattern` |
| `saemix(...)` | `nlmefitsa(..., compat="r")` (one chain, as saemix) |

R's `nlme` matches to ~1e-6. One difference: for nonlinear models R's
`nlme(method="REML")` is not a restricted-likelihood fit; pynlme's `RELME` is (and equals
`lme`'s REML on linear models). See [docs/troubleshooting.md](docs/troubleshooting.md).

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
