# pynlme: nonlinear mixed-effects models in Python

Fit nonlinear mixed-effects models (population PK/PD, growth curves, systems-biology
models) in Python. pynlme reimplements the published algorithms behind MATLAB's
`nlmefit`/`nlmefitsa` and R's `nlme`/`saemix`, reproduces their results, and is
validated against all of them.

pynlme is an independent project, not affiliated with MathWorks or the R project. Its
code is its own and contains no MATLAB or R code: the replicate modes reproduce the
reference programs' computations, worked out from the publications, their
documentation, outputs and traces, and their source (see
[Implementation: Sources](docs/implementation.md#sources)).

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
`stats` has MATLAB's fields and definitions. `nlmefit` reproduces MATLAB's iteration
step by step with MATLAB's default options, so on most data the result is MATLAB
R2023b's to ~1e-10 (checked on thousands of simulated datasets).
Start with the [tutorial](docs/tutorial.md).

### Python users

The same estimators with snake_case options and 0-based indices:

```python
import numpy as np
from pynlme import fit_nlme

def model(phi, t):
    return phi[0] * np.exp(-phi[1] * t[:, 0])

beta, psi, stats, b = fit_nlme(t, y, group, None, model, [10.0, 0.5])
beta, psi, stats, b = fit_nlme(t, y, group, None, model, [10.0, 0.5], method="SAEM",
                               random_state=1)
```

`t` holds the predictors (one row per observation), `group` the subject of each row;
data can also be given grouped, one row per subject. Options such as
`param_transform=[0, 1]`, `re_params_select=[0]` or `error_model="proportional"` follow
MATLAB's, and `vectorization="full"` evaluates all subjects in one model call. By
default the estimators follow MATLAB's; `compat="r"` follows R's.

`pynlme.fit` is a named-parameter interface (dict or Polars data, dict results); its
design is not settled yet and may change. See the [API reference](docs/api_reference.md).

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
`predict(fit, newdata, level=)` work as in R; the fit itself is a dict. R's defaults
are kept (`method="ML"`, `pdSymm` for a plain `random` formula), and the results match
R `nlme` to ~1e-4 or closer. Not yet
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

`compat="r"` reproduces nlme's iteration step by step, so pynlme returns R's fit even
where the model has several local solutions, and R's `method="REML"` (which for
nonlinear models is not a restricted-likelihood fit). The default `compat="matlab"`
RELME is a true REML fit. See [docs/troubleshooting.md](docs/troubleshooting.md).

## What is validated

Every claim here is a test in [tests/validation/](tests/validation/); the details and
findings are in [validation/README.md](validation/README.md).

- **`nlmefit`**: matches MATLAB R2023b on every output (estimates, covariance,
  log-likelihood, AIC/BIC, standard errors, error parameters, all residual types,
  random effects) for LME and RELME with all four error models; on 3200 simulated
  fits, 2699 of the 2715 that never had a variance at zero agree with MATLAB to <1e-8.
  FOCE reaches MATLAB's optimum or a better one; FO reaches far better optima than
  MATLAB's (whose FO collapses the variances to zero).
- **`compat="r"`**: matches R `nlme` on fixed datasets (ML and nlme's REML), and gives
  R's fit on 2399 of 2400 freshly simulated datasets (the other has a variance that is
  not identified).
- **`nlmefitsa`** (SAEM): agrees with MATLAB and R `saemix` within Monte Carlo error;
  reaches the exact maximum likelihood on a linear model; removes the linearisation
  bias of `nlmefit` in simulation studies.
- **ODE models** (SciPy, or compiled with SUND): reproduce MATLAB's `ode45`-based fits.
- **Accuracy** ([validation/accuracy/](validation/accuracy/README.md)): on 100 simulated
  datasets pynlme's estimates are unbiased and its 95% confidence intervals cover the
  true values 91-99% of the time, for LME, FOCE and SAEM, the same as MATLAB, R and
  nlmixr2 (except MATLAB's FOCE and nlmixr2's SAEM, which are biased for one
  parameter).

Supported: `ApproximationType` LME/RELME/FO/FOCE; error models constant, proportional,
combined, exponential; `ParamTransform`; fixed- and random-effects designs including
covariate models; diagonal or full covariance; SAEM options and log-likelihoods. Not
yet: `FEObsDesign`/`REObsDesign`, block `CovPattern`, `LogLikMethod="gq"`,
`Replicates`. Unsupported options raise errors; none are silently ignored.

## Speed

pynlme against the original programs (MATLAB R2023b `nlmefit`/`nlmefitsa`, R `nlme`
and `saemix`), all run on the same machine, on the same data with the same start
values and options; times in milliseconds, the best of several runs (MATLAB after a
warm-up run). Each pynlme mode runs the iteration of the program next to it. The
models are called once for all subjects (`Vectorization="Full"` in pynlme and MATLAB;
R and saemix do this anyway).

| LME | pynlme MATLAB mode (NumPy) | pynlme MATLAB mode (Rust) | MATLAB `nlmefit` | pynlme R mode (NumPy) | pynlme R mode (Rust) | R `nlme` |
|---|---|---|---|---|---|---|
| indometh | 142 | 18.9 | 72.2 | 81.3 | 18.7 | 46.0 |
| theoph | 28.3 | 16.2 | 16.1 | 108 | 24.2 | 39.0 |
| orange | 9.6 | 5.0 | 3.8 | 13.2 | 3.1 | 7.0 |
| linear | 24.5 | 7.6 | 20.9 | 11.1 | 4.5 | 7.0 |

| FOCE | pynlme (NumPy) | pynlme (Rust) | MATLAB `nlmefit` |
|---|---|---|---|
| indometh | 2695 | 693 | 1447 |
| theoph | 1966 | 617 | 1722 |
| orange | 411 | 96.5 | 97.9 |
| linear | 336 | 63.2 | 137 |

| SAEM | pynlme `nlmefitsa` (NumPy) | pynlme `nlmefitsa` (Rust) | MATLAB `nlmefitsa` | pynlme `pynlme.r.saemix` | R `saemix` |
|---|---|---|---|---|---|
| theoph | 246 | 163 | 875 | 238 | 1305 |
| orange | 175 | 110 | 1538 | 174 | 1253 |
| linear | 157 | 85.1 | 1665 | 131 | 466 |

In short: on these small problems all programs take milliseconds to a second, and
the differences rarely matter in practice. With the Rust kernels (the default):

- **LME**: pynlme is faster than MATLAB on indometh and linear (2.8-3.8 times), equal
  on theoph and slower on orange (5.0 vs 3.8 ms), and 1.6-2.5 times faster than R
  `nlme`.
- **FOCE**: pynlme is 2-3 times faster than MATLAB, and equal on orange. Its FOCE is
  its own iteration (it reaches MATLAB's optimum or a better one, not MATLAB's path).
- **SAEM**: pynlme is 5-20 times faster than MATLAB and 3.5-7 times faster than
  saemix, with the same iterations and chains as each.

Without the kernels (NumPy only) pynlme is 1.1-4 times slower than MATLAB and R for
LME and FOCE, and still faster for SAEM.

Machine: AMD Ryzen 7 7800X3D (8 cores, 16 threads), 32 GB RAM, Linux (Arch, kernel
7.2). pynlme 0.5.0 on Python 3.13, NumPy 2.5, SciPy 1.18; MATLAB R2023b Update 1
(Statistics and Machine Learning Toolbox 23.2, 8 computational threads); R 4.6.1, nlme
3.1-169, saemix 3.5.

We are not experts in tuning MATLAB or R for speed: the original programs run as a
typical user would run them (their default settings, an interpreted model function,
the system BLAS), and they could likely be made faster. The times also vary by
10-20% between runs, and R's timer resolves 1 ms. The cases are small; larger
problems (more subjects, ODE models) may rank differently.

**Larger problems** ([validation/scaling/](validation/scaling/README.md): 10, 100 and
1000 simulated subjects, a closed-form and an ODE model, also against nlmixr2). At 1000
subjects pynlme is the fastest for LME (19 times faster than MATLAB on the closed-form
model; on the ODE model 10 times faster than MATLAB and 27 times faster than R
`nlme`) and for SAEM (2.5-85 times faster than MATLAB, saemix and nlmixr2 on the
closed-form model, 1.3 times faster than nlmixr2 on the ODE model). Its FOCE is 70
times faster than MATLAB's, but nlmixr2's FOCEi is 6 times faster than pynlme on the
closed-form model and far faster on the ODE model, where nlmixr2 uses sensitivity
equations and pynlme finite differences. All programs reach the same fits.

Verify on your machine: `bash validation/run_benchmarks.sh` runs every benchmark with
what is installed (pynlme always; R and MATLAB when available, otherwise their stored
timings are used) and prints these tables. Or step by step: `Rscript validation/r/benchmark.R`, `validation/matlab/benchmark.m` in
MATLAB, then `uv run python validation/benchmark.py --markdown`.

The Rust kernels (`rust/`, built into the package as `pynlme._kernels`) compile the
inner loops of every algorithm: both LME modes, FO/FOCE and SAEM's sampler; the future
custom mode will use them too. The model function stays Python and is called back
once per step for all subjects. The kernels do the same arithmetic as the NumPy code,
which runs instead when they are missing or `PYNLME_KERNELS=0` is set. Without `Vectorization="Full"` the model is called once per
subject: same results, several times slower.

## ODE models

- **SUND** (recommended): compile the model with the
  [SUND toolbox](https://isbgroup.eu/sund-toolbox) and use
  `pynlme.sund_model("model_name", feature, parameters, v_states=[...])`
  (`pip install "pynlme[sund]"`); see [examples/sund_ode_model.py](examples/sund_ode_model.py).
- **SciPy**: `pynlme.ode_model(rhs, y0, output)` with a vectorised right-hand side.
- **Doses and other inputs**: SUND activities, one set per experiment
  (`sund_model(..., activities={1: oral, 2: infusion})`, the experiment in a column of
  `V`); person-specific doses (e.g. per kg) are written in the model.
- **Several measured features** (e.g. concentration and effect): stack them in one
  response vector with a feature column in `X`, and pass `feature=["conc", "effect"]`
  (SUND) or a multi-column `output` (SciPy); each subject is simulated once.

On the Michaelis-Menten validation model `nlmefit` took 1.0 s with SUND (batched with
containers), 3.6 s with batched SciPy, and 3-5 s in MATLAB with `ode45` (single runs,
not part of the benchmark above). Analytic
derivatives (`pynlme.with_jacobian`) replace finite differences when available.

## Installation

Python 3.11-3.14, on NumPy and SciPy, with compiled kernels in Rust. The wheels from
[Releases](https://github.com/willov/PyNLME/releases) (Linux, Windows, macOS) need no
compiler:

```bash
pip install pynlme-<version>-cp311-abi3-<platform>.whl            # a wheel from Releases
pip install git+https://github.com/willov/PyNLME                  # builds from source
pip install "pynlme[sund] @ git+https://github.com/willov/PyNLME"   # with SUND models
```

Installing from git (or the source archive) compiles the kernels and needs a Rust
toolchain ([rustup.rs](https://rustup.rs)).

SUND compiles your model files on `sund.install_model("model.txt")`, with the system
C compiler if there is one and otherwise the Zig compiler it installs with it.

For development:

```bash
git clone https://github.com/willov/PyNLME && cd PyNLME
uv sync --dev --extra sund     # builds the Rust kernels; leave out --extra sund to skip SUND
uv run pytest                  # --runslow adds the simulation studies
```

`uv sync` rebuilds the kernels when files in `rust/` change, and installs exactly what
is requested: running it without `--extra sund`
uninstalls SUND (its folder of compiled models stays behind as an empty `sund`
package; rerun with `--extra sund` to restore it). The validation tests compare with R
when `Rscript` with `nlme` and `jsonlite` is available, and with MATLAB when
`PYNLME_MATLAB` names the MATLAB command (see
[validation/README.md](validation/README.md)).

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
