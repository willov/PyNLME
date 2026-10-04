# PyNLME - Nonlinear Mixed-Effects Models for Python

A Python implementation of MATLAB's `nlmefit` and `nlmefitsa`, with the same call syntax, so that MATLAB users can move their nonlinear mixed-effects models to Python.

## AI Disclaimer

I (willov) asked Claude Sonnet 4 in GitHub Copilot agent mode to "create a Python NLME implementation with a rust backend with a similar call syntax to MATLAB's nlmefitsa". Please use the package with caution, it is very likely that the AI have hallucinated quirks not tested. If you find it useful, please assist in creating tests to test the reliability of the package.

## ⚠️ Current Status

Everything below is checked by the tests in [tests/validation/](tests/validation/); see [validation/](validation/README.md) for details and findings.

- **`nlmefit`** matches MATLAB R2023b `nlmefit` to about 1e-6 on every output (estimates, covariance, log-likelihood, AIC/BIC, standard errors, `rmse`, `errorparam`, all five residual types, random effects) for `"LME"` and `"RELME"`, with constant, proportional, combined and exponential error models. `"FOCE"` reaches MATLAB's optimum or a better one; `"FO"` reaches far better optima than MATLAB's, whose FO collapses all variances to zero on these datasets. On 640 randomly simulated datasets `nlmefit` reaches R `nlme`'s solution or a better one in all but one case, and it returns estimates where R fails to converge.
- **`nlmefitsa`** (SAEM) agrees with MATLAB `nlmefitsa` and R `saemix` within Monte Carlo error, for all four error models, reaches the exact maximum-likelihood estimates on a linear model, and removes the linearisation bias of `nlmefit` in simulation studies (log absorption rate: +1% vs -15%).
- **MATLAB or R conventions**: `compat="matlab"` (default) or `compat="r"` selects defaults and reporting conventions (standard-error scaling, BIC, REML variance, `RefineBeta0`, SAEM chains). Estimates are the same either way; explicit options override.
- **Implemented options**: `ApproximationType` (`LME`, `RELME`, `FO`, `FOCE`), `ErrorModel`, `ErrorParameters`, `ParamTransform`, `CovPattern` (diagonal or full), `FEParamsSelect`/`FEConstDesign`/`FEGroupDesign` (and `RE*` equivalents for `nlmefit`; `REParamsSelect` for `nlmefitsa`), `RefineBeta0`, `Vectorization` (`SinglePhi`, `SingleGroup`, `Full`), `Options` (`MaxIter`, `TolFun`, `TolX`, `Display`), and for `nlmefitsa` `NIterations`, `NMCMCIterations`, `NBurnIn`, `NChains`, `LogLikMethod` (`is`, `lin`), `ComputeStdErrors`. Anything else raises `NotImplementedError` rather than being ignored.
- **Not implemented yet**: non-constant error models with `FO`/`FOCE`, `LogLikMethod="gq"`, `Replicates`, `FEObsDesign`/`REObsDesign`, `OutputFcn`.

### ODE models (SUND or SciPy)

Models defined by differential equations work as any other model function:

- **SUND** (recommended): compile the model with the [SUND toolbox](https://isbgroup.eu/sund-toolbox) and wrap the simulation with `pynlme.sund_model(simulation, feature, parameters, v_states=[...])`. Group predictors such as the dose can set initial states or parameters. Install with `pip install "pynlme[sund]"`. See [examples/sund_ode_model.py](examples/sund_ode_model.py).
- **SciPy**: `pynlme.ode_model(rhs, y0, output)` with a right-hand side vectorised over systems; with `Vectorization="Full"` all subjects are solved in one `solve_ivp` call.

On the Michaelis-Menten validation model (no closed form) all three reproduce MATLAB's `ode45`-based `nlmefit` fit. Full fits (tight integrator tolerances, since `nlmefit` differentiates the model numerically):

| Model | `nlmefit` LME | `nlmefitsa` (50/50/50 iterations) |
|---|---|---|
| SciPy, one solve per subject | 45 s | 429 s |
| SciPy batched (`ode_model`, `Vectorization="Full"`) | 3.6 s | 27 s |
| SUND (`sund_model`) | 9.7 s | 43 s |
| MATLAB `ode45` per subject | 3-5 s | ~325 s (default iterations) |

A single SUND simulation is ~40x faster than a single SciPy solve, but the batched SciPy model solves all subjects in one call. SUND would overtake it with forward sensitivities (exact Jacobians instead of finite differences, so looser tolerances) and a batched simulate call.

### Speed

Pure Python/NumPy so far (the Rust extension is not used). With `Vectorization="Full"` (the model function accepts one parameter row per observation, as in MATLAB), from `uv run python validation/benchmark.py`:

| | pynlme | R | MATLAB R2023b |
|---|---|---|---|
| `nlmefit` LME (4 datasets) | 0.006-0.056 s | `nlme` 0.007-0.044 s | 0.02-0.23 s |
| `nlmefit` FOCE | 0.4-3.7 s | – | 14-49 s |
| `nlmefitsa`, MATLAB's default chains | 0.23-0.57 s | – | 1.3-3.7 s |
| `nlmefitsa`, 1 chain | 0.15-0.34 s | `saemix` 0.45-1.32 s | – |

SAEM is faster than both R and MATLAB on identical problems; LME is at parity with R `nlme` (slightly slower on one dataset). With the default `Vectorization="SinglePhi"` everything is correct but slower, because the model is called once per group.

## 🚀 Quick Start

The MATLAB documentation example, in pynlme:

```python
import numpy as np
from pynlme import nlmefit

# MATLAB: model = @(phi,t)(phi(1).*exp(-phi(2).*t)+phi(3).*exp(-phi(4).*t));
def model(phi, t):
    return phi[0] * np.exp(-phi[1] * t) + phi[2] * np.exp(-phi[3] * t)

# time, concentration, subject: the indomethacin data (validation/datasets/indometh.csv)
beta, psi, stats, b = nlmefit(time, concentration, subject, None, model,
                              [1, 1, 1, 1], ParamTransform=[0, 1, 0, 1])
```

Name-value arguments use MATLAB's names (`ParamTransform=...`, `REParamsSelect=[2, 3]`), and parameter indices given under MATLAB names are 1-based, as in MATLAB. The equivalent snake_case names (`param_transform`, `re_params_select`) use 0-based indices. Data can be stacked (MATLAB's layout: `y` a vector and `group` per observation) or grouped (`y` with one row per group, `group` optional); the layout follows from the shape of `y` or is set with `DataFormat="stacked"`/`"grouped"`. As in MATLAB, `modelfun` is called as `modelfun(phi, x)` when `V` is empty and `modelfun(phi, x, v)` otherwise, `psi` is r-by-r and `b` is r-by-m.

### 🎯 Multi-Dimensional Input Format (New!)

PyNLME now supports both traditional "stacked" format and an intuitive "grouped" format:

```python
# Traditional stacked format (MATLAB-style)
X_stacked = np.array([[1], [2], [3], [4], [1], [2], [3], [4]])  # All measurements
y_stacked = np.array([10, 7, 5, 3, 12, 8, 6, 4])               # All responses  
group = np.array([0, 0, 0, 0, 1, 1, 1, 1])                     # Group indicators

# New grouped format (each row = one subject)
X_grouped = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])  # Row per subject
y_grouped = np.array([[10, 7, 5, 3], [12, 8, 6, 4]]) # Row per subject
# No group parameter needed!

# Both work identically
beta1, psi1, _, _ = fit_nlme(X_stacked, y_stacked, group, None, model, beta0)
beta2, psi2, _, _ = fit_nlme(X_grouped, y_grouped, None, None, model, beta0)
# Results are identical: np.allclose(beta1, beta2) == True
```

### MATLAB Compatibility

For users migrating from MATLAB, PyNLME provides identical function signatures:

```python
# MATLAB-style usage (compatibility aliases)
from pynlme import nlmefit, nlmefitsa

# Maximum Likelihood Estimation (same as fit_nlme with method='MLE')
beta, psi, stats, b = nlmefit(t, y, group, None, exponential_decay, [10.0, 0.5])

# SAEM Algorithm (same as fit_nlme with method='SAEM') 
beta, psi, stats, b = nlmefitsa(t, y, group, None, exponential_decay, [10.0, 0.5])
```

## 📦 Installation

### Option 1: Install from GitHub Release (Recommended)

Pre-built wheels are available for easy installation without requiring Rust:

```bash
# Latest release (v0.3.0+):
# Choose the appropriate wheel for your platform:

# Linux (x86_64):
uv add https://github.com/willov/PyNLME/releases/download/v0.3.0/pynlme-0.3.0-cp311-abi3-linux_x86_64.whl

# Windows (x86_64):
uv add https://github.com/willov/PyNLME/releases/download/v0.3.0/pynlme-0.3.0-cp311-abi3-win_amd64.whl

# macOS (Intel):
uv add https://github.com/willov/PyNLME/releases/download/v0.3.0/pynlme-0.3.0-cp311-abi3-macosx_10_12_x86_64.whl

# macOS (Apple Silicon):
uv add https://github.com/willov/PyNLME/releases/download/v0.3.0/pynlme-0.3.0-cp311-abi3-macosx_11_0_arm64.whl

# Or using pip instead of uv:
pip install https://github.com/willov/PyNLME/releases/download/v0.3.0/pynlme-0.3.0-cp311-abi3-linux_x86_64.whl

# Or browse releases to find the exact wheel for your platform:
# https://github.com/willov/PyNLME/releases/latest
```

**Platform compatibility:**
- **Python**: 3.11, 3.12, 3.13+ (single wheel works for all)
- **Linux**: x86_64 (Intel/AMD 64-bit)
- **Windows**: x86_64 (Intel/AMD 64-bit)  
- **macOS**: Intel x86_64 and Apple Silicon ARM64

### Option 2: Install from source (for developers)

```bash
# Requires Rust toolchain for building
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

git clone https://github.com/willov/PyNLME
cd PyNLME
uv sync
uv run maturin develop  # Build Rust extension
```

> **Note**: The `cp311-abi3` wheels use Python's stable ABI and work with Python 3.11+ including future versions. You don't need separate wheels for each Python version. Option 2 is for developers who want to build from source or contribute to the project.

## 📚 Documentation

- **[Complete Documentation](docs/README.md)** - Full feature overview and examples
- **[API Reference](docs/api_reference.md)** - Detailed function documentation  
- **[Implementation Details](docs/implementation.md)** - Technical architecture
- **[Examples](examples/)** - Usage examples and tutorials
- **[Changelog](CHANGELOG.md)** - Version history and release notes

## 🧪 Quick Test

```bash
# Run the full test suite (for developers)
uv run pytest tests/ -v

# Verify installation works (for end users)
python -c "
import pynlme
print(f'PyNLME {pynlme.__version__} installed successfully!')
"

# Or use the comprehensive verification script
curl -sSL https://raw.githubusercontent.com/willov/PyNLME/main/scripts/verify-installation.py | python3
```

## 💡 Important Note

When using UV for package management, run Python scripts with `uv run python script.py` to ensure proper access to installed packages.

## 📄 License

MIT License - see LICENSE file for details.

## 📖 References

This implementation is based on the following papers:

1. Lindstrom, M. J., and D. M. Bates. "Nonlinear mixed-effects models for repeated measures data." Biometrics. Vol. 46, 1990, pp. 673–687.
2. Davidian, M., and D. M. Giltinan. Nonlinear Models for Repeated Measurements Data. New York: Chapman & Hall, 1995.
3. Pinheiro, J. C., and D. M. Bates. "Approximations to the log-likelihood function in the nonlinear mixed-effects model." Journal of Computational and Graphical Statistics. Vol. 4, 1995, pp. 12–35.
4. Demidenko, E. Mixed Models: Theory and Applications. Hoboken, NJ: John Wiley & Sons, Inc., 2004.
