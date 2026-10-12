# pynlme examples

Runnable scripts, from the repository root: `uv run python examples/<name>.py` (or
`python examples/<name>.py` with pynlme installed). The plots need `matplotlib`; the
SUND example needs `pip install "pynlme[sund]"`.

## Coming from MATLAB

| Example | Shows |
|---|---|
| [`matlab_documentation_examples.py`](matlab_documentation_examples.py) | MATLAB's `nlmefit` documentation example "Transform and Plot Fitted Model" (indomethacin), with the same calls and results |
| [`matlab_comparison.py`](matlab_comparison.py) | `nlmefit` (LME) and `nlmefitsa` (SAEM) on the same simulated PK data, compared with each other |

## Python interface

| Example | Shows |
|---|---|
| [`basic_usage.py`](basic_usage.py) | A first fit: an exponential decay with `fit_nlme` and `nlmefit` (LME) and SAEM, the outputs and a plot |
| [`api_comparison.py`](api_comparison.py) | The MATLAB-style functions next to the Python ones (`fit_nlme`, `fit_mle`, `fit_saem`) on the same data |
| [`function_aliases_demo.py`](function_aliases_demo.py) | `fit_nlme` as the one Python entry point for `nlmefit` and `nlmefitsa` (`method="ML"` or `"SAEM"`) |
| [`advanced_usage.py`](advanced_usage.py) | One- and two-compartment PK models compared, with diagnostics and plots |

## Data layouts

| Example | Shows |
|---|---|
| [`grouped_vs_stacked_comparison.py`](grouped_vs_stacked_comparison.py) | Stacked data (one row per observation) and grouped data (one row per subject) give the same fit |
| [`multidimensional_input_demo.py`](multidimensional_input_demo.py) | Several predictors per observation, in both layouts |

## ODE models and case studies

| Example | Shows |
|---|---|
| [`sund_ode_model.py`](sund_ode_model.py) | A Michaelis-Menten PK model compiled with SUND, fitted with `nlmefit` and `nlmefitsa` |
| [`multiple_features.py`](multiple_features.py) | A concentration and an effect (PK/PD) in the same subjects, stacked with a feature column, one ODE simulation per subject |
| [`warfarin_case_study.py`](warfarin_case_study.py) | A larger simulated PK study with covariates (age, weight, sex), SAEM and diagnostic plots |

More: the [tutorial](../docs/tutorial.md) walks through these features step by step,
and the [API reference](../docs/api_reference.md) documents every function and option.
