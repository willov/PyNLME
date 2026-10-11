"""
pynlme - Nonlinear Mixed-Effects Models for Python

Nonlinear mixed-effects (NLME) models in Python (NumPy, SciPy): MATLAB's
nlmefit and nlmefitsa (Lindstrom-Bates, FO, FOCE, SAEM), validated against
MATLAB R2023b, R nlme and R saemix.

Quick Start
-----------
>>> import numpy as np
>>> from pynlme import fit_nlme
>>>
>>> def model(phi, x):
...     return phi[0] * np.exp(-phi[1] * x[:, 0])
>>>
>>> beta, psi, stats, b = fit_nlme(X, y, group, None, model, beta0)

Also: nlmefit/nlmefitsa (MATLAB names and options), compat="r" (R's nlme
iteration and conventions), pynlme.r (R's nlme() and saemix interfaces), and
ODE models via sund_model (SUND) or ode_model (SciPy).
"""

from .api import fit, predict
from .data_types import NLMEOptions, NLMEStats
from .model import with_jacobian
from .nlmefit import fit_mle, fit_nlme, fit_saem, nlmefit, nlmefitsa
from .ode import ode_model
from .sund_adapter import sund_model
from .utils import (
    detect_data_format,
    generate_design_matrix,
    stack_grouped_data,
    transform_parameters,
)

try:
    from importlib.metadata import version

    __version__ = version("pynlme")
except (ImportError, ModuleNotFoundError):
    # Fallback if package not installed or importlib.metadata unavailable
    __version__ = "unknown"

__all__ = [
    # Named parameters, dict data and results
    "fit",
    "predict",
    # Python interface with arrays
    "fit_nlme",
    "fit_mle",
    "fit_saem",
    # MATLAB-compatible functions
    "nlmefit",
    "nlmefitsa",
    "ode_model",
    "with_jacobian",
    "sund_model",
    # Data types
    "NLMEStats",
    "NLMEOptions",
    # Utilities
    "detect_data_format",
    "generate_design_matrix",
    "stack_grouped_data",
    "transform_parameters",
]
