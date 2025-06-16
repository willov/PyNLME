"""
PyNLME - Nonlinear Mixed-Effects Models for Python

A high-performance Python library for nonlinear mixed-effects (NLME) modeling
with a Rust backend. PyNLME provides both a modern Pythonic API and
MATLAB-compatible interfaces for seamless migration.

Quick Start
-----------
>>> import numpy as np
>>> from pynlme import fit_nlme
>>>
>>> # Define your model
>>> def model(phi, x, v=None):
...     return phi[0] * np.exp(-phi[1] * x)
>>>
>>> # Fit the model
>>> beta, psi, stats, b = fit_nlme(X, y, group, None, model, beta0)

For MATLAB users, nlmefit() and nlmefitsa() are also available.
"""

from .data_types import ErrorModel, NLMEOptions, NLMEStats
from .nlmefit import fit_nlme, fit_mle, fit_saem, nlmefit, nlmefitsa
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
    # Primary Python interface (recommended)
    "fit_nlme",
    "fit_mle",
    "fit_saem",
    # MATLAB-compatible functions
    "nlmefit",
    "nlmefitsa",
    # Data types
    "NLMEStats",
    "NLMEOptions",
    "ErrorModel",
    # Utilities
    "detect_data_format",
    "generate_design_matrix",
    "stack_grouped_data",
    "transform_parameters",
]
