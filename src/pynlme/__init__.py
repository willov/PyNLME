"""
PyNLME - Nonlinear Mixed-Effects Models for Python

A Python implementation of nonlinear mixed-effects (NLME) models with a Rust backend,
providing functionality similar to MATLAB's nlmefit and nlmefitsa functions.
"""

from .data_types import ErrorModel, NLMEOptions, NLMEStats
from .nlmefit import fit_nlme, nlmefit, nlmefitsa  # Pythonic interface
from .utils import generate_design_matrix, transform_parameters

try:
    from importlib.metadata import version

    __version__ = version("pynlme")
except Exception:
    # Fallback if package not installed or importlib.metadata unavailable
    __version__ = "unknown"
__all__ = [
    # MATLAB-compatible functions
    "nlmefit",
    "nlmefitsa",
    # Pythonic interface
    "fit_nlme",
    # Data types
    "NLMEStats",
    "NLMEOptions",
    "ErrorModel",
    # Utilities
    "generate_design_matrix",
    "transform_parameters",
]
