"""
Error types for PyNLME.
"""


class PyNLMEError(Exception):
    """Base exception for PyNLME errors."""

    pass


class ConvergenceError(PyNLMEError):
    """Raised when algorithm fails to converge."""

    pass


class ModelEvaluationError(PyNLMEError):
    """Raised when model function evaluation fails."""

    pass


class ValidationError(PyNLMEError):
    """Raised when input validation fails."""

    pass
