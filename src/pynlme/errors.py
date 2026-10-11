"""
Error types for pynlme.
"""


class pynlmeError(Exception):
    """Base exception for pynlme errors."""

    pass


class ConvergenceError(pynlmeError):
    """Raised when algorithm fails to converge."""

    pass


class ModelEvaluationError(pynlmeError):
    """Raised when model function evaluation fails."""

    pass


class ValidationError(pynlmeError):
    """Raised when input validation fails."""

    pass
