"""
Residuals for nlmefit and nlmefitsa, defined as MATLAB's nlmeres does.

All weighted residuals use the error standard deviation g evaluated at the
population prediction (random effects zero):

    pres  = y - f(beta, 0)                 ires  = y - f(beta, b)
    pwres = R0^-T pres,     R0'R0 = Z0 Psi Z0' + diag(g^2), Z0 = df/db at b = 0
    iwres = ires / g
    cwres = Rb^-T (y - f(beta, b) + Zb b),  Rb'Rb = Zb Psi Zb' + diag(g^2)
"""

import numpy as np

from .model import GroupedModel


def error_sd(error_model: str, errorparam: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Standard deviation of the error for predictions f."""
    if error_model in ("constant", "exponential"):
        return np.full_like(f, errorparam[0])
    if error_model == "proportional":
        return f * errorparam[0]
    if error_model == "combined":
        return errorparam[0] + errorparam[1] * np.abs(f)
    raise ValueError(f"Unknown error model '{error_model}'")


def compute_residuals(
    model: GroupedModel, y: np.ndarray, beta, b, psi, error_model: str, errorparam
) -> dict[str, np.ndarray | None]:
    n = model.n_obs
    out = {k: np.empty(n) for k in ("pres", "ires", "pwres", "iwres", "cwres")}
    weighted_ok = True
    # The model for all groups at once (one call each with Vectorization="Full").
    groups = np.arange(model.n_groups)
    etas0 = np.array([model.eta(i, beta, None) for i in groups])
    etasb = np.array([model.eta(i, beta, b[:, i]) for i in groups])
    fs0, fsb = model.predict_many(groups, etas0), model.predict_many(groups, etasb)
    Js0, Jsb = model.jacobian_many(groups, etas0), model.jacobian_many(groups, etasb)
    for i, rows in enumerate(model.rows):
        f0, fb = fs0[i], fsb[i]
        yi = y[rows]
        out["pres"][rows] = yi - f0
        out["ires"][rows] = yi - fb
        if not weighted_ok:
            continue
        Z0 = Js0[i] @ model.B[i]
        Zb = Jsb[i] @ model.B[i]
        g = error_sd(error_model, errorparam, f0)
        try:
            R0 = np.linalg.cholesky(Z0 @ psi @ Z0.T + np.diag(g**2))
            Rb = np.linalg.cholesky(Zb @ psi @ Zb.T + np.diag(g**2))
        except np.linalg.LinAlgError:
            # As MATLAB: no weighted residuals when an error variance is
            # (close to) zero, e.g. a proportional error at a zero prediction.
            weighted_ok = False
            continue
        out["iwres"][rows] = (yi - fb) / g
        out["pwres"][rows] = np.linalg.solve(R0, yi - f0)
        out["cwres"][rows] = np.linalg.solve(Rb, yi - fb + Zb @ b[:, i])
    if not weighted_ok:
        out["iwres"] = out["pwres"] = out["cwres"] = None
    return out
