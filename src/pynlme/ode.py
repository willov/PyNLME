"""
Model functions defined by ordinary differential equations.

``ode_model`` turns a vectorised right-hand side into a model function for
nlmefit/nlmefitsa. With Vectorization="Full" every distinct parameter row
(one per group, MCMC chain or finite-difference perturbation) becomes one ODE
system, and all systems are stacked and solved in a single solve_ivp call
instead of one call per group.

Example (oral dose, Michaelis-Menten elimination; states [gut, conc]):

    def rhs(t, y, phi, dose):
        ka, V, vmax, km = phi.T
        gut, c = y.T
        return np.column_stack([-ka * gut, ka * gut / V - vmax * c / (km + c)])

    model = ode_model(rhs, y0=lambda phi, dose: np.column_stack([dose[:, 0], 0 * dose[:, 0]]),
                      output=lambda y, phi, dose: y[:, 1])
    nlmefit(time, conc, subject, dose, model, beta0, Vectorization="Full", ...)
"""

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp


def ode_model(
    rhs: Callable,
    y0: Callable,
    output: Callable,
    t0: float = 0.0,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    feature_column: int = 1,
) -> Callable:
    """Build a model function from a vectorised ODE.

    Parameters
    ----------
    rhs : callable
        ``rhs(t, y, phi, v)`` returning dy/dt with shape (K, n_states) for K
        systems; y is (K, n_states), phi (K, p), v (K, g) or None.
    y0 : callable
        ``y0(phi, v)`` returning the initial states at ``t0``, (K, n_states).
    output : callable
        ``output(y, phi, v)`` returning the observed quantity, shape (K,),
        from states y of shape (K, n_states). For several observed features
        return one column per feature, shape (K, n_features); column
        ``feature_column`` of x then gives each row's feature (0-based), so
        that all observations of a subject are stacked in one response vector.
    t0 : float
        Time of the initial state (e.g. the dose); observations are at t >= t0.
    feature_column : int
        Column of x holding the feature index (several features only).
    method, rtol, atol
        Passed to scipy.integrate.solve_ivp. The tolerances apply to the
        stacked system; tight values keep finite-difference Jacobians accurate.

    Returns
    -------
    callable
        ``model(phi, x, v=None)``: x[:, 0] are the observation times. phi is a
        parameter vector (one system) or a matrix with one row per observation
        (Vectorization="Full"); consecutive identical rows of phi and v form
        one system.
    """

    def model(phi, x, v=None):
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        t = x[:, 0]
        n = len(t)
        phi = np.asarray(phi, dtype=float)
        if phi.ndim == 1:
            phi = np.broadcast_to(phi, (n, phi.size))
        if v is not None:
            v = np.asarray(v, dtype=float)
            v = np.broadcast_to(v.reshape(1, -1) if v.ndim <= 1 else v, (n, v.shape[-1]))

        # Systems are runs of identical (phi, v) rows.
        key = phi if v is None else np.hstack([phi, v])
        new = np.ones(n, dtype=bool)
        new[1:] = np.any(key[1:] != key[:-1], axis=1)
        system = np.cumsum(new) - 1
        first = np.flatnonzero(new)
        P = phi[first]
        Vs = None if v is None else v[first]
        Y0 = np.asarray(y0(P, Vs), dtype=float)
        K, n_states = Y0.shape

        times = np.unique(t)
        if times[0] < t0:
            raise ValueError("observation times must be >= t0")

        def f(time, flat):
            return np.asarray(rhs(time, flat.reshape(K, n_states), P, Vs), dtype=float).ravel()

        t_end = float(times[-1])
        if t_end <= t0:
            states = np.repeat(Y0[None], len(times), axis=0)
        else:
            sol = solve_ivp(f, (t0, t_end), Y0.ravel(), t_eval=times, method=method, rtol=rtol, atol=atol)
            if not sol.success:
                return np.full(n, np.nan)
            states = sol.y.T.reshape(len(times), K, n_states)
        at_obs = states[np.searchsorted(times, t), system]  # (n, n_states)
        values = np.asarray(output(at_obs, phi, v), dtype=float)
        if values.ndim == 2 and values.shape[1] > 1:
            from .sund_adapter import _feature_rows

            return values[np.arange(n), _feature_rows(x, feature_column, values.shape[1])]
        return values.reshape(n)

    return model
