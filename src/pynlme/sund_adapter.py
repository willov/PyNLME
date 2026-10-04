"""
Use a compiled SUND model (https://isbgroup.eu/sund-toolbox) as an nlmefit or
nlmefitsa model function.

    import sund
    from pynlme import nlmefit, sund_model

    sund.install_model("mm_pk.txt")
    sim = sund.Simulation(models=[sund.load_model("mm_pk")], time_unit="h")
    model = sund_model(sim, feature="conc", parameters=["ka", "V", "vmax", "km"], v_states=["gut"])
    nlmefit(time, conc, subject, dose, model, beta0, ParamTransform=[1, 1, 1, 1])

phi supplies the named SUND parameters (in order); the columns of V (group
predictors such as the dose) set the named initial states, then the named
parameters. All other parameters and initial states keep the model's values.
"""

from collections.abc import Sequence

import numpy as np


def sund_model(
    simulation,
    feature: str,
    parameters: Sequence[str],
    v_states: Sequence[str] = (),
    v_parameters: Sequence[str] = (),
    t0: float = 0.0,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-12,
):
    """Build a model function from a ``sund.Simulation``.

    Parameters
    ----------
    simulation : sund.Simulation
        The simulation object (models loaded, time unit set).
    feature : str
        Name of the SUND feature that is observed.
    parameters : sequence of str
        SUND parameter names set from phi, in phi's order.
    v_states, v_parameters : sequence of str
        Initial states, then parameters, set from the columns of V.
    t0 : float
        Start time of each simulation (the initial states apply here).
    rel_tol, abs_tol : float
        Integrator tolerances. Tighter than SUND's defaults because nlmefit
        differentiates the model numerically.

    Returns
    -------
    callable
        ``model(phi, x, v=None)`` with x[:, 0] the observation times. Works
        with Vectorization "SinglePhi" and "Full" (consecutive identical rows of
        phi and v are simulated once).
    """
    simulation.set_options({"rel_tol": rel_tol, "abs_tol": abs_tol})
    param_names = list(simulation.parameter_names)
    state_names = list(simulation.state_names)
    feature_index = list(simulation.feature_names).index(feature)
    p_idx = np.array([param_names.index(p) for p in parameters], dtype=int)
    vs_idx = np.array([state_names.index(s) for s in v_states], dtype=int)
    vp_idx = np.array([param_names.index(p) for p in v_parameters], dtype=int)
    p_default = np.array(simulation.parameter_values, dtype=float)
    x_default = np.array(simulation.state_values, dtype=float)
    n_vs = len(vs_idx)

    def simulate(phi_row, v_row, times):
        p = p_default.copy()
        p[p_idx] = phi_row
        x0 = x_default.copy()
        if v_row is not None:
            x0[vs_idx] = v_row[:n_vs]
            p[vp_idx] = v_row[n_vs : n_vs + len(vp_idx)]
        grid = times if times[0] == t0 else np.concatenate([[t0], times])
        simulation.simulate(time_vector=grid, parameter_values=p, state_values=x0)
        values = np.asarray(simulation.feature_values)[:, feature_index]
        return values if times[0] == t0 else values[1:]

    def model(phi, x, v=None):
        t = np.asarray(x, dtype=float).reshape(len(x), -1)[:, 0]
        n = len(t)
        phi = np.asarray(phi, dtype=float)
        if phi.ndim == 1:
            phi = np.broadcast_to(phi, (n, phi.size))
        if v is not None:
            v = np.asarray(v, dtype=float)
            v = np.broadcast_to(v.reshape(1, -1) if v.ndim <= 1 else v, (n, v.shape[-1]))
        key = phi if v is None else np.hstack([phi, v])
        starts = np.flatnonzero(np.r_[True, np.any(key[1:] != key[:-1], axis=1)])
        ends = np.r_[starts[1:], n]
        out = np.empty(n)
        for a, b in zip(starts, ends, strict=True):
            times, inverse = np.unique(t[a:b], return_inverse=True)
            values = simulate(phi[a], None if v is None else v[a], times)
            out[a:b] = values[inverse]
        return out

    return model
