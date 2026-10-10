"""
Use a compiled SUND model (https://isbgroup.eu/sund-toolbox) as an nlmefit or
nlmefitsa model function.

    import sund
    from pynlme import nlmefit, sund_model

    sund.install_model("mm_pk.txt")
    model = sund_model(
        "mm_pk", feature="conc", parameters=["ka", "V", "vmax", "km"], v_states=["gut"]
    )
    nlmefit(time, conc, subject, dose, model, beta0, ParamTransform=[1, 1, 1, 1],
            Vectorization="Full")

phi supplies the named SUND parameters (in order); the columns of V (group
predictors such as the dose) set the named initial states, then the named
parameters. All other parameters and initial states keep the model's values.

Several observed features (e.g. a concentration and an effect): stack all
observations of a subject into one response vector and add a predictor column
with each row's feature, as an index into ``feature``:

    model = sund_model("pkpd", feature=["conc", "effect"], parameters=[...])
    X = np.column_stack([time, feature_index])   # 0 = conc, 1 = effect
    nlmefit(X, y, subject, dose, model, beta0, Vectorization="Full")

Each subject is simulated once and read out at all its features. The features
share one error model (as in MATLAB's nlmefit).

Inputs (doses, infusions, other time-varying inputs) come from SUND activities,
one set per experiment, with a column of V saying which experiment each subject
belongs to. Give each experiment as a function of the container: sund_model
creates the simulations, puts subjects in containers and builds each copy's
activities, so subjects of an experiment are simulated in batches:

    def oral(container):
        a = sund.Activity(time_unit="h", container=container)
        a.add_state_manipulation(name="gut", t=[0, 12], f=[100, 100], mode="add")
        return a

    model = sund_model("pk", "conc", ["ka", "V", "CL"], v_parameters=["BW"],
                       activities={1: oral, 2: infusion})   # V = [BW, experiment]

Ready Activity objects ({1: activity}) also work; then each subject is
simulated on its own. Person-specific inputs (e.g. a dose per kg) belong in the
model: an input per kg times a body-weight parameter set from V.

With Vectorization="Full" the parameter sets of a model call (subjects, MCMC
chains, finite-difference perturbations) are simulated in batches: one SUND
simulation holds ``batch_size`` copies of the model in separate containers.
The integrator treats a batch as one system with a dense linear solver, so
batches beyond a few dozen copies get slower than separate simulations; 16
was fastest on the validation model.
"""

from collections.abc import Sequence

import numpy as np


def sund_model(
    model,
    feature: str | Sequence[str],
    parameters: Sequence[str],
    v_states: Sequence[str] = (),
    v_parameters: Sequence[str] = (),
    t0: float = 0.0,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-12,
    batch_size: int = 16,
    time_unit: str | None = None,
    feature_column: int = 1,
    activities=None,
    activity_column: int | None = None,
):
    """Build a model function from an installed SUND model.

    Parameters
    ----------
    model : str or sund.Simulation
        Name of an installed SUND model (enables batched simulation with
        containers), or a ready ``sund.Simulation`` (simulated one parameter
        set at a time).
    feature : str or sequence of str
        Name of the SUND feature that is observed, or several names: then
        column ``feature_column`` of x gives each row's feature as a 0-based
        index into this list.
    parameters : sequence of str
        SUND parameter names set from phi, in phi's order.
    v_states, v_parameters : sequence of str
        Initial states, then parameters, set from the columns of V.
    t0 : float
        Start time of each simulation (the initial states apply here).
    rel_tol, abs_tol : float
        Integrator tolerances. Tighter than SUND's defaults because nlmefit
        differentiates the model numerically.
    batch_size : int
        Parameter sets per batched simulation (model name only).
    time_unit : str, optional
        Simulation time unit; defaults to the model's.
    feature_column : int
        Column of x holding the feature index (several features only).
    activities : dict, optional
        The model's inputs per experiment: experiment ID -> a function
        ``container -> Activity`` (or list), batched in containers; or ready
        ``Activity`` objects, one subject at a time. The ID is read from column
        ``activity_column`` of V. Needs ``model`` as a name.
    activity_column : int, optional
        Column of V with the experiment ID. Default: the column after those
        used by ``v_states`` and ``v_parameters``.

    Returns
    -------
    callable
        ``model(phi, x, v=None)`` with x[:, 0] the observation times. Works
        with Vectorization "SinglePhi" and "Full" (consecutive identical rows of
        phi and v are simulated once).
    """
    import sund

    options = {"rel_tol": rel_tol, "abs_tol": abs_tol}
    if activities is not None and not isinstance(model, str):
        raise ValueError(
            "activities need the model as the name of an installed SUND model"
        )
    if isinstance(model, str):
        single = sund.load_model(model)
        model_class = sund.import_model(model)
        unit = time_unit or single.time_unit
        reference = sund.Simulation(models=[single], time_unit=unit)
    else:
        reference, model_class, unit = model, None, None
        batch_size = 1
    reference.set_options(options)

    param_names = list(reference.parameter_names)
    state_names = list(reference.state_names)
    feature_names = list(reference.feature_names)
    n_f = len(feature_names)
    multiple = not isinstance(feature, str)
    observed = list(feature) if multiple else [feature]
    missing = [f for f in observed if f not in feature_names]
    if missing:
        raise ValueError(
            f"unknown SUND features {missing}; the model has {feature_names}"
        )
    feature_index = np.array([feature_names.index(f) for f in observed], dtype=int)
    p_idx = np.array([param_names.index(p) for p in parameters], dtype=int)
    vs_idx = np.array([state_names.index(s) for s in v_states], dtype=int)
    vp_idx = np.array([param_names.index(p) for p in v_parameters], dtype=int)
    p_default = np.array(reference.parameter_values, dtype=float)
    x_default = np.array(reference.state_values, dtype=float)
    n_vs = len(vs_idx)
    batches: dict[int, object] = {1: reference}
    if activity_column is None:
        activity_column = n_vs + len(vp_idx)
    if activities is not None and not isinstance(activities, dict):
        raise TypeError(
            "activities must be a dict from experiment ID to activities (or a function "
            "container -> activities); person-specific inputs go in the model, e.g. a "
            "dose per kg times a body-weight parameter set from V"
        )
    by_experiment = (
        {float(k): a for k, a in activities.items()} if activities is not None else None
    )
    activity_sims: dict = {}

    def as_list(acts):
        return list(acts) if isinstance(acts, list | tuple) else [acts]

    def activity_spec(v_row):
        """(experiment, spec) for a subject. spec is the experiment's activities,
        or a builder ``container -> activities`` (then subjects are batched in
        containers)."""
        if v_row is None or len(v_row) <= activity_column:
            raise ValueError(
                f"activities need the experiment ID in column {activity_column} of V"
            )
        key = float(v_row[activity_column])
        if key not in by_experiment:
            raise KeyError(
                f"no activities for experiment {v_row[activity_column]:g} "
                f"(have {sorted(by_experiment)})"
            )
        return key, by_experiment[key]

    def is_builder(spec) -> bool:
        return callable(spec) and not isinstance(spec, sund.Activity)

    def activity_simulation(key, acts):
        """One model with these activities (cached per experiment or subject)."""
        if key not in activity_sims:
            sim = sund.Simulation(
                models=[sund.load_model(model)],
                activities=as_list(acts),
                time_unit=unit,
            )
            sim.set_options(options)
            activity_sims[key] = sim
        return activity_sims[key]

    def built_simulation(keys, builders):
        """Model copies in containers pynlme0, pynlme1, ..., each with the
        activities its builder makes for that container (cached per batch)."""
        if ("built", keys) not in activity_sims:
            containers = [f"pynlme{j}" for j in range(len(builders))]
            acts = [
                a
                for c, build in zip(containers, builders, strict=True)
                for a in as_list(build(c))
            ]
            sim = sund.Simulation(
                models=[model_class(container=c) for c in containers],
                activities=acts,
                time_unit=unit,
            )
            sim.set_options(options)
            activity_sims["built", keys] = sim
        return activity_sims["built", keys]

    def simulation(k: int):
        """A simulation holding k model copies (cached)."""
        if k not in batches:
            copies = [model_class(container=f"pynlme{j}") for j in range(k)]
            sim = sund.Simulation(models=copies, time_unit=unit)
            sim.set_options(options)
            batches[k] = sim
        return batches[k]

    def values_for(P, X0, times, sim=None):
        """Observed feature values (len(times), k, n_observed) for k parameter sets."""
        k = len(P)
        grid = times if times[0] == t0 else np.concatenate([[t0], times])
        sim = simulation(k) if sim is None else sim
        sim.simulate(
            time_vector=grid, parameter_values=P.ravel(), state_values=X0.ravel()
        )
        # Columns: model copy j, feature i at j * n_f + i.
        values = np.asarray(sim.feature_values)[:, : k * n_f].reshape(
            len(grid), k, n_f
        )[:, :, feature_index]
        return values if times[0] == t0 else values[1:]

    def model_fn(phi, x, v=None):
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        t = x[:, 0]
        n = len(t)
        which = (
            _feature_rows(x, feature_column, len(observed))
            if multiple
            else np.zeros(n, dtype=int)
        )
        phi = np.asarray(phi, dtype=float)
        if phi.ndim == 1:
            phi = np.broadcast_to(phi, (n, phi.size))
        if v is not None:
            v = np.asarray(v, dtype=float)
            v = np.broadcast_to(
                v.reshape(1, -1) if v.ndim <= 1 else v, (n, v.shape[-1])
            )

        # Systems are runs of identical (phi, v) rows.
        key = phi if v is None else np.hstack([phi, v])
        new = np.r_[True, np.any(key[1:] != key[:-1], axis=1)]
        system = np.cumsum(new) - 1
        first = np.flatnonzero(new)
        P = np.tile(p_default, (len(first), 1))
        P[:, p_idx] = phi[first]
        X0 = np.tile(x_default, (len(first), 1))
        if v is not None:
            X0[:, vs_idx] = v[first, :n_vs]
            P[:, vp_idx] = v[first, n_vs : n_vs + len(vp_idx)]

        out = np.empty(n)
        if activities is not None:
            built = []  # (system, key, builder): batched in containers
            for s, row in enumerate(first):
                key, spec = activity_spec(None if v is None else v[row])
                if is_builder(spec):
                    built.append((s, key, spec))
                    continue
                rows = system == s  # plain activities: one subject at a time
                times = np.unique(t[rows])
                f = values_for(
                    P[s : s + 1], X0[s : s + 1], times, activity_simulation(key, spec)
                )
                out[rows] = f[np.searchsorted(times, t[rows]), 0, which[rows]]
            # Subjects of the same experiment together: a batch then shares its
            # dose and switch times, at which the integrator restarts.
            built.sort(key=lambda item: (str(item[1]), item[0]))
            for start in range(0, len(built), batch_size):
                chunk = built[start : start + batch_size]
                systems = np.array([s for s, _, _ in chunk])
                rows = np.isin(system, systems)
                times = np.unique(t[rows])
                sim = built_simulation(
                    tuple(k for _, k, _ in chunk), [b for _, _, b in chunk]
                )
                f = values_for(P[systems], X0[systems], times, sim)
                order = np.argsort(
                    systems
                )  # copy j holds systems[j]; find each row's copy
                copy = order[np.searchsorted(systems[order], system[rows])]
                out[rows] = f[np.searchsorted(times, t[rows]), copy, which[rows]]
            return out
        for start in range(0, len(first), batch_size):
            stop = min(start + batch_size, len(first))
            rows = (system >= start) & (system < stop)
            times = np.unique(t[rows])
            f = values_for(P[start:stop], X0[start:stop], times)
            out[rows] = f[
                np.searchsorted(times, t[rows]), system[rows] - start, which[rows]
            ]
        return out

    return model_fn


def _feature_rows(x: np.ndarray, column: int, n_features: int) -> np.ndarray:
    """Each row's feature index from column ``column`` of x, validated."""
    if x.shape[1] <= column:
        raise ValueError(f"several features need their index in column {column} of X")
    index = x[:, column]
    which = index.astype(int)
    if np.any(which != index) or np.any(which < 0) or np.any(which >= n_features):
        raise ValueError(
            f"feature indices in column {column} of X must be integers "
            f"0..{n_features - 1}"
        )
    return which
