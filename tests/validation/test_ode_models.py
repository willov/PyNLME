"""
Models defined by ODEs: SciPy per group, pynlme.ode_model (batched), and
compiled SUND models through pynlme.sund_model.

The Michaelis-Menten PK model (validation/make_mm_dataset.py) has no closed
form; MATLAB's reference solves it with ode45 per group. ODE solvers agree to
~1e-9 relative at these tolerances. That noise enters the finite-difference
Jacobians, and the Lindstrom-Bates iteration then stops once changes reach the
noise level (~1e-4), so estimates are compared with rtol 1e-3 plus atol 1e-3
(on the log scale; standard errors are ~0.05).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pynlme import nlmefit

from .cases import load_reference, michaelis_menten_batched, mm_pk, theoph, theoph_ode

MODELS = Path(__file__).resolve().parents[2] / "validation" / "models"


def _compare_to_matlab(beta, psi, stats):
    ref = load_reference("matlab_mm_pk_lme_tight.json")
    np.testing.assert_allclose(beta, ref["beta"], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.diag(psi), np.diag(np.atleast_2d(ref["psi"])), rtol=1e-2)
    np.testing.assert_allclose(stats.errorparam, np.ravel(ref["stats"]["errorparam"]), rtol=1e-3)
    assert stats.logl == pytest.approx(ref["stats"]["logl"], abs=5e-3)


@pytest.mark.slow
def test_ode_solution_reproduces_closed_form_fit():
    """Theophylline's model solved as an ODE gives the closed-form fit."""
    a, b = theoph(), theoph_ode()
    fit_a = nlmefit(a.X, a.y, a.group, a.V, a.modelfun, a.beta0, **a.kwargs)
    fit_b = nlmefit(b.X, b.y, b.group, b.V, b.modelfun, b.beta0, **b.kwargs)
    np.testing.assert_allclose(fit_b[0], fit_a[0], rtol=1e-3)
    np.testing.assert_allclose(np.diag(fit_b[1]), np.diag(fit_a[1]), rtol=1e-2)
    assert fit_b[2].logl == pytest.approx(fit_a[2].logl, abs=5e-3)


def test_batched_ode_model_matches_matlab():
    c = mm_pk()
    beta, psi, stats, _ = nlmefit(
        c.X, c.y, c.group, c.V, michaelis_menten_batched(), c.beta0, **c.kwargs, Vectorization="Full"
    )
    _compare_to_matlab(beta, psi, stats)


def test_ode_model_single_and_full_calls_agree():
    c = mm_pk()
    model = michaelis_menten_batched()
    phi = np.array([1.0, 10.0, 2.0, 1.5])
    rows = c.group == c.group[0]
    single = model(phi, c.X[rows][:, None], c.V[:1])
    two = np.concatenate([c.X[rows], c.X[rows]])[:, None]
    phis = np.vstack([np.tile(phi, (rows.sum(), 1)), np.tile(phi * 1.1, (rows.sum(), 1))])
    doses = np.full((len(two), 1), c.V[0])
    full = model(phis, two, doses)
    np.testing.assert_allclose(full[: rows.sum()], single, rtol=1e-8)
    assert not np.allclose(full[rows.sum():], single)


def test_sund_model_matches_matlab():
    sund = pytest.importorskip("sund")
    from pynlme import sund_model

    # sund.install_model only accepts relative paths (it globs the argument).
    sund.install_model(os.path.relpath(MODELS / "mm_pk_sund.txt"))
    sim = sund.Simulation(models=[sund.load_model("mm_pk")], time_unit="h")
    model = sund_model(sim, "conc", ["ka", "V", "vmax", "km"], v_states=["gut"])
    c = mm_pk()
    beta, psi, stats, _ = nlmefit(c.X, c.y, c.group, c.V, model, c.beta0, **c.kwargs, Vectorization="Full")
    _compare_to_matlab(beta, psi, stats)
