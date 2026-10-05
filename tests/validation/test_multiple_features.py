"""
Several observed features per subject (e.g. a concentration and an effect),
stacked into one response vector with a feature-index column in X, as with
MATLAB's nlmefit. The ODE adapters simulate each subject once and read out
each row's feature.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from pynlme import nlmefit, ode_model

MODELS = Path(__file__).resolve().parents[2] / "validation" / "models"
PARAMS = ["ka", "V", "vmax", "km", "emax", "ec50"]
TRUTH = np.array([1.0, 10.0, 2.0, 1.5, 100.0, 2.0])


def rhs(t, y, phi, dose):
    ka, V, vmax, km = phi[:, 0], phi[:, 1], phi[:, 2], phi[:, 3]
    gut, c = y[:, 0], y[:, 1]
    return np.column_stack([-ka * gut, ka * gut / V - vmax * c / (km + c)])


def y0(phi, dose):
    return np.column_stack([dose[:, 0], np.zeros(len(phi))])


def outputs(y, phi, dose):
    c = y[:, 1]
    return np.column_stack([c, phi[:, 4] * c / (phi[:, 5] + c)])  # conc, effect


def stacked_data(m=12, seed=0):
    """Each subject: conc and effect at the same times, proportional error."""
    rng = np.random.default_rng(seed)
    times = np.array([0.5, 1, 2, 4, 6, 8, 12])
    model = ode_model(rhs, y0, outputs)
    X, y, group, dose = [], [], [], []
    for i in range(m):
        phi = TRUTH * np.exp(np.r_[rng.normal(0, 0.3), 0, 0, rng.normal(0, 0.2), 0, 0])
        d = rng.choice([20.0, 40.0])
        x = np.column_stack([np.tile(times, 2), np.repeat([0, 1], len(times))])
        f = model(phi, x, np.array([d]))
        X.append(x)
        y.append(f * (1 + 0.1 * rng.normal(size=f.size)))
        group.append(np.full(len(f), i))
        dose.append(d)
    return np.vstack(X), np.concatenate(y), np.concatenate(group), np.array(dose)[:, None]


def test_ode_model_reads_out_each_rows_feature():
    single = {k: ode_model(rhs, y0, lambda y, p, d, k=k: outputs(y, p, d)[:, k]) for k in (0, 1)}
    stacked = ode_model(rhs, y0, outputs)
    times = np.array([0.5, 1.0, 3.0])
    x = np.column_stack([np.r_[times, times], [0, 0, 0, 1, 1, 1]])
    out = stacked(TRUTH, x, np.array([30.0]))
    np.testing.assert_allclose(out[:3], single[0](TRUTH, times[:, None], np.array([30.0])))
    np.testing.assert_allclose(out[3:], single[1](TRUTH, times[:, None], np.array([30.0])))
    with pytest.raises(ValueError, match="column 1"):
        stacked(TRUTH, times[:, None], np.array([30.0]))
    with pytest.raises(ValueError, match="0..1"):
        stacked(TRUTH, np.column_stack([times, [0, 2, 1]]), np.array([30.0]))


def test_sund_model_reads_out_each_rows_feature():
    sund = pytest.importorskip("sund")
    from pynlme import sund_model

    sund.install_model(os.path.relpath(MODELS / "mm_pkpd_sund.txt"))
    X, y, group, dose = stacked_data(m=3)
    reference = ode_model(rhs, y0, outputs, rtol=1e-12, atol=1e-14)
    for model in (sund_model("mm_pkpd", ["conc", "effect"], PARAMS, v_states=["gut"]),
                  sund_model("mm_pkpd", ["conc", "effect"], PARAMS, v_states=["gut"], batch_size=1)):
        # Full vectorization: one row of phi and V per observation, three subjects.
        phi = np.tile(TRUTH, (len(y), 1)) * (1 + 0.1 * group[:, None])
        v = dose[group]
        np.testing.assert_allclose(model(phi, X, v), reference(phi, X, v), rtol=1e-6)
    with pytest.raises(ValueError, match="unknown SUND features"):
        sund_model("mm_pkpd", ["conc", "response"], PARAMS, v_states=["gut"])


@pytest.mark.slow
def test_stacked_fit_recovers_both_features():
    X, y, group, dose = stacked_data()
    model = ode_model(rhs, y0, outputs, rtol=1e-8, atol=1e-10)
    beta0 = np.log(TRUTH * np.array([1.3, 0.8, 1.2, 0.8, 1.2, 0.8]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beta, psi, stats, b = nlmefit(X, y, group, dose, model, beta0, ParamTransform=[1] * 6,
                                      REParamsSelect=[1, 4], ErrorModel="proportional", Vectorization="Full")
    # 12 subjects, 10% error: the effect parameters come only from the effect rows.
    np.testing.assert_allclose(np.exp(beta), TRUTH, rtol=0.25)
    assert stats.errorparam[0] == pytest.approx(0.1, rel=0.3)
    assert b.shape == (2, 12)
