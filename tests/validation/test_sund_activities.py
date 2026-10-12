"""
Model inputs from SUND activities (doses, infusions), shared per experiment or
built per subject, checked against closed-form one-compartment solutions.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from pynlme import nlmefit

sund = pytest.importorskip("sund")
from pynlme import sund_model  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "validation" / "models"
PARAMS = ["ka", "V", "CL"]


@pytest.fixture(scope="module", autouse=True)
def installed():
    sund.install_model(os.path.relpath(MODELS / "pk_inputs_sund.txt"))


def oral(times, ka, V, CL, doses):
    """Concentration after oral doses {time: amount}."""
    ke = CL / V
    c = np.zeros_like(times)
    for td, d in doses.items():
        s = np.clip(times - td, 0, None)
        c += np.where(
            times > td,
            d * ka / (V * (ka - ke)) * (np.exp(-ke * s) - np.exp(-ka * s)),
            0,
        )
    return c


def infusion(times, V, CL, rate, duration):
    """Concentration during and after a constant-rate infusion from time 0."""
    ke = CL / V
    during = rate / CL * (1 - np.exp(-ke * np.minimum(times, duration)))
    return np.where(
        times <= duration, during, during * np.exp(-ke * (times - duration))
    )


def experiments():
    oral_doses = sund.Activity(time_unit="h")
    oral_doses.add_state_manipulation(
        name="gut", t=[0.0, 12.0], f=[100.0, 100.0], mode="add"
    )
    iv = sund.Activity(time_unit="h")  # 0.5 mg/kg/h for 2 h
    iv.add_output(
        type="piecewise_constant",
        name="infusion_per_kg_in",
        t=[0.0, 2.0],
        f=[0.0, 0.5, 0.0],
    )
    return {1: oral_doses, 2: iv}


TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 16.0, 24.0])
SUBJECTS = [  # phi (ka, V, CL), V row (BW, experiment)
    (np.array([1.2, 10.0, 2.0]), np.array([70.0, 1.0])),
    (np.array([0.8, 12.0, 1.5]), np.array([50.0, 2.0])),
    (np.array([1.0, 9.0, 2.5]), np.array([90.0, 2.0])),
]


def builders():
    """The experiments as functions of the container: sund_model batches subjects in
    containers, each copy with the activities built for it."""

    def oral_doses(container):
        a = sund.Activity(time_unit="h", container=container)
        a.add_state_manipulation(
            name="gut", t=[0.0, 12.0], f=[100.0, 100.0], mode="add"
        )
        return a

    def iv(container):
        a = sund.Activity(time_unit="h", container=container)
        a.add_output(
            type="piecewise_constant",
            name="infusion_per_kg_in",
            t=[0.0, 2.0],
            f=[0.0, 0.5, 0.0],
        )
        return a

    return {1: oral_doses, 2: iv}


def test_activities_per_experiment():
    model = sund_model(
        "pk_inputs", "conc", PARAMS, v_parameters=["BW"], activities=experiments()
    )
    subjects = [  # phi (ka, V, CL), V row (BW, experiment)
        (np.array([1.2, 10.0, 2.0]), np.array([70.0, 1.0])),
        (np.array([0.8, 12.0, 1.5]), np.array([50.0, 2.0])),
        (np.array([1.0, 9.0, 2.5]), np.array([90.0, 2.0])),
    ]
    # Vectorization="Full": one row of phi and V per observation, all subjects.
    phi = np.repeat([s[0] for s in subjects], len(TIMES), axis=0)
    v = np.repeat([s[1] for s in subjects], len(TIMES), axis=0)
    x = np.tile(TIMES, len(subjects))[:, None]
    out = model(phi, x, v).reshape(len(subjects), -1)
    (ka, V, CL), _ = subjects[0]
    np.testing.assert_allclose(
        out[0], oral(TIMES, ka, V, CL, {0.0: 100.0, 12.0: 100.0}), rtol=1e-6
    )
    for i in (1, 2):
        (ka, V, CL), (bw, _) = subjects[i]
        np.testing.assert_allclose(
            out[i], infusion(TIMES, V, CL, 0.5 * bw, 2.0), rtol=1e-6, atol=1e-12
        )
    # One subject at a time (SinglePhi), in another order: the same values.
    np.testing.assert_allclose(
        model(subjects[2][0], TIMES[:, None], subjects[2][1]), out[2], rtol=1e-12
    )
    np.testing.assert_allclose(
        model(subjects[0][0], TIMES[:, None], subjects[0][1]), out[0], rtol=1e-12
    )


def test_activities_must_be_per_experiment():
    with pytest.raises(TypeError, match="per kg"):
        sund_model("pk_inputs", "conc", PARAMS, activities=lambda v: experiments()[1])


def test_activity_errors():
    model = sund_model(
        "pk_inputs", "conc", PARAMS, v_parameters=["BW"], activities=experiments()
    )
    with pytest.raises(KeyError, match="experiment 3"):
        model(np.array([1.0, 10.0, 2.0]), TIMES[:, None], np.array([70.0, 3.0]))
    with pytest.raises(ValueError, match="column 1 of V"):
        model(np.array([1.0, 10.0, 2.0]), TIMES[:, None], np.array([70.0]))
    with pytest.raises(ValueError, match="name of an installed"):
        sund_model(
            sund.Simulation(models=[sund.load_model("pk_inputs")], time_unit="h"),
            "conc",
            PARAMS,
            activities=experiments(),
        )


@pytest.mark.parametrize(
    "activities", [experiments, builders], ids=["plain", "builders"]
)
def test_fit_with_two_experiments(activities):
    """Subjects from an oral and an infusion experiment, fitted together."""
    rng = np.random.default_rng(1)
    truth = np.array([1.0, 10.0, 2.0])
    X, y, group, V = [], [], [], []
    for i in range(12):
        ka, vol, cl = truth * np.exp(np.r_[0, rng.normal(0, 0.2), rng.normal(0, 0.2)])
        bw, exp_id = rng.uniform(50, 90), 1 + i % 2
        c = (
            oral(TIMES, ka, vol, cl, {0.0: 100.0, 12.0: 100.0})
            if exp_id == 1
            else infusion(TIMES, vol, cl, 0.5 * bw, 2.0)
        )
        X.append(TIMES)
        y.append(c * (1 + 0.05 * rng.normal(size=c.size)))
        group.append(np.full(len(TIMES), i))
        V.append([bw, exp_id])
    model = sund_model(
        "pk_inputs", "conc", PARAMS, v_parameters=["BW"], activities=activities()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beta, psi, stats, b = nlmefit(
            np.concatenate(X),
            np.concatenate(y),
            np.concatenate(group),
            np.array(V),
            model,
            np.log(truth * 1.2),
            ParamTransform=[1, 1, 1],
            REParamsSelect=[2, 3],
            ErrorModel="proportional",
            Vectorization="Full",
        )
    np.testing.assert_allclose(np.exp(beta), truth, rtol=0.15)
    assert stats.errorparam[0] == pytest.approx(0.05, rel=0.3)


@pytest.mark.parametrize("batch_size", [16, 2, 1])
def test_activity_builders_batched(batch_size):
    subjects = SUBJECTS + [(np.array([1.5, 8.0, 1.0]), np.array([60.0, 1.0]))]
    model = sund_model(
        "pk_inputs",
        "conc",
        PARAMS,
        v_parameters=["BW"],
        activities=builders(),
        batch_size=batch_size,
    )
    phi = np.repeat([s[0] for s in subjects], len(TIMES), axis=0)
    v = np.repeat([s[1] for s in subjects], len(TIMES), axis=0)
    out = model(phi, np.tile(TIMES, len(subjects))[:, None], v).reshape(
        len(subjects), -1
    )
    for i, ((ka, V, CL), (bw, exp_id)) in enumerate(subjects):
        expected = (
            oral(TIMES, ka, V, CL, {0.0: 100.0, 12.0: 100.0})
            if exp_id == 1
            else infusion(TIMES, V, CL, 0.5 * bw, 2.0)
        )
        np.testing.assert_allclose(out[i], expected, rtol=1e-6, atol=1e-12)
    plain = sund_model(
        "pk_inputs", "conc", PARAMS, v_parameters=["BW"], activities=experiments()
    )
    np.testing.assert_allclose(
        plain(phi, np.tile(TIMES, len(subjects))[:, None], v).reshape(
            len(subjects), -1
        ),
        out,
        rtol=1e-7,
        atol=1e-12,
    )
