"""pynlme.r saemix interface against R saemix (validation/r/saemix_references.R)."""

import numpy as np
import pytest

from pynlme.r import (
    coef,
    eta,
    logLik,
    phi,
    predict,
    psi,
    saemix,
    saemixData,
    saemixModel,
    summary,
)
from tests.validation.cases import load_csv, load_reference


def one_compartment(psi, id, xidep):  # noqa: A002 (saemix's argument name)
    dose, t = xidep[:, 0], xidep[:, 1]
    ke, ka, cl = psi[id, 0], psi[id, 1], psi[id, 2]
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def one_compartment_v(psi, id, xidep):  # noqa: A002
    dose, t = xidep[:, 0], xidep[:, 1]
    ka, v, cl = psi[id, 0], psi[id, 1], psi[id, 2]
    k = cl / v
    return dose * ka / (v * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))


@pytest.fixture(scope="module")
def theoph():
    return saemixData(
        load_csv("theoph.csv"),
        name_group="subject",
        name_predictors=["dose", "time"],
        name_response="conc",
        name_covariates=["wt"],
    )


@pytest.fixture(scope="module")
def runs(theoph):
    model = saemixModel(
        one_compartment,
        psi0={"ke": np.exp(-2.5), "ka": np.exp(0.5), "cl": np.exp(-3.2)},
        transform_par=[1, 1, 1],
        covariance_model=np.diag([0, 1, 1]),
    )
    return [saemix(model, theoph, {"seed": s}) for s in range(3)]


def test_matches_saemix(runs):
    ref = load_reference("r_saemix_theoph.json")
    beta = np.mean([np.log(list(r["population"].values())) for r in runs], axis=0)
    omega = np.mean([np.diag(r["omega"]) for r in runs], axis=0)
    sigma = np.mean([r["respar"]["a.1"] for r in runs])
    logl = np.mean([logLik(r) for r in runs])
    # Means over seeds agree within 5 standard errors of the difference.
    se = np.sqrt(np.square(ref["beta_sd"]) * (1 / 10 + 1 / 3))
    assert np.all(np.abs(beta - ref["beta_mean"]) < 5 * se)
    se = np.sqrt(np.square(ref["psi_diag_sd"]) * (1 / 10 + 1 / 3))
    assert np.all(
        np.abs(omega - ref["psi_diag_mean"])
        < 5 * se + 0.02 * np.array(ref["psi_diag_mean"])
    )
    assert sigma == pytest.approx(ref["sigma_mean"], rel=0.01)
    assert logl == pytest.approx(ref["logl_mean"], abs=0.2)


def test_outputs(runs, theoph):
    fit = runs[0]
    assert list(coef(fit)["fixed"]) == ["ke", "ka", "cl"]
    # Delta-method SEs of the natural-scale parameters.
    for n in ("ka", "cl"):
        assert fit["se_population"][n] == pytest.approx(
            fit["population"][n] * fit["se"][n]
        )
    y = theoph["data"]["conc"]
    np.testing.assert_allclose(
        predict(fit, type="icpred"), y - fit["stats"]["ires"], rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(
        predict(fit, type="ppred"), y - fit["stats"]["pres"], rtol=1e-8, atol=1e-12
    )
    assert np.all(np.isfinite(predict(fit, type="ypred")))
    # Modes are close to, but not the same as, the conditional means.
    mode, mean = psi(fit)[1], psi(fit, "mean")[1]
    assert mode["ka"] != mean["ka"] and mode["ka"] == pytest.approx(mean["ka"], rel=0.1)
    assert mode["ke"] == pytest.approx(fit["population"]["ke"])  # no random effect
    assert phi(fit)[1]["ka"] == pytest.approx(np.log(mode["ka"]))
    assert set(eta(fit)[1]) == {"ka", "cl"}
    text = summary(fit)
    assert "importance sampling" in text and "omega2.ka" in text


def test_covariate_model_matches_saemix(theoph):
    """R: saemix(..., covariate.model = matrix(c(0, 0, 1), ncol = 3)), seed 1:
    ka 1.455 (SE 0.311), V 0.439 (0.012), CL 0.116 (0.090),
    beta_wt(CL) -0.0150 (0.0111), a.1 0.797, -2LL 380.39."""
    model = saemixModel(
        one_compartment_v,
        psi0={"ka": [1, 0], "V": [20, 0], "CL": [0.5, 0]},
        transform_par=[1, 1, 1],
        covariate_model=[[0, 0, 1]],
        covariance_model=np.diag([1, 0, 1]),
    )
    fit = saemix(model, theoph, {"seed": 1})
    pop = fit["population"]
    assert pop["ka"] == pytest.approx(1.455, rel=0.05)
    assert pop["V"] == pytest.approx(0.439, rel=0.02)
    assert pop["CL"] == pytest.approx(0.116, rel=0.1)
    assert fit["covariate_effects"]["beta_wt(CL)"] == pytest.approx(-0.0150, rel=0.05)
    assert fit["respar"]["a.1"] == pytest.approx(0.797, rel=0.02)
    assert -2 * logLik(fit) == pytest.approx(380.39, abs=1.0)
    se = fit["se_population"]
    assert se["ka"] == pytest.approx(0.311, rel=0.15)
    assert se["V"] == pytest.approx(0.0118, rel=0.15)
    assert se["CL"] == pytest.approx(0.090, rel=0.15)
    assert fit["se_covariate_effects"]["beta_wt(CL)"] == pytest.approx(0.0111, rel=0.15)
    assert "beta_wt(CL)" in summary(fit)


def test_errors(theoph):
    model = saemixModel(
        one_compartment,
        psi0={"ke": 0.1, "ka": 1.6, "cl": 0.04},
        transform_par=[1, 1, 1],
    )
    with pytest.raises(ValueError, match="unsupported control"):
        saemix(model, theoph, {"nbiter.saem": [10, 10]})
    with pytest.raises(NotImplementedError):
        saemixModel(one_compartment, psi0={"ke": 0.1}, omega_init=np.eye(1))
    with pytest.raises(NotImplementedError):
        saemixModel(
            one_compartment,
            psi0={"a": 1, "b": 1, "c": 1},
            covariance_model=[[1, 1, 0], [1, 1, 0], [0, 0, 1]],
        )
    with pytest.raises(TypeError):
        saemix(theoph, model)
