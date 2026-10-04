"""pynlme.fit: named parameters, dict/dataframe data, dict results."""

import numpy as np
import pytest

from pynlme import fit, nlmefit
from tests.validation.cases import load_csv, per_group


def one_compartment(p, x, v):
    ke, ka, cl, t = p["ke"], p["ka"], p["cl"], x["time"]
    return v["dose"] * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


START = {"ke": np.exp(-2.5), "ka": np.exp(0.5), "cl": np.exp(-3.2)}
LOG = {"ke": "log", "ka": "log", "cl": "log"}


@pytest.fixture(scope="module")
def theoph():
    return load_csv("theoph.csv")


def _fit(data, **kwargs):
    return fit(one_compartment, data, response="conc", group="subject", predictors=["time"],
               group_predictors=["dose"], start=START, transform=LOG, random=["ka", "cl"], **kwargs)


def test_matches_nlmefit(theoph):
    d = theoph

    def model(phi, t, v):
        ke, ka, cl = phi[0], phi[1], phi[2]
        t = t[:, 0]
        return v[0] * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

    beta, psi, stats, b = nlmefit(d["time"], d["conc"], d["subject"], per_group(d["dose"], d["subject"]),
                                  model, [-2.5, 0.5, -3.2], ParamTransform=[1, 1, 1], REParamsSelect=[2, 3])
    r = _fit(d)
    assert list(r["fixed"]) == ["ke", "ka", "cl"]
    np.testing.assert_allclose(list(r["fixed"].values()), beta, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(r["typical"]["cl"], np.exp(beta[2]), rtol=1e-5)
    np.testing.assert_allclose(r["psi"]["matrix"], psi, rtol=1e-4, atol=1e-8)
    assert r["psi"]["names"] == ["ka", "cl"]
    np.testing.assert_allclose(r["stats"]["logl"], stats.logl, rtol=1e-7)
    np.testing.assert_allclose(list(r["se"].values()), stats.sebeta, rtol=1e-4)
    assert set(r["random_effects"]) == set(np.unique(d["subject"]).tolist())
    first = min(r["random_effects"])
    np.testing.assert_allclose(r["random_effects"][first]["cl"], b[1, 0], rtol=1e-4, atol=1e-7)
    assert r["transform"]["ke"] == "log"


def test_polars_input(theoph):
    pl = pytest.importorskip("polars")
    a = _fit(theoph)
    b = _fit(pl.DataFrame(theoph))
    assert a["fixed"] == pytest.approx(b["fixed"])


def test_saem(theoph):
    r = _fit(theoph, method="SAEM", random_state=1)
    assert r["method"] == "SAEM"
    assert 0.02 < r["typical"]["cl"] < 0.06


def test_errors(theoph):
    with pytest.raises(ValueError, match="random"):
        fit(one_compartment, theoph, "conc", "subject", START, ["time"], ["dose"], random=["v"])
    with pytest.raises(ValueError, match="transform"):
        fit(one_compartment, theoph, "conc", "subject", START, ["time"], ["dose"], transform={"ke": "sqrt"})
    with pytest.raises(KeyError, match="weight"):
        fit(one_compartment, theoph, "conc", "subject", START, ["weight"], ["dose"])
    bad = dict(theoph, dose=theoph["dose"] + np.arange(len(theoph["dose"])))
    with pytest.raises(ValueError, match="not constant within groups"):
        fit(one_compartment, bad, "conc", "subject", START, ["time"], ["dose"])
