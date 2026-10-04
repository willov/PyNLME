"""pynlme.r.nlme: the R calls of validation/r/nlme_references.R, run in Python."""

import numpy as np
import pytest

from pynlme.r import (
    AIC,
    BIC,
    VarCorr,
    fitted,
    fixef,
    logLik,
    nlme,
    pdDiag,
    ranef,
    resid,
    summary,
    vcov,
)
from tests.validation.cases import load_csv, load_reference

THEOPH = ("conc ~ dose * exp(lKe + lKa - lCl) * (exp(-exp(lKe) * time) - exp(-exp(lKa) * time))"
          " / (exp(lKa) - exp(lKe))")


def _theoph_wt():
    d = load_csv("theoph.csv")
    return {**d, "wtc": d["wt"] - 70}


CASES = {
    "indometh_lme": lambda: nlme(
        "conc ~ SSbiexp(time, p1, p2, p3, p4)", data=load_csv("indometh.csv"),
        fixed="p1 + p2 + p3 + p4 ~ 1", random=pdDiag("p1 + p2 + p3 + p4 ~ 1"), groups="~subject",
        start={"p1": 0.5, "p2": -1.3, "p3": 2.8, "p4": 0.8}),
    "theoph_lme": lambda: nlme(
        THEOPH, data=load_csv("theoph.csv"), fixed="lKe + lKa + lCl ~ 1",
        random=pdDiag("lKa + lCl ~ 1"), groups="~subject", start={"lKe": -2.5, "lKa": 0.5, "lCl": -3.2}),
    "theoph_wt_lme": lambda: nlme(
        THEOPH, data=_theoph_wt(), fixed=["lKe ~ 1", "lKa ~ 1", "lCl ~ wtc"],
        random=pdDiag("lKa + lCl ~ 1 | subject"), start=[-2.5, 0.5, -3.2, 0]),
    "orange_lme": lambda: nlme(
        "circumference ~ Asym / (1 + exp(-(age - xmid) / scal))", data=load_csv("orange.csv"),
        fixed="Asym + xmid + scal ~ 1", random="Asym ~ 1", groups="~tree",
        start={"Asym": 192, "xmid": 727, "scal": 348}),
}


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    return CASES[request.param](), load_reference(f"r_nlme_{request.param}.json")


def test_matches_r(case):
    fit, ref = case
    assert list(fixef(fit)) == ref["beta_names"]
    np.testing.assert_allclose(list(fixef(fit).values()), ref["beta"], rtol=1e-4)
    np.testing.assert_allclose(list(fit["se_fixef"].values()), ref["sebeta"], rtol=1e-3)
    assert fit["call"]["re_names"] == list(np.atleast_1d(ref["re_names"]))
    psi = np.atleast_2d(ref["psi"])
    np.testing.assert_allclose(fit["psi"]["matrix"], psi, rtol=1e-4, atol=1e-4 * psi.diagonal().max())
    assert fit["sigma"] == pytest.approx(ref["sigma"], rel=1e-4)
    assert logLik(fit) == pytest.approx(ref["logl"], abs=1e-3)
    b = np.array([list(ranef(fit)[g].values()) for g in sorted(ranef(fit))]).T
    ref_b = np.atleast_2d(ref["b"])
    np.testing.assert_allclose(b, ref_b, atol=1e-4 * np.abs(ref_b).max())


def test_accessors(case):
    fit, _ = case
    k = len(fixef(fit)) + len(fit["psi"]["names"]) + 1  # fixed, diagonal psi (all diag cases), sigma
    if fit["call"]["structure"] == "diag":
        assert AIC(fit) == pytest.approx(-2 * logLik(fit) + 2 * k)
        assert BIC(fit) == pytest.approx(-2 * logLik(fit) + np.log(fit["call"]["n_obs"]) * k)
    assert vcov(fit).shape == (len(fixef(fit)),) * 2
    assert len(fitted(fit)) == fit["call"]["n_obs"]
    assert VarCorr(fit)["Residual"]["StdDev"] == fit["sigma"]
    text = summary(fit)
    assert "Number of Groups" in text and repr(text) == str(text)


def test_full_covariance_and_user_function():
    """R's default random structure is pdSymm; functions in the caller's scope work."""
    def logistic(t, a, m, s):
        return a / (1 + np.exp(-(t - m) / s))

    fit = nlme("circumference ~ logistic(age, Asym, xmid, scal)", data=load_csv("orange.csv"),
               fixed="Asym + xmid + scal ~ 1", random="Asym + xmid ~ 1 | tree",
               start={"Asym": 192, "xmid": 727, "scal": 348})
    # R: nlme(..., random = Asym + xmid ~ 1) gives logLik -130.9016, corr 0.771.
    assert logLik(fit) == pytest.approx(-130.9016, abs=2e-3)
    assert VarCorr(fit)["Corr"][1, 0] == pytest.approx(0.771, abs=1e-3)
    assert "Log-Cholesky" in summary(fit)
    np.testing.assert_allclose(fitted(fit) + resid(fit), load_csv("orange.csv")["circumference"])
    np.testing.assert_allclose(resid(fit, type="pearson"), resid(fit) / fit["sigma"])


def test_polars():
    pl = pytest.importorskip("polars")
    d = load_csv("orange.csv")
    a = CASES["orange_lme"]()
    b = nlme("circumference ~ Asym / (1 + exp(-(age - xmid) / scal))", data=pl.DataFrame(d),
             fixed="Asym + xmid + scal ~ 1", random="Asym ~ 1", groups="~tree",
             start={"Asym": 192, "xmid": 727, "scal": 348})
    assert fixef(a) == pytest.approx(fixef(b))


def test_errors():
    d = load_csv("orange.csv")
    kw = {"data": d, "fixed": "Asym + xmid + scal ~ 1", "start": {"Asym": 192, "xmid": 727, "scal": 348}}
    model = "circumference ~ Asym / (1 + exp(-(age - xmid) / scal))"
    with pytest.raises(ValueError, match="grouping"):
        nlme(model, random="Asym ~ 1", **kw)
    with pytest.raises(NameError, match="foo"):
        nlme("circumference ~ foo(age, Asym)", random="Asym ~ 1", groups="~tree", **kw)
    with pytest.raises(ValueError, match="start"):
        nlme(model, random="Asym ~ 1", groups="~tree", **{**kw, "start": {"Asym": 192}})
    with pytest.raises(NotImplementedError):
        nlme(model, random="Asym ~ age", groups="~tree", **kw)
    with pytest.raises(NotImplementedError):
        nlme(model, random="Asym ~ 1", groups="~tree", **{**kw, "fixed": ["Asym ~ log(age)", "xmid + scal ~ 1"]})


def test_predict():
    from pynlme.r import predict

    fit = CASES["orange_lme"]()
    np.testing.assert_allclose(predict(fit), fitted(fit), rtol=1e-8)
    new = {"tree": np.array([1.0, 42.0]), "age": np.array([1000.0, 1000.0])}
    ind, pop = predict(fit, new), predict(fit, new, level=0)
    assert np.isnan(ind[1])  # tree not in the fit: NA in R
    f = fixef(fit)
    assert pop[1] == pytest.approx(f["Asym"] / (1 + np.exp(-(1000 - f["xmid"]) / f["scal"])))
    assert ind[0] == pytest.approx((f["Asym"] + ranef(fit)[1]["Asym"]) / (1 + np.exp(-(1000 - f["xmid"]) / f["scal"])))
