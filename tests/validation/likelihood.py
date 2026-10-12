"""Exact (importance-sampled) log-likelihood at given parameter values.

Used as a referee when two stochastic fits disagree: the estimates with the
higher exact likelihood are the better ones.
"""

import numpy as np

import pynlme.saem as saem
from pynlme.data_types import NLMEOptions
from pynlme.nlmefit import _build_model, _parse_options
from pynlme.utils import validate_inputs


def loglik_at(
    case,
    beta,
    psi,
    error_model,
    errorparam,
    chains=8,
    iterations=300,
    draws=20000,
    seed=0,
):
    """log L(beta, psi, errorparam): MCMC conditional moments + importance sampling."""
    parsed, matlab_style = _parse_options(case.kwargs)
    options = NLMEOptions(**parsed)
    X, y, g, V, b0 = validate_inputs(case.X, case.y, case.group, case.V, case.beta0)
    model = _build_model(X, y, g, V, case.modelfun, b0, options, matlab_style)
    errorparam = np.ravel(errorparam)
    params = {"proportional": (0.0, errorparam[0]), "combined": tuple(errorparam)}.get(
        error_model, (errorparam[0], 0.0)
    )
    rng = np.random.default_rng(seed)
    s = saem._SAEM(
        model,
        np.asarray(beta, float),
        psi.shape[0] > 1 and np.any(psi - np.diag(np.diag(psi))),
        rng,
        chains,
        saem.ErrorModel(error_model, params),
    )
    r = s.r
    Phi = s.mu()[s.sub]
    F, _ = s.predict(Phi)
    scales = [0.5 * np.ones(r), 0.5 * np.ones(r)]
    m1, m2 = np.zeros((s.m, r)), np.zeros((s.m, r, r))
    burn = iterations // 2
    for k in range(iterations):
        s.mcmc(Phi, F, psi, (2, 2, 2), scales)
        if k >= burn:
            P = Phi.reshape(chains, s.m, r)
            w = 1 / (k - burn + 1)
            m1 += w * (P.mean(axis=0) - m1)
            m2 += w * (np.einsum("cmi,cmj->mij", P, P) / chains - m2)
    cond_cov = m2 - np.einsum("mi,mj->mij", m1, m1)
    return saem._loglik_importance(s, m1, cond_cov, psi, draws, rng)
