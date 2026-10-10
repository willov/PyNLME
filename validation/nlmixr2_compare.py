"""
pynlme's FOCE and SAEM against nlmixr2 (FOCEi, SAEM) and MATLAB's FOCE on the
benchmark cases: time and the exact log-likelihood of each program's estimates.

Run from the repository root, after validation/r/nlmixr2_benchmark.R and
validation/matlab/benchmark.m:  uv run python validation/nlmixr2_compare.py

The programs report different likelihood approximations (FOCE, FOCEi, linearised),
so each program's estimates are scored with the same exact log-likelihood
(importance sampling, tests/validation/likelihood.py): higher is better. With an
additive error model FOCEi and FOCE are the same method.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pynlme import nlmefit, nlmefitsa  # noqa: E402
from tests.validation.cases import load_reference, vectorized  # noqa: E402
from tests.validation.likelihood import loglik_at  # noqa: E402
from validation.benchmark import CASES, best_of, matlab_time  # noqa: E402


def timed(fn, n=3):
    result = fn()
    return result, best_of(n, fn)


def nlmixr2_estimates(fit: dict):
    theta = fit["theta"]
    beta = np.array([v for k, v in theta.items() if k != "add.sd"])
    return beta, np.atleast_2d(np.array(fit["omega"], float)), theta["add.sd"]


def main() -> None:
    warnings.simplefilter("ignore")
    nlmixr2 = load_reference("nlmixr2_results.json")
    rows = []
    for name, factory in CASES.items():
        case = vectorized(factory())

        def call(f, case=case, **kw):
            return lambda: f(
                case.X,
                case.y,
                case.group,
                case.V,
                case.modelfun,
                case.beta0,
                **case.kwargs,
                **kw,
            )

        fits = []
        (beta, psi, stats, _), t = timed(call(nlmefit, ApproximationType="FOCE"))
        fits.append(("FOCE", "pynlme", t, beta, psi, stats.errorparam))
        ref = load_reference(f"matlab_{name}_foce_default.json")
        fits.append(
            (
                "FOCE",
                "MATLAB `nlmefit`",
                matlab_time(name, "foce"),
                np.ravel(ref["beta"]),
                np.atleast_2d(np.array(ref["psi"], float)),
                ref["stats"]["errorparam"],
            )
        )
        fits.append(("FOCE", "nlmixr2 FOCEi", None, None, None, None))
        if name != "indometh":  # not a usable SAEM case (see README)
            (beta, psi, stats, _), t = timed(call(nlmefitsa, random_state=1, NChains=1))
            fits.append(("SAEM", "pynlme `nlmefitsa`", t, beta, psi, stats.errorparam))
            fits.append(("SAEM", "nlmixr2 SAEM", None, None, None, None))

        for method, program, t, beta, psi, errorparam in fits:
            if program.startswith("nlmixr2"):
                fit = nlmixr2[name]["focei" if method == "FOCE" else "saem"]
                if "error" in fit:
                    rows.append((name, method, program, None, None, fit["error"]))
                    continue
                t = fit["elapsed_s"]
                beta, psi, errorparam = nlmixr2_estimates(fit)
            ll = loglik_at(case, beta, psi, "constant", errorparam)
            rows.append((name, method, program, t, ll, ""))

    print("| Case | Method | Program | Time (ms) | Exact log-likelihood |")
    print("|---|---|---|---|---|")
    for name, method, program, t, ll, note in rows:
        t_text = "–" if t is None else f"{1000 * t:.0f}"
        ll_text = note if ll is None else f"{ll:.3f}"
        print(f"| {name} | {method} | {program} | {t_text} | {ll_text} |")
    v = nlmixr2["versions"]
    print(
        f"\nnlmixr2 {v['nlmixr2']} (nlmixr2est {v['nlmixr2est']}, rxode2 "
        f"{v['rxode2']}), {v['r']}"
    )


if __name__ == "__main__":
    main()
