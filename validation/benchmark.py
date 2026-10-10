"""
Time pynlme on the validation cases against R (nlme, saemix) and MATLAB.

Run from the repository root:  uv run python validation/benchmark.py
R timings come from validation/r/benchmark.R (best of 20 runs for nlme, 5 for
saemix); MATLAB timings are the single runs recorded in the reference files.
pynlme times are the best of several runs, with the default SinglePhi model
calls and with Vectorization="Full".
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pynlme import nlmefit, nlmefitsa  # noqa: E402
from tests.validation.cases import (  # noqa: E402
    indometh,
    linear_sim,
    load_reference,
    orange,
    theoph,
    vectorized,
)

CASES = {"indometh": indometh, "theoph": theoph, "orange": orange, "linear": linear_sim}


def best_of(n, fn) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def matlab_time(name: str, method: str, options: str = "tight") -> float:
    if method == "saem":
        return float(
            np.mean(
                [
                    load_reference(f"matlab_{name}_saem_seed{s}.json")["elapsed_s"]
                    for s in range(1, 6)
                ]
            )
        )
    return load_reference(f"matlab_{name}_{method}_{options}.json")["elapsed_s"]


# nlmeControl of validation/r/benchmark.R
R_CONTROL = {"compat": "r", "max_iter": 200, "pnls_tol": 1e-6, "tol_x": 1e-7}


def kernels(on: bool):
    """Switch the Rust kernels (pynlme._kernels) on or off; returns
    whether they are installed."""
    import pynlme.lindstrom_bates as lb

    if not hasattr(kernels, "installed"):
        kernels.installed = lb._kernels
    lb._kernels = kernels.installed if on else None
    return kernels.installed is not None


def fit(case, f=nlmefit, **kw):
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


def main() -> None:
    warnings.simplefilter("ignore")
    if "--markdown" in sys.argv:
        return markdown()
    r = load_reference("r_timings.json")
    rows = []
    for name, factory in CASES.items():
        c = factory()
        cv = vectorized(c)
        # Like for like: each mode follows its reference's iteration and options.
        rows.append(
            (
                name,
                "LME, compat matlab",
                best_of(5, fit(c)),
                best_of(5, fit(cv)),
                "",
                f"MATLAB {matlab_time(name, 'lme', 'default'):.2f}",
            )
        )
        rows.append(
            (
                name,
                "LME, compat r",
                best_of(5, fit(c, **R_CONTROL)),
                best_of(5, fit(cv, **R_CONTROL)),
                f"R nlme {r['nlme'][name]:.3f}",
                "",
            )
        )
        rows.append(
            (
                name,
                "nlmefit FOCE",
                best_of(1, fit(c, ApproximationType="FOCE")),
                best_of(1, fit(cv, ApproximationType="FOCE")),
                "",
                f"MATLAB {matlab_time(name, 'foce'):.2f}",
            )
        )
        if name != "indometh":  # not a usable SAEM case (see README)
            rows.append(
                (
                    name,
                    "nlmefitsa",
                    best_of(1, fit(c, nlmefitsa, random_state=1)),
                    best_of(2, fit(cv, nlmefitsa, random_state=1)),
                    "",
                    f"MATLAB {matlab_time(name, 'saem'):.2f}",
                )
            )
            rows.append(
                (
                    name,
                    "nlmefitsa 1 chain",
                    best_of(1, fit(c, nlmefitsa, random_state=1, NChains=1)),
                    best_of(2, fit(cv, nlmefitsa, random_state=1, NChains=1)),
                    f"saemix {r['saemix'][name]:.3f}",
                    "",
                )
            )

    print(
        f"{'case':<10}{'method':<20}{'SinglePhi (s)':>14}{'Full (s)':>10}   reference "
        "(s)"
    )
    for name, method, t1, t2, ref_r, ref_m in rows:
        refs = ", ".join(x for x in (ref_r, ref_m) if x)
        print(f"{name:<10}{method:<20}{t1:>14.3f}{t2:>10.3f}   {refs}")


def markdown() -> None:
    """The README's speed table (Vectorization="Full"), in milliseconds."""
    r = load_reference("r_timings.json")
    has_kernels = kernels(True)

    def ms(t):
        return f"{1000 * t:.0f}" if t >= 0.1 else f"{1000 * t:.1f}"

    print(
        "| Case | MATLAB mode | MATLAB mode, Rust kernels | MATLAB | R mode | R mode, "
        "Rust kernels | R `nlme` |"
        " FOCE | MATLAB FOCE | SAEM, 1 chain | `saemix` |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, factory in CASES.items():
        cv = vectorized(factory())
        kernels(False)
        lme_m = best_of(5, fit(cv))
        lme_r = best_of(5, fit(cv, **R_CONTROL))
        foce = best_of(1, fit(cv, ApproximationType="FOCE"))
        saem = (
            best_of(2, fit(cv, nlmefitsa, random_state=1, NChains=1))
            if name != "indometh"
            else None
        )
        lme_mk = lme_rk = None
        if has_kernels:
            kernels(True)
            lme_mk = best_of(5, fit(cv))
            lme_rk = best_of(5, fit(cv, **R_CONTROL))
        cells = [
            name,
            ms(lme_m),
            "–" if lme_mk is None else ms(lme_mk),
            ms(matlab_time(name, "lme", "default")),
            ms(lme_r),
            "–" if lme_rk is None else ms(lme_rk),
            ms(r["nlme"][name]),
            ms(foce),
            ms(matlab_time(name, "foce")),
            "–" if saem is None else ms(saem),
            "–" if saem is None else ms(r["saemix"][name]),
        ]
        print("| " + " | ".join(cells) + " |")
    kernels(True)


if __name__ == "__main__":
    main()
