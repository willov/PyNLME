"""
Time pynlme on the validation cases against R (nlme, saemix) and MATLAB.

Run from the repository root:  uv run python validation/benchmark.py
All programs run on the same machine, on the same cases, start values and
options. R timings come from validation/r/benchmark.R (best of 20 runs for nlme, 5
for saemix), MATLAB timings from validation/matlab/benchmark.m (best of 10 runs for
LME, 3 for FOCE and SAEM, after a warm-up run); pynlme times are the best of
several runs here. `--markdown` prints the README's table (Vectorization="Full")
and the machine's specifications.
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


def matlab_time(name: str, method: str, full: bool = True) -> float:
    """MATLAB's time (validation/matlab/benchmark.m): method lme, foce or saem."""
    timings = load_reference("matlab_timings.json")["timings"][name]
    return timings[f"{method}_{'full' if full else 'single'}"]


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
                f"MATLAB {matlab_time(name, 'lme', False):.3f}/"
                f"{matlab_time(name, 'lme'):.3f}",
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
                f"MATLAB {matlab_time(name, 'foce', False):.3f}/"
                f"{matlab_time(name, 'foce'):.3f}",
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
                    f"MATLAB {matlab_time(name, 'saem', False):.3f}/"
                    f"{matlab_time(name, 'saem'):.3f}",
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
        "(s; MATLAB SinglePhi/Full)"
    )
    for name, method, t1, t2, ref_r, ref_m in rows:
        refs = ", ".join(x for x in (ref_r, ref_m) if x)
        print(f"{name:<10}{method:<20}{t1:>14.3f}{t2:>10.3f}   {refs}")


def saemix_fit(name: str):
    """pynlme.r.saemix on the case as validation/r/benchmark.R runs it in R (saemix's
    defaults, no importance-sampling log-likelihood)."""
    from pynlme.r import saemix, saemixData, saemixModel
    from tests.validation.cases import load_csv

    if name == "theoph":
        d = load_csv("theoph.csv")
        data = {
            "id": d["subject"],
            "dose": d["dose"],
            "time": d["time"],
            "y": d["conc"],
        }
        predictors = ["dose", "time"]

        def model(psi, id, x):
            ke, ka, cl = psi[id, 0], psi[id, 1], psi[id, 2]
            return (
                x[:, 0]
                * ke
                * ka
                / (cl * (ka - ke))
                * (np.exp(-ke * x[:, 1]) - np.exp(-ka * x[:, 1]))
            )

        psi0 = dict(zip(["ke", "ka", "cl"], np.exp([-2.5, 0.5, -3.2]), strict=True))
        transform, covariance = [1, 1, 1], np.diag([0, 1, 1])
    elif name == "orange":
        d = load_csv("orange.csv")
        data = {"id": d["tree"], "age": d["age"], "y": d["circumference"]}
        predictors = ["age"]

        def model(psi, id, x):
            return psi[id, 0] / (1 + np.exp(-(x[:, 0] - psi[id, 1]) / psi[id, 2]))

        psi0 = {"A": 192.0, "x": 727.0, "s": 348.0}
        transform, covariance = [0, 0, 0], np.diag([1, 0, 0])
    else:
        d = load_csv("linear_sim.csv")
        data = {"id": d["id"], "x": d["x"], "y": d["y"]}
        predictors = ["x"]

        def model(psi, id, x):
            return psi[id, 0] + psi[id, 1] * x[:, 0]

        psi0 = {"a": 1.0, "b": 1.0}
        transform, covariance = [0, 0], np.ones((2, 2))
    sdata = saemixData(
        data, name_group="id", name_predictors=predictors, name_response="y"
    )
    smodel = saemixModel(
        model, psi0, transform_par=transform, covariance_model=covariance
    )
    return lambda: saemix(smodel, sdata, {"seed": 1, "ll.is": False})


def specs() -> str:
    """The machine and software the timings were taken on."""
    import platform

    import psutil
    import scipy

    import pynlme

    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in fh
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    r = load_reference("r_timings.json")
    m = load_reference("matlab_timings.json")
    return (
        f"{cpu}, {psutil.cpu_count(logical=False)} cores "
        f"({psutil.cpu_count()} threads), "
        f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM, "
        f"{platform.system()} {platform.release()}. "
        f"pynlme {pynlme.__version__} on Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}; "
        f"MATLAB {m['matlab_version']} (Statistics and Machine Learning Toolbox "
        f"{m['toolbox_version']}, {m['threads']} computational threads); "
        f"{r['r_version']}, nlme {r['nlme_version']}, saemix {r['saemix_version']}."
    )


def markdown() -> None:
    """The README's speed tables (Vectorization="Full"), in milliseconds."""
    r = load_reference("r_timings.json")
    has_kernels = kernels(True)

    def ms(t):
        if t is None:
            return "–"
        return f"{1000 * t:.0f}" if t >= 0.1 else f"{1000 * t:.1f}"

    def both(n, fn):
        """(NumPy, Rust) times: the best of n runs without and with the kernels."""
        kernels(False)
        numpy_time = best_of(n, fn)
        kernels(True)
        return numpy_time, (best_of(n, fn) if has_kernels else None)

    lme, foce, saem = [], [], []
    for name, factory in CASES.items():
        cv = vectorized(factory())
        lme.append(
            [name, *both(10, fit(cv)), matlab_time(name, "lme")]
            + [*both(10, fit(cv, **R_CONTROL)), r["nlme"][name]]
        )
        foce.append(
            [
                name,
                *both(3, fit(cv, ApproximationType="FOCE")),
                matlab_time(name, "foce"),
            ]
        )
        if name != "indometh":  # not a usable SAEM case (see README)
            saem.append(
                [
                    name,
                    *both(3, fit(cv, nlmefitsa, random_state=1, NChains=1)),
                    matlab_time(name, "saem"),
                    best_of(3, saemix_fit(name)),
                    r["saemix"][name],
                ]
            )

    def table(header, rows):
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join([row[0], *map(ms, row[1:])]) + " |")
        print()

    table(
        [
            "LME",
            "pynlme MATLAB mode (NumPy)",
            "pynlme MATLAB mode (Rust)",
            "MATLAB `nlmefit`",
            "pynlme R mode (NumPy)",
            "pynlme R mode (Rust)",
            "R `nlme`",
        ],
        lme,
    )
    table(["FOCE", "pynlme (NumPy)", "pynlme (Rust)", "MATLAB `nlmefit`"], foce)
    table(
        [
            "SAEM",
            "pynlme `nlmefitsa` (NumPy)",
            "pynlme `nlmefitsa` (Rust)",
            "MATLAB `nlmefitsa`",
            "pynlme `pynlme.r.saemix`",
            "R `saemix`",
        ],
        saem,
    )
    print("Machine: " + specs())


if __name__ == "__main__":
    main()
