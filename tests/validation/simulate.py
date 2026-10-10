"""
Simulated datasets with known parameters, for randomized comparisons against R
and for simulation (bias and coverage) studies.

Each family draws its own true parameters unless they are given, uses
unbalanced sampling, and mirrors a fit in validation/r/fit_manifest.R.
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Truth:
    beta: np.ndarray
    psi: np.ndarray
    sigma: float


@dataclass
class SimulatedDataset:
    name: str
    family: str
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    V: np.ndarray | None  # one value per group, or None
    modelfun: Callable
    beta0: np.ndarray
    truth: Truth
    kwargs: dict = field(default_factory=dict)

    def fit_args(self):
        return self.X, self.y, self.group, self.V, self.modelfun, self.beta0


def oral(phi, t, dose):
    ke, ka, cl = phi
    return dose * ke * ka / (cl * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def logistic(phi, t):
    return phi[0] / (1 + np.exp(-(t - phi[1]) / phi[2]))


def linear(phi, x):
    return phi[0] + phi[1] * x


def emax(phi, c):
    return phi[0] + phi[1] * c / (phi[2] + c)


@dataclass
class Family:
    modelfun: Callable
    kwargs: dict  # nlmefit options (MATLAB names)
    transform: np.ndarray  # ParamTransform codes, to map eta -> phi
    re_index: np.ndarray  # 0-based parameters with random effects
    uses_v: bool
    draw_truth: Callable  # rng -> Truth
    draw_design: Callable  # rng, m -> (list of x arrays, V or None)
    draw_start: Callable  # rng, beta -> beta0
    m_range: tuple[int, int]


def _sd_cov(sd, corr=0.0):
    psi = np.diag(np.asarray(sd) ** 2)
    if len(sd) == 2 and corr:
        psi[0, 1] = psi[1, 0] = corr * sd[0] * sd[1]
    return psi


def _times(rng, m, n_lo, n_hi, lo, hi):
    return [
        np.sort(rng.uniform(lo, hi, rng.integers(n_lo, n_hi + 1))) for _ in range(m)
    ]


FAMILIES = {
    "oral": Family(
        modelfun=oral,
        kwargs={"ParamTransform": [1, 1, 1], "REParamsSelect": [2, 3]},
        transform=np.array([1, 1, 1]),
        re_index=np.array([1, 2]),
        uses_v=True,
        draw_truth=lambda rng: Truth(
            beta=np.log(
                [
                    rng.uniform(0.05, 0.15),
                    rng.uniform(0.8, 2.5),
                    rng.uniform(0.02, 0.06),
                ]
            ),
            psi=_sd_cov(rng.uniform(0.1, 0.5, 2)),
            sigma=rng.uniform(0.3, 0.8),
        ),
        draw_design=lambda rng, m: (
            _times(rng, m, 5, 12, 0.1, 24),
            rng.uniform(3, 6, m),
        ),
        draw_start=lambda rng, beta: beta + rng.normal(0, 0.1, beta.size),
        m_range=(8, 40),
    ),
    "logistic": Family(
        modelfun=logistic,
        kwargs={"REParamsSelect": [1, 2]},
        transform=np.array([0, 0, 0]),
        re_index=np.array([0, 1]),
        uses_v=False,
        draw_truth=lambda rng: (
            lambda b: Truth(
                beta=b,
                psi=_sd_cov(
                    [rng.uniform(0.05, 0.2) * b[0], rng.uniform(0.02, 0.1) * b[1]]
                ),
                sigma=0.03 * b[0],
            )
        )(
            np.array(
                [rng.uniform(100, 300), rng.uniform(400, 900), rng.uniform(200, 400)]
            )
        ),
        draw_design=lambda rng, m: (_times(rng, m, 5, 10, 100, 1600), None),
        draw_start=lambda rng, beta: beta * rng.uniform(0.9, 1.1, beta.size),
        m_range=(6, 30),
    ),
    "linear": Family(
        modelfun=linear,
        kwargs={"CovPattern": np.ones((2, 2))},
        transform=np.array([0, 0]),
        re_index=np.array([0, 1]),
        uses_v=False,
        draw_truth=lambda rng: Truth(
            beta=rng.normal(0, 2, 2),
            psi=_sd_cov(rng.uniform(0.2, 1.0, 2), rng.uniform(-0.6, 0.6)),
            sigma=rng.uniform(0.2, 1.0),
        ),
        draw_design=lambda rng, m: (_times(rng, m, 3, 8, 0, 10), None),
        draw_start=lambda rng, beta: beta + rng.normal(0, 0.5, beta.size),
        m_range=(8, 40),
    ),
    "emax": Family(
        modelfun=emax,
        kwargs={"ParamTransform": [0, 0, 1], "REParamsSelect": [1, 3]},
        transform=np.array([0, 0, 1]),
        re_index=np.array([0, 2]),
        uses_v=False,
        draw_truth=lambda rng: Truth(
            beta=np.array(
                [rng.uniform(5, 20), rng.uniform(20, 80), np.log(rng.uniform(2, 20))]
            ),
            psi=_sd_cov([rng.uniform(1, 4), rng.uniform(0.1, 0.4)]),
            sigma=rng.uniform(1, 3),
        ),
        draw_design=lambda rng, m: (
            [
                np.sort(
                    np.exp(rng.uniform(np.log(0.1), np.log(100), rng.integers(4, 9)))
                )
                for _ in range(m)
            ],
            None,
        ),
        draw_start=lambda rng, beta: beta * np.array([*rng.uniform(0.85, 1.15, 2), 1])
        + np.array([0, 0, rng.normal(0, 0.2)]),
        m_range=(10, 40),
    ),
}


def _phi(eta, codes):
    phi = eta.copy()
    phi[codes == 1] = np.exp(phi[codes == 1])
    return phi


def simulate(
    family: str,
    rng: np.random.Generator,
    name: str | None = None,
    m: int | None = None,
    truth: Truth | None = None,
    start_at_truth: bool = False,
) -> SimulatedDataset:
    fam = FAMILIES[family]
    truth = truth or fam.draw_truth(rng)
    m = m or int(rng.integers(fam.m_range[0], fam.m_range[1] + 1))
    xs, v = fam.draw_design(rng, m)
    b = rng.multivariate_normal(np.zeros(len(fam.re_index)), truth.psi, m)
    X, y, group = [], [], []
    for i in range(m):
        eta = truth.beta.copy()
        eta[fam.re_index] += b[i]
        phi = _phi(eta, fam.transform)
        f = fam.modelfun(phi, xs[i], v[i]) if fam.uses_v else fam.modelfun(phi, xs[i])
        X.append(xs[i])
        y.append(f + rng.normal(0, truth.sigma, len(f)))
        group.append(np.full(len(f), i + 1))
    beta0 = truth.beta.copy() if start_at_truth else fam.draw_start(rng, truth.beta)
    return SimulatedDataset(
        name=name or family,
        family=family,
        X=np.concatenate(X),
        y=np.concatenate(y),
        group=np.concatenate(group),
        V=v if fam.uses_v else None,
        modelfun=fam.modelfun,
        beta0=beta0,
        truth=truth,
        kwargs=dict(fam.kwargs),
    )


def write_for_r(datasets: list[SimulatedDataset], directory: Path) -> Path:
    """Write CSVs and a manifest for validation/r/fit_manifest.R."""
    manifest = []
    for d in datasets:
        path = directory / f"{d.name}.csv"
        v_by_obs = d.V[d.group.astype(int) - 1] if d.V is not None else [""] * len(d.y)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "x", "y", "v"])
            w.writerows(zip(d.group.astype(int), d.X, d.y, v_by_obs, strict=True))
        manifest.append(
            {
                "name": d.name,
                "family": d.family,
                "file": str(path),
                "start": d.beta0.tolist(),
            }
        )
    out = directory / "manifest.json"
    out.write_text(json.dumps(manifest))
    return out
