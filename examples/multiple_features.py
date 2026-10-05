"""
Fit a concentration and an effect measured in the same subjects (PK/PD), with
one ODE simulation per subject. Run: uv run python examples/multiple_features.py

As with MATLAB's nlmefit, all observations of a subject are stacked into one
response vector, and a column of X says which feature each row is. With SUND,
the model would be sund_model("mm_pkpd", feature=["conc", "effect"], ...)
(validation/models/mm_pkpd_sund.txt).
"""

import numpy as np

from pynlme import nlmefit, ode_model

names = ["ka", "V", "vmax", "km", "emax", "ec50"]


def rhs(t, y, phi, dose):  # states: gut amount, concentration
    ka, V, vmax, km = phi[:, 0], phi[:, 1], phi[:, 2], phi[:, 3]
    gut, c = y[:, 0], y[:, 1]
    return np.column_stack([-ka * gut, ka * gut / V - vmax * c / (km + c)])


def y0(phi, dose):  # the dose starts in the gut
    return np.column_stack([dose[:, 0], np.zeros(len(phi))])


def observed(y, phi, dose):  # one column per feature: 0 = conc, 1 = effect
    c = y[:, 1]
    return np.column_stack([c, phi[:, 4] * c / (phi[:, 5] + c)])


model = ode_model(rhs, y0, observed)

# Simulated study: 12 subjects, both features at 7 times, 10% proportional error.
rng = np.random.default_rng(0)
truth = np.array([1.0, 10.0, 2.0, 1.5, 100.0, 2.0])
times = np.array([0.5, 1, 2, 4, 6, 8, 12])
X, y, subject, dose = [], [], [], []
for i in range(12):
    phi = truth * np.exp(np.r_[rng.normal(0, 0.3), 0, 0, rng.normal(0, 0.2), 0, 0])
    d = rng.choice([20.0, 40.0])
    x = np.column_stack([np.tile(times, 2), np.repeat([0, 1], len(times))])  # time, feature
    f = model(phi, x, np.array([d]))
    X.append(x)
    y.append(f * (1 + 0.1 * rng.normal(size=f.size)))
    subject.append(np.full(len(f), i))
    dose.append(d)
X, y, subject, dose = np.vstack(X), np.concatenate(y), np.concatenate(subject), np.array(dose)[:, None]

beta, psi, stats, b = nlmefit(
    X, y, subject, dose, model, np.log(truth * 1.2),
    ParamTransform=[1] * 6,      # all parameters positive (log scale)
    REParamsSelect=[1, 4],       # ka and km vary between subjects
    ErrorModel="proportional",   # the features are on different scales
    Vectorization="Full",
)
for name, est, true in zip(names, np.exp(beta), truth, strict=True):
    print(f"{name:5s} {est:8.3f}  (true {true})")
print(f"proportional error SD {stats.errorparam[0]:.3f} (true 0.1)")
