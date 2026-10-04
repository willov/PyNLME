# SUND wishlist from pynlme

Requests for the [SUND toolbox](https://gitlab.liu.se/ISBgroup/projects/sund) that
came out of integrating it into pynlme (nonlinear mixed-effects fitting, a Python
replacement for MATLAB's `nlmefit`/`nlmefitsa`). Each item has the evidence behind
it and a way to check it.

## How pynlme uses SUND

`pynlme.sund_model` ([src/pynlme/sund_adapter.py](../src/pynlme/sund_adapter.py))
wraps an installed SUND model as a model function `f(phi, t, v)`: `phi` sets named
parameters, `v` (per-subject covariates such as the dose) sets initial states or
parameters, and one feature is returned at the observation times.

A fit evaluates the model very many times:

- **nlmefit** (Lindstrom-Bates / FOCE) differentiates the model by central finite
  differences with respect to each parameter, for every subject, every iteration.
  This needs tight integrator tolerances (rel 1e-10, abs 1e-12); with SUND's
  default 1e-6 the derivatives are too noisy.
- **nlmefitsa** (SAEM) runs Metropolis-Hastings for every subject and MCMC chain,
  ~10 proposals per iteration, ~400 iterations: 10^5-10^6 simulations per fit.

Parameter sets within one model call are independent (different subjects, chains
or finite-difference perturbations), so they can be simulated in any order or in
parallel.

Benchmark case: the Michaelis-Menten PK model
([validation/models/mm_pk_sund.txt](../validation/models/mm_pk_sund.txt); 2 states,
4 parameters) with 16 subjects
([validation/datasets/mm_pk.csv](../validation/datasets/mm_pk.csv)).

## Measurements (SUND 2.5.0, Linux x86_64)

| | time |
|---|---|
| One `simulate` call, rel_tol 1e-6 (Adams) | 44-67 us |
| One `simulate` call, rel_tol 1e-10 | 188 us (Adams), 292 us (BDF) |
| SciPy LSODA, same model, rel_tol 1e-10 | 1600 us |
| One simulation with K models in containers, K = 16 / 96 / 384 | 0.9 / 13 / 205 ms |
| K separate `simulate` calls, K = 16 / 96 / 384 | 3.4 / 20 / 72 ms |
| `nlmefit` LME fit: one subject per call / containers of 16 | 9.7 s / 1.0 s |
| `nlmefitsa`, 150 iterations: one subject per call / containers of 16 | 43 s / 15 s |

## Wishlist (most impact first)

### 1. Forward sensitivities

Return d(feature)/d(parameter) for selected parameters, and ideally d(feature)/d(initial
state), at the output times, computed by the integrator (CVODES forward sensitivity
analysis) during `simulate`.

**Why:** pynlme computes Jacobians by central finite differences: 2p extra
simulations per subject per iteration, at tolerances tight enough to make the
differences accurate. Exact sensitivities remove those simulations and allow the
default tolerances, which are 3-4x cheaper per call (table above). Expected gain for
`nlmefit`: several-fold beyond what batching gives.

**Suggested interface:** e.g. `sim.simulate(..., sensitivities=["ka", "V", ...])`,
then `sim.feature_sensitivities` with shape (time, feature, parameter). For
containers, sensitivities per container.

**Done when:** sensitivities agree with central finite differences at rel_tol 1e-10 to
~1e-6, and cost noticeably less than 2p separate simulations.

### 2. Block-diagonal (or sparse) linear solver for container simulations

**Evidence:** batching parameter sets as model copies in containers works and is the
biggest win so far (10x for `nlmefit`), but scaling is superlinear: 384 copies in one
simulation take 205 ms versus 72 ms as separate calls. The copies are uncoupled, so
the system Jacobian is block diagonal, but the cost suggests a dense solve of the
whole stacked system (~K^3).

**Ask:** detect or let the user declare that containers are independent, and use a
block-diagonal or sparse (e.g. KLU) linear solver, or Adams with functional
iteration where suitable, so that cost grows linearly with the number of copies.

**Done when:** one simulation with K independent copies costs at most about K times
one simulation, up to K in the thousands.

### 3. A batched simulate call

**Ask:** simulate K parameter sets and K initial-state vectors of the same model in one
call: `parameter_values` (K x n_p), `state_values` (K x n_x), returning
(K x time x feature), without the user building K containers. Allow a different time
vector per set, or a shared union of times. Ideally run the sets in parallel
internally.

**Why:** containers do this today but need K model objects and a simulation object per
batch size (pynlme caches them); a direct API would be simpler and avoid the stacked
linear system of item 2.

### 4. Release the GIL in `simulate`

Lets callers parallelise independent simulations with threads. Even without item 3
this would scale fits across cores.

**Done when:** N threads each calling `simulate` on their own `Simulation` scale close to
N times.

### 5. Failures that do not throw

When the integrator fails for a parameter set (e.g. extreme values proposed by the
MCMC sampler), return NaN or a status flag for that set instead of raising. In a batch,
a failure in one set should not affect the others. pynlme treats such sets as rejected
proposals; exceptions are slow and abort a whole batch.

### 6. Lower per-call overhead

- Accept numpy arrays without copying; return `feature_values` as a numpy view.
- Optionally compute only selected features.
- Avoid redoing setup when only parameter or state values change between calls with
  the same time vector.

### 7. Absolute paths in `install_model`

`sund.install_model("/abs/path/model.txt")` raises
`NotImplementedError: Non-relative patterns are unsupported` because the argument is
passed to `Path.glob`. pynlme works around it with `os.path.relpath`.

### 8. No recompilation when unchanged

`install_model` recompiles on every call (seconds per test run). Skipping it when the
model file's hash matches the installed module would help test suites and notebooks.

### 9. Tolerances in the constructor

e.g. `sund.Simulation(models=..., rel_tol=1e-10, abs_tol=1e-12)`, plus a short guide to
choosing method and tolerances for parameter estimation.

## Reproducing the measurements

From the pynlme repository (with `pip install "pynlme[sund]"`):

```python
import os, time
import numpy as np
import sund

sund.install_model(os.path.relpath("validation/models/mm_pk_sund.txt"))
Model = sund.import_model("mm_pk")
t = np.array([0, .25, .5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 24.0])
for K in (16, 96, 384):
    sim = sund.Simulation(models=[Model(container=f"s{k}") for k in range(K)], time_unit="h")
    sim.set_options({"rel_tol": 1e-10, "abs_tol": 1e-12})
    P = np.tile([1.0, 10.0, 2.0, 1.5], (K, 1)).ravel()
    X0 = np.tile([50.0, 0.0], (K, 1)).ravel()
    t0 = time.perf_counter()
    sim.simulate(time_vector=t, parameter_values=P, state_values=X0)
    print(K, f"{(time.perf_counter() - t0) * 1e3:.1f} ms")
```

Full fits: `examples/sund_ode_model.py`; the `batch_size` argument of
`pynlme.sund_model` controls the batch size.

When an item lands, `pynlme.sund_model` can use it: sensitivities would replace the
finite differences in `GroupedModel.jacobian_padded` for SUND models.
