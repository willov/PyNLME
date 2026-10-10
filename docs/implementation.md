# Implementation

For developers. The fitting code is Python/NumPy/SciPy in `src/pynlme/`; the inner
loops of the Lindstrom-Bates modes also exist as compiled Rust kernels in `rust/`
(module `pynlme._kernels`), which do the same arithmetic and are used when built
(`PYNLME_KERNELS=0` switches them off).

## Code structure

| Module | Role |
|---|---|
| `nlmefit.py` | Public functions: option parsing (MATLAB and Python names), input validation, dispatch, `stats` |
| `model.py` | `GroupedModel`: data split by group, designs `A_i`/`B_i`, parameter transforms, model calls (per group or batched), Jacobians (analytic or central differences) |
| `lindstrom_bates.py` | `"LME"`/`"RELME"` building blocks (LME solve, EM, nlm refinement, error-model weights, also used by FO/FOCE) and nlme's iteration for `compat="r"` |
| `matlab_lb.py` | `"LME"`/`"RELME"` for `compat="matlab"` (nlmefit's iteration) |
| `first_order.py` | `"FO"`/`"FOCE"` |
| `saem.py` | `nlmefitsa` |
| `residuals.py` | MATLAB's residual definitions |
| `ode.py`, `sund_adapter.py` | ODE model functions (SciPy; compiled SUND models) |
| `utils.py` | Input validation, stacked/grouped conversion |
| `api.py` | `fit`/`predict`: named parameters, dict or Polars data (provisional) |
| `r.py`, `r_saemix.py` | `pynlme.r`: R's `nlme()` and saemix interfaces |
| `rust/src/lib.rs` | Compiled kernels: LME solve and gradient, EM, nlm refinement, MATLAB-mode QR, objective and Nelder-Mead |

The model for group i: `phi_i = g^-1(eta_i)`, `eta_i = A_i beta + B_i b_i`,
`b_i ~ N(0, Psi)`, `y_i = f(phi_i, X_i[, V_i]) + g_i e_i`. Internally
`Psi = sigma^2 L L'` with a lower-triangular relative factor `L`, and `b_i = L u_i`
("spherical" random effects), which keeps everything defined when a variance is zero.

## Lindstrom-Bates (`"LME"`, `"RELME"`)

As MATLAB's `nlmefit` and R's `nlme`: alternate

1. **PNLS**: for fixed `L`, minimise `sum_i ||w_i (y_i - f_i)||^2 + ||u_i||^2` over
   `beta` and `u_i` (Levenberg-Marquardt; the block-arrow normal equations are solved
   by eliminating the per-group blocks). All groups live in zero-padded arrays and the
   block algebra is batched with `einsum`.
2. **LME step**: linearise `f` at the PNLS solution and maximise the profiled
   (restricted) likelihood of the resulting linear mixed model over `L`. The likelihood
   is computed with QR decompositions of `[Z_i L, X_i, w_i; I, 0, 0]` (cross-products
   cancel catastrophically for large `L`), `L` is log-Cholesky parameterised, and the
   gradient is analytic for ML.

Robustness measures, each found by the randomized comparison with R
(`tests/validation/test_random_vs_r.py`): an initial LME step at the start values (as
nlme), restarts for variance components collapsed to zero, damping when the iteration
cycles, the best state if it does not converge, and stopping when changes only jitter at
the model's numerical noise level (ODE models).

With `compat="matlab"` (the default), `"LME"`/`"RELME"` use `matlab_lb.py`, which
follows MATLAB R2023b's nlmefit: `RefineBeta0` by its Levenberg-Marquardt `LMfit`,
`Delta_0 = 0.375 sqrt(sum Z^2 / M)`, PNLS by Gauss-Newton with step halving for fixed
`Delta` (random effects whose precision exceeds 1e8 times the smallest held fixed), the
LME step by `fminsearch` (a reimplementation of MATLAB's Nelder-Mead) on `logm(Delta'Delta)`
or `chol`, at a tolerance that stays at 100 x TolX/TolFun (nlmefit tightens a misnamed
variable), the `tolFac` 2 -> 0 schedule for PNLS, error-model weights after each LME step,
and forward-difference derivatives with step `eps^(1/3) * eta` (statjacobian). With
MATLAB's default options its iterates agree with MATLAB's `OutputFcn` trace to ~1e-10.
The previous engine (`fit_lindstrom_bates`: Levenberg-Marquardt PNLS run to convergence,
L-BFGS-B LME step with analytic gradient, restarts of collapsed variances) is still used
for `compat="r"` with proportional or combined error models.

With `compat="r"` (constant or exponential error model) `fit_lindstrom_bates_nlme`
follows nlme 3.1's iteration instead (`src/nlme.c`, `nlme_iterate`/`nlme_increment`):
each PNLS iteration re-linearises, updates the relative covariance factor by 20 ML EM
iterations followed by R's `nlm` engine (`optif9`, reimplemented in `_optif9`; nlme
keeps its second-to-last iterate, because optif9 overwrites its input vector with every
accepted iterate but the last), and takes a step-halved Gauss-Newton step to the
GLS/BLUP solution; PNLS stops at nls's relative-offset criterion or after 7 iterations;
the LME step starts from the updated factor and the next PNLS from the previous PNLS
solution; convergence compares successive PNLS factors, or accepts converged fixed
effects when the LME step does not move (nlme: one nlminb iteration; pynlme also when
the reported likelihood has settled, since its L-BFGS-B LME step keeps moving along a
collapsed variance where nlminb stops). Traced against R's `verbose = TRUE` output, and
inner PNLS iterations against `pnlsMaxIter = k`, the iterates agree to the printed
digits. The PNLS step's ML update is also why nlme's REML is not a restricted-likelihood
fit.

Error models: weights `1/g(f)` fixed within an iteration and refreshed after each LME
step; `sum(log w)` added to the log-likelihood; combined `(a, b)` refitted by
Nelder-Mead; exponential by fitting `log y`.

## FO and FOCE

Maximise the Gaussian marginal likelihood of the model linearised in `b` at `b_hat`
(`0` for FO; conditional modes for FOCE, solved to near machine precision because the
objective is differentiated numerically) over `beta` and `L` with L-BFGS-B, `sigma^2`
profiled. Error models: the same outer reweighting loop as MATLAB.

## SAEM

As MATLAB's `nlmefitsa` and saemix (Kuhn & Lavielle):

- Latent individual parameters `phi_i[R]` for parameters with random effects; during
  the annealing phase also the parameters without random effects, with a pseudo-variance
  shrinking by 10^-3 over the phase (afterwards they are fixed and updated by a
  Gauss-Newton step plus stochastic approximation).
- Three Metropolis-Hastings kernels with adaptive scales; several chains per subject
  (default `max(1, round(100/m))`), handled as extra rows.
- Step size 1 in phases 1-2, `1/k` in phase 3; simulated annealing on variances in
  phase 1; in phase 2 eigenvalues of `Omega` may shrink by at most 3% per iteration
  (a singular `Omega` is absorbing; this happened in most runs with 20 subjects).
- Everything runs on flat arrays of all rows' observations, so one model call per
  proposal with `Vectorization="Full"`.

## Testing

- `tests/validation/`: fixed references (MATLAB, R nlme, saemix), randomized comparison
  with R on fresh simulated data each run, simulation studies (`--runslow`), ODE models.
- `tests/test_nlmefit_api.py`: options, conventions, invariances.
- `validation/`: datasets, reference scripts (R, MATLAB) and results, benchmark.

## Sources

pynlme implements published algorithms. Where a publication leaves details open (the
order of steps, tolerances, which iterate is kept), the replicate modes match the
reference programs; those details were established from their outputs and traces and by
reading their source code, and pynlme reproduces them. pynlme's code is its own
implementation of the same computations, with its own structure and names; it is not a
translation of MATLAB's or R's code, and contains none of it.

| Component | Published sources |
|---|---|
| Lindstrom-Bates (PNLS and LME steps) | Lindstrom & Bates (1990), *Biometrics* 46:673-687; Pinheiro & Bates (2000), *Mixed-Effects Models in S and S-PLUS*, ch. 2 and 7 |
| Profiled (restricted) likelihood via QR, precision factor Delta | Pinheiro & Bates (2000), sec. 2.2 (eqs. 2.17-2.23) |
| Starting value of Delta | Bates & Pinheiro (1998), *Computational methods for multilevel models* |
| EM for the variance components | Pinheiro & Bates (2000), sec. 2.3; Laird & Ware (1982) |
| Log-Cholesky and matrix-logarithm parametrisations | Pinheiro & Bates (1996), *Statistics and Computing* 6:289-296 |
| Quasi-Newton refinement (nlm/optif9 behaviour) | Dennis & Schnabel (1983), *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*, Appendix A (A5.6.3, A6.3.1, A9.4.2) |
| Nelder-Mead (MATLAB's fminsearch behaviour) | Nelder & Mead (1965), *Computer Journal* 7:308-313; Lagarias, Reeds, Wright & Wright (1998), *SIAM J. Optim.* 9:112-147 |
| Levenberg-Marquardt pre-fit | Levenberg (1944); Marquardt (1963) |
| FO and FOCE | Beal & Sheiner (1982); Lindstrom & Bates (1990) |
| SAEM, MCMC kernels, chains | Delyon, Lavielle & Moulines (1999); Kuhn & Lavielle (2005); Comets, Lavenu & Lavielle (2017), *J. Stat. Softw.* 80 |
| Importance-sampling log-likelihood | Kuhn & Lavielle (2005) |

