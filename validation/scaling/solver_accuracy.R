# R's ODE solvers at the scaling benchmark's settings (relative 1e-10, absolute
# 1e-12) on the problems of validation/scaling/solver_accuracy.py: deSolve's lsoda
# (as in nlme's model functions) and rxode2 (as nlmixr2 solves the models), for
# the non-stiff mm model and the stiff tmdd model.
#
# Run from the repository root, after `solver_accuracy.py write`:
#   Rscript validation/scaling/solver_accuracy.R
# Writes validation/scaling/results/solver_values_r_<model>.csv.

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(deSolve)
  library(rxode2)
})

rtol <- 1e-10
atol <- 1e-12

models <- list(
  mm = list(
    times = c(0.5, 1, 2, 3, 4, 6, 8, 12),
    params = c("ka", "V", "vmax", "km"),
    rhs = function(t, s, p) {
      list(c(-p[1] * s[1], p[1] * s[1] / p[2] - p[3] * s[2] / (p[4] + s[2])))
    },
    y0 = function(dose) c(dose, 0),
    observe = function(sol) sol[, 3],
    rx = rxode2({
      d/dt(gut) <- -ka * gut
      d/dt(cent) <- ka * gut / V - vmax * cent / (km + cent)
      obs <- cent
    }),
    dose_cmt = "gut"
  ),
  tmdd = list(
    times = c(0.5, 1, 2, 4, 8, 12, 24, 48, 72, 96),
    params = c("kel", "kint", "R0"),
    rhs = function(t, s, p) {
      binding <- 100 * s[1] * (p[3] + s[2]) - 1 * s[3]
      list(c(-p[1] * s[1] - binding, -0.1 * s[2] - binding, binding - p[2] * s[3]))
    },
    y0 = function(dose) c(dose, 0, 0),
    observe = function(sol) log(sol[, 2]),
    rx = rxode2({
      receptor <- R0 + Rdev
      binding <- 100 * L * receptor - 1 * Cx
      d/dt(L) <- -kel * L - binding
      d/dt(Rdev) <- -0.1 * Rdev - binding
      d/dt(Cx) <- binding - kint * Cx
      obs <- log(L)
    }),
    dose_cmt = "L"
  )
)

dir.create("validation/scaling/results", showWarnings = FALSE)
for (name in names(models)) {
  m <- models[[name]]
  problems <- read.csv(sprintf("validation/scaling/data/solver_problems_%s.csv", name))
  lsoda_values <- t(sapply(seq_len(nrow(problems)), function(i) {
    p <- unlist(problems[i, m$params])
    sol <- lsoda(m$y0(problems$dose[i]), c(0, m$times), m$rhs, p,
                 rtol = rtol, atol = atol)
    m$observe(sol)[-1]
  }))
  rxode2_values <- t(sapply(seq_len(nrow(problems)), function(i) {
    p <- unlist(problems[i, m$params])
    ev <- et(amt = problems$dose[i], cmt = m$dose_cmt) |> et(m$times)
    sol <- rxSolve(m$rx, p, ev, rtol = rtol, atol = atol)
    sol$obs[match(m$times, sol$time)]
  }))
  out <- cbind(lsoda_values, rxode2_values)
  colnames(out) <- c(rep("deSolve lsoda (R nlme)", length(m$times)),
                     rep("rxode2 (nlmixr2)", length(m$times)))
  write.csv(out, sprintf("validation/scaling/results/solver_values_r_%s.csv", name),
            row.names = FALSE)
}
