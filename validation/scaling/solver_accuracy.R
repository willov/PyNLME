# R's ODE solvers at the scaling benchmark's settings (relative 1e-10, absolute
# 1e-12) on the problems of validation/scaling/solver_accuracy.py: deSolve's lsoda
# (as in nlme's model function) and rxode2 (as nlmixr2 solves the model).
#
# Run from the repository root, after `solver_accuracy.py write`:
#   Rscript validation/scaling/solver_accuracy.R
# Writes validation/scaling/results/solver_values_r.csv.

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(deSolve)
  library(rxode2)
})

problems <- read.csv("validation/scaling/data/solver_problems.csv")
times <- c(0.5, 1, 2, 3, 4, 6, 8, 12)
rtol <- 1e-10
atol <- 1e-12

rhs <- function(t, s, p) {
  list(c(-p[1] * s[1], p[1] * s[1] / p[2] - p[3] * s[2] / (p[4] + s[2])))
}
lsoda_values <- t(sapply(seq_len(nrow(problems)), function(i) {
  p <- unlist(problems[i, c("ka", "V", "vmax", "km")])
  sol <- lsoda(c(problems$dose[i], 0), c(0, times), rhs, p, rtol = rtol, atol = atol)
  sol[-1, 3]
}))

model <- rxode2({
  d/dt(gut) <- -ka * gut
  d/dt(cent) <- ka * gut / V - vmax * cent / (km + cent)
})
rxode2_values <- t(sapply(seq_len(nrow(problems)), function(i) {
  p <- unlist(problems[i, c("ka", "V", "vmax", "km")])
  ev <- et(amt = problems$dose[i], cmt = "gut") |> et(times)
  sol <- rxSolve(model, p, ev, rtol = rtol, atol = atol)
  sol$cent[match(times, sol$time)]
}))

out <- cbind(lsoda_values, rxode2_values)
colnames(out) <- c(rep("deSolve lsoda (R nlme)", length(times)),
                   rep("rxode2 (nlmixr2)", length(times)))
dir.create("validation/scaling/results", showWarnings = FALSE)
write.csv(out, "validation/scaling/results/solver_values_r.csv", row.names = FALSE)
