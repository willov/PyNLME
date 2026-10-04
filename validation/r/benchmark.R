# Timings of R nlme and saemix on the validation cases (best of several runs).
#
# Run from the repository root:  Rscript validation/r/benchmark.R
# Writes validation/reference/r_timings.json, used by validation/benchmark.py.

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(nlme)
  library(saemix)
  library(jsonlite)
})

best_of <- function(n, expr) {
  expr <- substitute(expr)
  env <- parent.frame()
  min(replicate(n, system.time(eval(expr, env))[["elapsed"]]))
}

read <- function(file) read.csv(file.path("validation/datasets", file))
ctrl <- nlmeControl(maxIter = 200, pnlsTol = 1e-6, tolerance = 1e-7, msMaxIter = 500)
timings <- list(nlme = list(), saemix = list())

indo <- groupedData(conc ~ time | subject, data = read("indometh.csv"))
timings$nlme$indometh <- best_of(20, nlme(conc ~ p1 * exp(-exp(p2) * time) + p3 * exp(-exp(p4) * time),
  data = indo, fixed = p1 + p2 + p3 + p4 ~ 1, random = pdDiag(p1 + p2 + p3 + p4 ~ 1),
  start = c(p1 = 0.5, p2 = -1.3, p3 = 2.8, p4 = 0.8), control = ctrl))

theo <- groupedData(conc ~ time | subject, data = read("theoph.csv"))
timings$nlme$theoph <- best_of(20, nlme(conc ~ dose * exp(lKe + lKa - lCl) *
  (exp(-exp(lKe) * time) - exp(-exp(lKa) * time)) / (exp(lKa) - exp(lKe)),
  data = theo, fixed = lKe + lKa + lCl ~ 1, random = pdDiag(lKa + lCl ~ 1),
  start = c(lKe = -2.5, lKa = 0.5, lCl = -3.2), control = ctrl))

orange <- groupedData(circumference ~ age | tree, data = read("orange.csv"))
timings$nlme$orange <- best_of(20, nlme(circumference ~ Asym / (1 + exp(-(age - xmid) / scal)),
  data = orange, fixed = Asym + xmid + scal ~ 1, random = Asym ~ 1,
  start = c(Asym = 192, xmid = 727, scal = 348), control = ctrl))

lin <- groupedData(y ~ x | id, data = read("linear_sim.csv"))
timings$nlme$linear <- best_of(20, lme(y ~ x, data = lin, random = ~ x | id, method = "ML"))

saemix_time <- function(data, predictors, model, psi0, transform, covariance) {
  sdata <- saemixData(name.data = data, name.group = "id", name.predictors = predictors,
                      name.response = "y", verbose = FALSE)
  smodel <- saemixModel(model = model, psi0 = psi0, transform.par = transform,
                        covariance.model = covariance, verbose = FALSE)
  best_of(5, saemix(smodel, sdata, control = list(seed = 1, displayProgress = FALSE,
    save = FALSE, save.graphs = FALSE, print = FALSE, warnings = FALSE, ll.is = FALSE)))
}
d <- read("theoph.csv")
timings$saemix$theoph <- saemix_time(
  data.frame(id = d$subject, dose = d$dose, time = d$time, y = d$conc), c("dose", "time"),
  function(psi, id, x) {
    ke <- psi[id, 1]; ka <- psi[id, 2]; cl <- psi[id, 3]
    x[, 1] * ke * ka / (cl * (ka - ke)) * (exp(-ke * x[, 2]) - exp(-ka * x[, 2]))
  },
  matrix(exp(c(-2.5, 0.5, -3.2)), ncol = 3, dimnames = list(NULL, c("ke", "ka", "cl"))),
  c(1, 1, 1), diag(c(0, 1, 1)))
d <- read("orange.csv")
timings$saemix$orange <- saemix_time(
  data.frame(id = d$tree, age = d$age, y = d$circumference), "age",
  function(psi, id, x) psi[id, 1] / (1 + exp(-(x[, 1] - psi[id, 2]) / psi[id, 3])),
  matrix(c(192, 727, 348), ncol = 3, dimnames = list(NULL, c("A", "x", "s"))),
  c(0, 0, 0), diag(c(1, 0, 0)))
d <- read("linear_sim.csv")
timings$saemix$linear <- saemix_time(
  data.frame(id = d$id, x = d$x, y = d$y), "x",
  function(psi, id, x) psi[id, 1] + psi[id, 2] * x[, 1],
  matrix(c(1, 1), ncol = 2, dimnames = list(NULL, c("a", "b"))), c(0, 0), matrix(1, 2, 2))

timings$r_version <- R.version.string
write(toJSON(timings, auto_unbox = TRUE, pretty = TRUE, digits = 6), "validation/reference/r_timings.json")
print(timings)
