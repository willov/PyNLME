# nlmixr2 (FOCEi and SAEM) on the benchmark cases, for comparison with pynlme's
# FOCE and SAEM: estimates, log-likelihood and timings.
#
# Run from the repository root:  Rscript validation/r/nlmixr2_benchmark.R
# Writes validation/reference/nlmixr2_results.json, used by
# validation/nlmixr2_compare.py.
#
# The models and start values of the fixed effects are those of
# tests/validation/cases.py (random effects on the same parameters, diagonal
# covariance except for the linear case, additive error). With an additive error
# model FOCEi and FOCE are the same method. nlmixr2 needs start values for the
# random-effects variances and the error; these are rough guesses (pynlme and MATLAB
# start them from a fit without random effects). Times are the best of 3 runs after
# a warm-up run (which also compiles the model). Covariates need lowercase names
# (nlmixr2 did not find a column named DOSE).

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(nlmixr2)
  library(jsonlite)
})

read <- function(file) read.csv(file.path("validation/datasets", file))

indometh <- function() {
  ini({
    tp1 <- 0.5; tp2 <- -1; tp3 <- 2.5; tp4 <- 0.5
    eta.p1 ~ 0.1; eta.p2 ~ 0.1; eta.p3 ~ 0.1; eta.p4 ~ 0.1
    add.sd <- 0.1
  })
  model({
    a1 <- tp1 + eta.p1
    k1 <- exp(tp2 + eta.p2)
    a2 <- tp3 + eta.p3
    k2 <- exp(tp4 + eta.p4)
    y <- a1 * exp(-k1 * t) + a2 * exp(-k2 * t)
    y ~ add(add.sd)
  })
}

theoph <- function() {
  ini({
    tke <- -2.5; tka <- 0.5; tcl <- -3.2
    eta.ka ~ 0.1; eta.cl ~ 0.1
    add.sd <- 0.7
  })
  model({
    ke <- exp(tke)
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    y <- dosemg * ke * ka / (cl * (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
    y ~ add(add.sd)
  })
}

orange <- function() {
  ini({
    tasym <- 192; txmid <- 727; tscal <- 348
    eta.asym ~ 1000
    add.sd <- 8
  })
  model({
    asym <- tasym + eta.asym
    y <- asym / (1 + exp(-(t - txmid) / tscal))
    y ~ add(add.sd)
  })
}

linear <- function() {
  ini({
    ta <- 1; tb <- 1
    eta.a + eta.b ~ c(0.1, 0.01, 0.1)  # a zero start covariance stays zero
    add.sd <- 0.3
  })
  model({
    a <- ta + eta.a
    b <- tb + eta.b
    y <- a + b * t
    y ~ add(add.sd)
  })
}

d <- read("indometh.csv")
data_indometh <- data.frame(ID = d$subject, TIME = d$time, DV = d$conc)
d <- read("theoph.csv")
data_theoph <- data.frame(ID = d$subject, TIME = d$time, DV = d$conc, dosemg = d$dose)
d <- read("orange.csv")
data_orange <- data.frame(ID = d$tree, TIME = d$age, DV = d$circumference)
d <- read("linear_sim.csv")
data_linear <- data.frame(ID = d$id, TIME = d$x, DV = d$y)

cases <- list(
  indometh = list(model = indometh, data = data_indometh),
  theoph = list(model = theoph, data = data_theoph),
  orange = list(model = orange, data = data_orange),
  linear = list(model = linear, data = data_linear)
)

controls <- list(
  focei = foceiControl(print = 0),
  saem = saemControl(print = 0, seed = 1)
)

fit_case <- function(case, est) {
  suppressMessages(suppressWarnings(
    nlmixr2(case$model, case$data, est = est, control = controls[[est]])
  ))
}

results <- list()
for (name in names(cases)) {
  for (est in names(controls)) {
    fit <- tryCatch(fit_case(cases[[name]], est), error = function(e) e)
    if (inherits(fit, "error")) {
      results[[name]][[est]] <- list(error = conditionMessage(fit))
      cat(name, est, "error:", conditionMessage(fit), "\n")
      next
    }
    times <- replicate(3, system.time(fit_case(cases[[name]], est))[["elapsed"]])
    omega <- fit$omega
    results[[name]][[est]] <- list(
      elapsed_s = min(times),
      theta = as.list(fit$theta),
      omega = unname(omega),
      omega_names = rownames(omega),
      objf = fit$objf,
      loglik = as.numeric(logLik(fit)),
      message = paste(fit$message, collapse = " ")
    )
    cat(sprintf("%-9s %-6s %8.3f s  logLik %.4f\n", name, est, min(times),
                as.numeric(logLik(fit))))
  }
}

results$versions <- list(
  r = R.version.string,
  nlmixr2 = as.character(packageVersion("nlmixr2")),
  nlmixr2est = as.character(packageVersion("nlmixr2est")),
  rxode2 = as.character(packageVersion("rxode2"))
)
write(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10),
      "validation/reference/nlmixr2_results.json")
