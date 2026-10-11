# Fit the accuracy-benchmark replicates (validation/accuracy/simulate.py) with R
# nlme, saemix and nlmixr2 (FOCEi, SAEM), with their defaults.
#
# Run from the repository root:  Rscript validation/accuracy/run_r.R
# Writes validation/accuracy/results/r.json: per method and replicate the fixed
# effects (log ke, log ka, log CL), their standard errors, the random-effect
# variances (log ka, log CL) and the error SD. saemix reports its fixed effects on
# the natural scale; they are converted (log of the estimate, SE / estimate).

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(nlme)
  library(saemix)
  library(nlmixr2)
  library(jsonlite)
})

data_dir <- "validation/accuracy/data"
truth <- fromJSON(file.path(data_dir, "truth.json"))

oral_conc <- function(lKe, lKa, lCl, dose, time) {
  ke <- exp(lKe); ka <- exp(lKa); cl <- exp(lCl)
  dose * ke * ka / (cl * (ka - ke)) * (exp(-ke * time) - exp(-ka * time))
}

fit_nlme <- function(d) {
  g <- groupedData(conc ~ time | id, data = d)
  fit <- nlme(conc ~ oral_conc(lKe, lKa, lCl, dose, time), data = g,
              fixed = lKe + lKa + lCl ~ 1, random = pdDiag(lKa + lCl ~ 1),
              start = c(lKe = -2.5, lKa = 0.5, lCl = -3.2))
  table <- summary(fit)$tTable
  vc <- as.numeric(VarCorr(fit)[, "Variance"])
  list(beta = unname(fixef(fit)), se = unname(table[, "Std.Error"]),
       omega2 = vc[1:2], sigma = fit$sigma)
}

fit_saemix <- function(d, seed) {
  sdata <- saemixData(name.data = data.frame(id = d$id, dose = d$dose, time = d$time,
                                             y = d$conc),
                      name.group = "id", name.predictors = c("dose", "time"),
                      name.response = "y", verbose = FALSE)
  model <- function(psi, id, x) {
    ke <- psi[id, 1]; ka <- psi[id, 2]; cl <- psi[id, 3]
    x[, 1] * ke * ka / (cl * (ka - ke)) * (exp(-ke * x[, 2]) - exp(-ka * x[, 2]))
  }
  smodel <- saemixModel(model = model,
                        psi0 = matrix(exp(c(-2.5, 0.5, -3.2)), ncol = 3,
                                      dimnames = list(NULL, c("ke", "ka", "cl"))),
                        transform.par = c(1, 1, 1), covariance.model = diag(c(0, 1, 1)),
                        verbose = FALSE)
  fit <- saemix(smodel, sdata, control = list(seed = seed, displayProgress = FALSE,
                                              save = FALSE, save.graphs = FALSE,
                                              print = FALSE, warnings = FALSE))
  r <- fit@results
  est <- r@fixed.effects
  list(beta = log(est), se = r@se.fixed / est, omega2 = diag(r@omega)[2:3],
       sigma = r@respar[1])
}

oral_model <- function() {
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

fit_nlmixr2 <- function(d, est, seed) {
  control <- if (est == "focei") foceiControl(print = 0) else
    saemControl(print = 0, seed = seed)
  fit <- suppressMessages(suppressWarnings(nlmixr2(
    oral_model, data.frame(ID = d$id, TIME = d$time, DV = d$conc, dosemg = d$dose),
    est = est, control = control)))
  par <- fit$parFixedDf
  list(beta = unname(par[c("tke", "tka", "tcl"), "Estimate"]),
       se = unname(par[c("tke", "tka", "tcl"), "SE"]),
       omega2 = unname(diag(fit$omega)), sigma = unname(fit$theta["add.sd"]))
}

methods <- list(
  "LME, nlme" = function(d, k) fit_nlme(d),
  "SAEM, saemix" = function(d, k) fit_saemix(d, 1000 + k),
  "FOCEi, nlmixr2" = function(d, k) fit_nlmixr2(d, "focei", 1000 + k),
  "SAEM, nlmixr2" = function(d, k) fit_nlmixr2(d, "saem", 1000 + k)
)

results <- lapply(methods, function(m) list())
for (k in seq_len(truth$replicates) - 1) {
  d <- read.csv(file.path(data_dir, sprintf("rep_%d.csv", k)))
  for (method in names(methods)) {
    results[[method]][[k + 1]] <- tryCatch(
      methods[[method]](d, k),
      error = function(e) list(error = conditionMessage(e))
    )
  }
  cat("replicate", k, "done\n")
}
dir.create("validation/accuracy/results", showWarnings = FALSE)
write(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10, na = "null"),
      "validation/accuracy/results/r.json")
