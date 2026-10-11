# Fit the accuracy-benchmark replicates (validation/accuracy/simulate.py) with R
# nlme, saemix and nlmixr2 (FOCEi, SAEM), with their defaults.
#
# Run from the repository root:  Rscript validation/accuracy/run_r.R [scenario ...]
# Writes validation/accuracy/results/<scenario>/r.json: per method and replicate the
# fixed effects (log ke, log ka, log CL, and the effect of log(wt/70) on log CL in
# the hard scenario), their standard errors, the random-effect variances and the
# error parameter. saemix reports the population parameters on the natural scale;
# they are converted (log of the estimate, SE / estimate). nlme's proportional error
# is varPower(fixed = 1) (SD proportional to the fitted value). The covariate effect
# starts at 0.5 in every program: from 0, nlmixr2's FOCEi left it at 0.

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(nlme)
  library(saemix)
  library(nlmixr2)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
scenarios <- if (length(args)) args else c("basic", "hard")

oral_conc <- function(lKe, lKa, lCl, dose, time) {
  ke <- exp(lKe); ka <- exp(lKa); cl <- exp(lCl)
  dose * ke * ka / (cl * (ka - ke)) * (exp(-ke * time) - exp(-ka * time))
}

fit_nlme <- function(d, hard) {
  g <- groupedData(conc ~ time | id, data = d)
  fit <- if (hard) {
    g$lwt <- log(g$wt / 70)
    nlme(conc ~ oral_conc(lKe, lKa, lCl, dose, time), data = g,
         fixed = list(lKe ~ 1, lKa ~ 1, lCl ~ lwt),
         random = pdSymm(lKe + lKa + lCl ~ 1), weights = varPower(fixed = 1),
         start = c(-2.5, 0.5, -3.2, 0.5))
  } else {
    nlme(conc ~ oral_conc(lKe, lKa, lCl, dose, time), data = g,
         fixed = lKe + lKa + lCl ~ 1, random = pdDiag(lKa + lCl ~ 1),
         start = c(lKe = -2.5, lKa = 0.5, lCl = -3.2))
  }
  table <- summary(fit)$tTable
  vc <- suppressWarnings(as.numeric(VarCorr(fit)[, "Variance"]))
  r <- if (hard) 3 else 2
  list(beta = unname(fixef(fit)), se = unname(table[, "Std.Error"]),
       omega2 = vc[1:r], sigma = fit$sigma)
}

fit_saemix <- function(d, seed, hard) {
  frame <- data.frame(id = d$id, dose = d$dose, time = d$time, y = d$conc)
  if (hard) frame$lwt <- log(d$wt / 70)
  sdata <- saemixData(name.data = frame,
                      name.group = "id", name.predictors = c("dose", "time"),
                      name.response = "y",
                      name.covariates = if (hard) "lwt" else character(0),
                      verbose = FALSE)
  model <- function(psi, id, x) {
    ke <- psi[id, 1]; ka <- psi[id, 2]; cl <- psi[id, 3]
    x[, 1] * ke * ka / (cl * (ka - ke)) * (exp(-ke * x[, 2]) - exp(-ka * x[, 2]))
  }
  smodel <- if (hard) {
    saemixModel(model = model,
                psi0 = matrix(c(exp(c(-2.5, 0.5, -3.2)), 0, 0, 0.5), ncol = 3,
                              byrow = TRUE,
                              dimnames = list(NULL, c("ke", "ka", "cl"))),
                transform.par = c(1, 1, 1), covariate.model = matrix(c(0, 0, 1), 1),
                covariance.model = matrix(1, 3, 3), error.model = "proportional",
                verbose = FALSE)
  } else {
    saemixModel(model = model,
                psi0 = matrix(exp(c(-2.5, 0.5, -3.2)), ncol = 3,
                              dimnames = list(NULL, c("ke", "ka", "cl"))),
                transform.par = c(1, 1, 1), covariance.model = diag(c(0, 1, 1)),
                verbose = FALSE)
  }
  fit <- saemix(smodel, sdata, control = list(seed = seed, displayProgress = FALSE,
                                              save = FALSE, save.graphs = FALSE,
                                              print = FALSE, warnings = FALSE))
  r <- fit@results
  est <- r@fixed.effects  # ke, ka, cl (natural scale), then the covariate effect
  se <- r@se.fixed
  population <- 1:3
  beta <- c(log(est[population]), est[-population])
  se <- c(se[population] / est[population], se[-population])
  list(beta = beta, se = se,
       omega2 = if (hard) diag(r@omega) else diag(r@omega)[2:3],
       sigma = if (hard) r@respar[2] else r@respar[1])
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

hard_model <- function() {
  ini({
    tke <- -2.5; tka <- 0.5; tcl <- -3.2; bwt <- 0.5
    eta.ke + eta.ka + eta.cl ~ c(0.1, 0.01, 0.1, 0.01, 0.01, 0.1)
    prop.sd <- 0.2
  })
  model({
    ke <- exp(tke + eta.ke)
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + bwt * lwt + eta.cl)
    y <- dosemg * ke * ka / (cl * (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
    y ~ prop(prop.sd)
  })
}

fit_nlmixr2 <- function(d, est, seed, hard) {
  control <- if (est == "focei") foceiControl(print = 0) else
    saemControl(print = 0, seed = seed)
  frame <- data.frame(ID = d$id, TIME = d$time, DV = d$conc, dosemg = d$dose)
  if (hard) frame$lwt <- log(d$wt / 70)
  fit <- suppressMessages(suppressWarnings(nlmixr2(
    if (hard) hard_model else oral_model, frame, est = est, control = control)))
  par <- fit$parFixedDf
  names <- if (hard) c("tke", "tka", "tcl", "bwt") else c("tke", "tka", "tcl")
  list(beta = unname(par[names, "Estimate"]), se = unname(par[names, "SE"]),
       omega2 = unname(diag(fit$omega)),
       sigma = unname(fit$theta[if (hard) "prop.sd" else "add.sd"]))
}

for (scenario in scenarios) {
  data_dir <- file.path("validation/accuracy/data", scenario)
  truth <- fromJSON(file.path(data_dir, "truth.json"))
  hard <- scenario == "hard"
  methods <- list(
    "LME, nlme" = function(d, k) fit_nlme(d, hard),
    "SAEM, saemix" = function(d, k) fit_saemix(d, 1000 + k, hard),
    "FOCEi, nlmixr2" = function(d, k) fit_nlmixr2(d, "focei", 1000 + k, hard),
    "SAEM, nlmixr2" = function(d, k) fit_nlmixr2(d, "saem", 1000 + k, hard)
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
    cat(scenario, "replicate", k, "done\n")
  }
  out <- file.path("validation/accuracy/results", scenario)
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  write(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10, na = "null"),
        file.path(out, "r.json"))
}
