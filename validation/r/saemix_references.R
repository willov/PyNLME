# Reference fits with R saemix (SAEM, the algorithm behind MATLAB nlmefitsa).
#
# Run from the repository root:  Rscript validation/r/saemix_references.R
# SAEM is stochastic, so each case is fitted with several seeds; the JSON holds
# every run plus the mean and standard deviation across runs. Fixed effects are
# reported on the transformed scale (as MATLAB and pynlme do), random-effects
# variances on the transformed scale, and the log-likelihood by importance
# sampling (ll.is).

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(saemix)
  library(jsonlite)
})

data_dir <- "validation/datasets"
out_dir <- "validation/reference"
n_seeds <- 10

run_case <- function(name, data, predictors, model, psi0, transform, covariance, notes) {
  sdata <- saemixData(name.data = data, name.group = "id", name.predictors = predictors,
                      name.response = "y", verbose = FALSE)
  smodel <- saemixModel(model = model, psi0 = psi0, transform.par = transform,
                        covariance.model = covariance, error.model = "constant",
                        verbose = FALSE)
  runs <- lapply(seq_len(n_seeds), function(seed) {
    t0 <- Sys.time()
    fit <- saemix(smodel, sdata, control = list(
      seed = seed, displayProgress = FALSE, save = FALSE, save.graphs = FALSE,
      print = FALSE, warnings = FALSE, ll.is = TRUE, ll.gq = FALSE))
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")
    res <- fit@results
    fixed <- res@fixed.effects
    fixed[transform == 1] <- log(fixed[transform == 1])
    re <- which(diag(covariance) > 0)
    list(seed = seed, beta = fixed, psi = unname(res@omega[re, re, drop = FALSE]),
         sigma = res@respar[1], logl = res@ll.is, elapsed_s = elapsed)
  })
  beta <- do.call(rbind, lapply(runs, `[[`, "beta"))
  psi_diag <- do.call(rbind, lapply(runs, function(r) diag(r$psi)))
  res <- list(
    case = name, engine = "R saemix",
    engine_version = as.character(packageVersion("saemix")),
    r_version = R.version.string, n_seeds = n_seeds,
    settings = "saemix defaults: nbiter.saemix = c(300, 100), nb.chains = 1",
    beta_mean = colMeans(beta), beta_sd = apply(beta, 2, sd),
    psi_diag_mean = colMeans(psi_diag), psi_diag_sd = apply(psi_diag, 2, sd),
    sigma_mean = mean(sapply(runs, `[[`, "sigma")),
    logl_mean = mean(sapply(runs, `[[`, "logl")),
    logl_sd = sd(sapply(runs, `[[`, "logl")),
    elapsed_s_mean = mean(sapply(runs, `[[`, "elapsed_s")),
    runs = runs, notes = notes
  )
  write(toJSON(res, digits = NA, auto_unbox = TRUE, pretty = TRUE),
        file.path(out_dir, sprintf("r_saemix_%s.json", name)))
  cat(sprintf("%-10s beta=%s (sd %s) logl=%.3f  %.2fs/run\n", name,
              paste(signif(res$beta_mean, 5), collapse = ", "),
              paste(signif(res$beta_sd, 2), collapse = ", "), res$logl_mean,
              res$elapsed_s_mean))
}

read <- function(file) read.csv(file.path(data_dir, file))

# Indomethacin: MATLAB nlmefitsa doc example, ParamTransform = [0 1 0 1].
d <- read("indometh.csv")
run_case("indometh", data.frame(id = d$subject, time = d$time, y = d$conc), "time",
         function(psi, id, xidep) {
           t <- xidep[, 1]
           psi[id, 1] * exp(-psi[id, 2] * t) + psi[id, 3] * exp(-psi[id, 4] * t)
         },
         psi0 = matrix(c(0.5, exp(-1.3), 2.8, exp(0.8)), ncol = 4,
                       dimnames = list(NULL, c("p1", "p2", "p3", "p4"))),
         transform = c(0, 1, 0, 1), covariance = diag(4),
         notes = "All four parameters random, diagonal covariance.")

# Theophylline: dose as predictor, random effects on log ka and log cl only.
d <- read("theoph.csv")
run_case("theoph", data.frame(id = d$subject, dose = d$dose, time = d$time, y = d$conc),
         c("dose", "time"),
         function(psi, id, xidep) {
           dose <- xidep[, 1]; t <- xidep[, 2]
           ke <- psi[id, 1]; ka <- psi[id, 2]; cl <- psi[id, 3]
           dose * ke * ka / (cl * (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
         },
         psi0 = matrix(exp(c(-2.5, 0.5, -3.2)), ncol = 3,
                       dimnames = list(NULL, c("ke", "ka", "cl"))),
         transform = c(1, 1, 1), covariance = diag(c(0, 1, 1)),
         notes = "No random effect on log ke (REParamsSelect = [2 3]).")

# Orange trees: random effect on Asym only.
d <- read("orange.csv")
run_case("orange", data.frame(id = d$tree, age = d$age, y = d$circumference), "age",
         function(psi, id, xidep) {
           t <- xidep[, 1]
           psi[id, 1] / (1 + exp(-(t - psi[id, 2]) / psi[id, 3]))
         },
         psi0 = matrix(c(192, 727, 348), ncol = 3,
                       dimnames = list(NULL, c("Asym", "xmid", "scal"))),
         transform = c(0, 0, 0), covariance = diag(c(1, 0, 0)),
         notes = "Random effect on Asym only (REParamsSelect = 1).")

# Linear model with full covariance: exact ML is known (lme).
d <- read("linear_sim.csv")
run_case("linear", data.frame(id = d$id, x = d$x, y = d$y), "x",
         function(psi, id, xidep) psi[id, 1] + psi[id, 2] * xidep[, 1],
         psi0 = matrix(c(1, 1), ncol = 2, dimnames = list(NULL, c("a", "b"))),
         transform = c(0, 0), covariance = matrix(1, 2, 2),
         notes = "Full covariance (CovPattern = ones(2)).")
