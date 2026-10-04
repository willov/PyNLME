# Reference fits with R nlme (Lindstrom-Bates), the counterpart of MATLAB
# nlmefit with ApproximationType "LME" (method = "ML") and "RELME" (method = "REML").
#
# Run from the repository root:  Rscript validation/r/nlme_references.R
# Writes validation/reference/r_nlme_<case>.json. Parameters are reported in the
# same order and on the same (transformed) scale as the pynlme/MATLAB call in
# tests/validation/cases.py.

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(nlme)
  library(jsonlite)
})

data_dir <- "validation/datasets"
out_dir <- "validation/reference"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
# nlme's default pnlsTol (1e-3) stops the inner PNLS step early enough to shift
# the indomethacin estimates by ~2e-3; with 1e-6 they match MATLAB's published
# nlmefit values to all four printed decimals.
ctrl <- nlmeControl(maxIter = 200, pnlsTol = 1e-6, tolerance = 1e-7,
                    msMaxIter = 500)

# Random-effects covariance (on the response scale) in the order of `re_names`.
psi_matrix <- function(fit, re_names) {
  re <- fit$modelStruct$reStruct[[1]]
  m <- pdMatrix(re) * fit$sigma^2
  m[re_names, re_names, drop = FALSE]
}

# Random effects as an r-by-m matrix (MATLAB layout), groups in ascending order.
re_matrix <- function(fit, re_names) {
  r <- as.matrix(ranef(fit))
  r <- r[order(as.numeric(rownames(r))), re_names, drop = FALSE]
  t(unname(r))
}

write_case <- function(name, fit, re_names, method, approx, elapsed, notes) {
  tt <- summary(fit)$tTable
  groups <- sort(as.numeric(rownames(as.matrix(ranef(fit)))))
  res <- list(
    case = name,
    engine = "R nlme",
    engine_version = as.character(packageVersion("nlme")),
    r_version = R.version.string,
    method = method,
    approximation_type = approx,
    beta_names = rownames(tt),
    beta = unname(tt[, "Value"]),
    sebeta = unname(tt[, "Std.Error"]),
    re_names = re_names,
    psi = unname(psi_matrix(fit, re_names)),
    sigma = fit$sigma,
    logl = as.numeric(logLik(fit)),
    groups = groups,
    b = re_matrix(fit, re_names),
    elapsed_s = as.numeric(elapsed, units = "secs"),
    notes = notes
  )
  path <- file.path(out_dir, sprintf("r_nlme_%s.json", name))
  write(toJSON(res, digits = NA, auto_unbox = TRUE, pretty = TRUE), path)
  cat(sprintf("%-22s logl=%10.4f  beta=%s\n", name, res$logl,
              paste(signif(res$beta, 6), collapse = ", ")))
}

timed <- function(expr) {
  t0 <- Sys.time()
  fit <- expr
  list(fit = fit, elapsed = Sys.time() - t0)
}

# --- Indomethacin: MATLAB nlmefit doc example, ParamTransform = [0 1 0 1] -----
# Only ML: nlme's method = "REML" is not a restricted-likelihood fit for
# nonlinear models (it returns the ML estimates with sigma^2 scaled by
# N/(N - p)), so it is not a valid reference for MATLAB's "RELME".
indo <- groupedData(conc ~ time | subject,
                    data = read.csv(file.path(data_dir, "indometh.csv")))
r <- timed(nlme(conc ~ p1 * exp(-exp(p2) * time) + p3 * exp(-exp(p4) * time),
                data = indo, fixed = p1 + p2 + p3 + p4 ~ 1,
                random = pdDiag(p1 + p2 + p3 + p4 ~ 1),
                start = c(p1 = 0.5, p2 = -1.3, p3 = 2.8, p4 = 0.8),
                method = "ML", control = ctrl))
write_case("indometh_lme", r$fit, c("p1", "p2", "p3", "p4"), "ML", "LME",
           r$elapsed, "Diagonal random-effects covariance on all four parameters.")

# --- Theophylline: one-compartment oral absorption, Dose as group predictor ---
theo <- groupedData(conc ~ time | subject,
                    data = read.csv(file.path(data_dir, "theoph.csv")))
theo$wtc <- theo$wt - 70
theo_formula <- conc ~ dose * exp(lKe + lKa - lCl) *
  (exp(-exp(lKe) * time) - exp(-exp(lKa) * time)) / (exp(lKa) - exp(lKe))
r <- timed(nlme(theo_formula, data = theo,
                fixed = lKe + lKa + lCl ~ 1,
                random = pdDiag(lKa + lCl ~ 1),
                start = c(lKe = -2.5, lKa = 0.5, lCl = -3.2),
                method = "ML", control = ctrl))
write_case("theoph_lme", r$fit, c("lKa", "lCl"), "ML", "LME", r$elapsed,
           "Random effects on lKa and lCl only (REParamsSelect = [2 3]).")

# nlme's own REML: not a restricted-likelihood fit for nonlinear models (its
# PNLS step updates the covariance by ML EM iterations), but the reference for
# pynlme's compat = "r".
r <- timed(nlme(theo_formula, data = theo,
                fixed = lKe + lKa + lCl ~ 1,
                random = pdDiag(lKa + lCl ~ 1),
                start = c(lKe = -2.5, lKa = 0.5, lCl = -3.2),
                method = "REML", control = ctrl))
write_case("theoph_nlme_reml", r$fit, c("lKa", "lCl"), "REML", "RELME", r$elapsed,
           "nlme(method = 'REML'); reference for compat = 'r' only.")

# Weight (centred at 70 kg) as a covariate on log clearance (FEGroupDesign).
r <- timed(nlme(theo_formula, data = theo,
                fixed = list(lKe ~ 1, lKa ~ 1, lCl ~ wtc),
                random = pdDiag(lKa + lCl ~ 1),
                start = c(-2.5, 0.5, -3.2, 0),
                method = "ML", control = ctrl))
write_case("theoph_wt_lme", r$fit, c("lKa", "lCl.(Intercept)"), "ML", "LME",
           r$elapsed,
           "lCl = beta3 + beta4 * (wt - 70); random effects on lKa and lCl.")

# --- Orange trees: logistic growth, random asymptote only ---------------------
orange <- groupedData(circumference ~ age | tree,
                      data = read.csv(file.path(data_dir, "orange.csv")))
r <- timed(nlme(circumference ~ Asym / (1 + exp(-(age - xmid) / scal)),
                data = orange, fixed = Asym + xmid + scal ~ 1,
                random = Asym ~ 1, start = c(Asym = 192, xmid = 727, scal = 348),
                method = "ML", control = ctrl))
write_case("orange_lme", r$fit, "Asym", "ML", "LME", r$elapsed,
           "Random effect on Asym only (REParamsSelect = 1).")

# nlme's REML on the linear model (differs from lme's true REML below).
lin <- groupedData(y ~ x | id, data = read.csv(file.path(data_dir, "linear_sim.csv")))
r <- timed(nlme(y ~ a + b * x, data = lin, fixed = a + b ~ 1, random = a + b ~ 1,
                start = c(a = 1, b = 1), method = "REML", control = ctrl))
write_case("linear_nlme_reml", r$fit, c("a", "b"), "REML", "RELME", r$elapsed,
           "nlme(method = 'REML') on the linear model; reference for compat = 'r' only.")

# --- Linear model: nlmefit on a linear model must equal the exact lme answer --
# This is also the reference for RELME, using lme's (true) REML.
lin <- groupedData(y ~ x | id, data = read.csv(file.path(data_dir, "linear_sim.csv")))
re_names <- c("(Intercept)", "x")
for (m in c("ML", "REML")) {
  r <- timed(lme(y ~ x, data = lin, random = ~ x | id, method = m,
                 control = lmeControl(msTol = 1e-12, tolerance = 1e-10)))
  fit <- r$fit
  name <- ifelse(m == "ML", "linear_lme", "linear_relme")
  res <- list(
    case = name, engine = "R nlme::lme",
    engine_version = as.character(packageVersion("nlme")),
    r_version = R.version.string, method = m,
    approximation_type = ifelse(m == "ML", "LME", "RELME"),
    beta_names = names(fixef(fit)), beta = unname(fixef(fit)),
    sebeta = unname(summary(fit)$tTable[, "Std.Error"]),
    re_names = re_names, psi = unname(getVarCov(fit)[re_names, re_names]),
    sigma = fit$sigma, logl = as.numeric(logLik(fit)),
    groups = sort(as.numeric(rownames(ranef(fit)))),
    b = re_matrix(fit, re_names),
    elapsed_s = as.numeric(r$elapsed, units = "secs"),
    notes = "Exact linear mixed model; full 2x2 covariance (CovPattern = ones(2))."
  )
  write(toJSON(res, digits = NA, auto_unbox = TRUE, pretty = TRUE),
        file.path(out_dir, sprintf("r_nlme_%s.json", name)))
  cat(sprintf("%-22s logl=%10.4f  beta=%s\n", name, res$logl,
              paste(signif(res$beta, 6), collapse = ", ")))
}
