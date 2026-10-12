# Fit datasets listed in a manifest with R nlme (ML), for the randomized
# comparison in tests/validation/test_random_vs_r.py.
#
# Usage: Rscript validation/r/fit_manifest.R manifest.json results.json
# The model families mirror tests/validation/simulate.py.

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(nlme)
  library(jsonlite)
})

args <- commandArgs(TRUE)
manifest <- fromJSON(args[1], simplifyVector = FALSE)
ctrl <- nlmeControl(maxIter = 200, pnlsTol = 1e-6, tolerance = 1e-7, msMaxIter = 500)

fit_case <- function(case) {
  d <- groupedData(y ~ x | id, data = read.csv(case$file))
  s <- unlist(case$start)
  switch(case$family,
    oral = nlme(y ~ v * exp(lKe + lKa - lCl) * (exp(-exp(lKe) * x) - exp(-exp(lKa) * x)) /
                  (exp(lKa) - exp(lKe)),
                data = d, fixed = lKe + lKa + lCl ~ 1, random = pdDiag(lKa + lCl ~ 1),
                start = c(lKe = s[1], lKa = s[2], lCl = s[3]), method = "ML", control = ctrl),
    logistic = nlme(y ~ Asym / (1 + exp(-(x - xmid) / scal)), data = d,
                    fixed = Asym + xmid + scal ~ 1, random = pdDiag(Asym + xmid ~ 1),
                    start = c(Asym = s[1], xmid = s[2], scal = s[3]), method = "ML",
                    control = ctrl),
    linear = lme(y ~ x, data = d, random = ~ x | id, method = "ML",
                 control = lmeControl(msTol = 1e-12, tolerance = 1e-10)),
    emax = nlme(y ~ e0 + emax * x / (exp(lec50) + x), data = d,
                fixed = e0 + emax + lec50 ~ 1, random = pdDiag(e0 + lec50 ~ 1),
                start = c(e0 = s[1], emax = s[2], lec50 = s[3]), method = "ML",
                control = ctrl)
  )
}

results <- list()
for (case in manifest) {
  fit <- try(fit_case(case), silent = TRUE)
  results[[case$name]] <- if (inherits(fit, "try-error")) {
    list(error = trimws(as.character(fit)))
  } else {
    psi <- if (inherits(fit, "nlme")) {
      pdMatrix(fit$modelStruct$reStruct[[1]]) * fit$sigma^2
    } else {
      getVarCov(fit)
    }
    list(beta = unname(fixef(fit)), psi = matrix(as.numeric(psi), nrow(psi)),
         sigma = fit$sigma, logl = as.numeric(logLik(fit)))
  }
}
write(toJSON(results, digits = NA, auto_unbox = TRUE), args[2])
