# Time R nlme, saemix and nlmixr2 on the scaling datasets
# (validation/scaling/make_data.py).
#
# Run from the repository root:  Rscript validation/scaling/run_r.R
# Writes validation/scaling/results/r.json.
#
# Default options, as validation/scaling/run_pynlme.py: nlmeControl(), saemix's
# defaults, nlmixr2's defaults. The ODE model is solved with the common tolerance
# (relative 1e-8, absolute 1e-10): by deSolve's lsoda for nlme (per subject) and by
# rxode2 for nlmixr2; the multi-dose model (mmd) with deSolve events and nlmixr2's
# event table. saemix runs the closed-form model only. Sizes run in
# increasing order; once a run takes longer than 60 s the larger sizes are skipped.
# Runs shorter than 5 s are repeated and the best of 3 is kept; the first run of
# each method is an untimed warm-up (nlmixr2 compiles the model then).

if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(nlme)
  library(saemix)
  library(nlmixr2)
  library(deSolve)
  library(jsonlite)
})

sizes <- as.numeric(strsplit(Sys.getenv("SCALING_SIZES", "10,100,1000"), ",")[[1]])
limit <- 60
rtol <- 1e-8
atol <- 1e-10

read_case <- function(name, n) {
  read.csv(file.path("validation/scaling/data", sprintf("%s_%d.csv", name, n)))
}

# --- nlme ---------------------------------------------------------------------
oral_conc <- function(lKe, lKa, lCl, dose, time) {
  ke <- exp(lKe); ka <- exp(lKa); cl <- exp(lCl)
  dose * ke * ka / (cl * (ka - ke)) * (exp(-ke * time) - exp(-ka * time))
}

# Michaelis-Menten model, solved per subject (parameters constant within id).
mm_conc <- function(lka, lV, lvmax, lkm, dose, time, id) {
  out <- numeric(length(time))
  rhs <- function(t, s, p) {
    list(c(-p[1] * s[1], p[1] * s[1] / p[2] - p[3] * s[2] / (p[4] + s[2])))
  }
  for (subject in unique(id)) {
    rows <- which(id == subject)
    k <- rows[1]
    p <- exp(c(lka[k], lV[k], lvmax[k], lkm[k]))
    times <- sort(unique(time[rows]))
    sol <- lsoda(c(dose[k], 0), c(0, times), rhs, p, rtol = rtol, atol = atol)
    out[rows] <- sol[match(time[rows], sol[, 1]), 3]
  }
  out
}

# The mm model with the dose added to the gut at 0, 12 and 24 h (deSolve events).
mmd_conc <- function(lka, lV, lvmax, lkm, dose, time, id) {
  out <- numeric(length(time))
  rhs <- function(t, s, p) {
    list(c(-p[1] * s[1], p[1] * s[1] / p[2] - p[3] * s[2] / (p[4] + s[2])))
  }
  for (subject in unique(id)) {
    rows <- which(id == subject)
    k <- rows[1]
    p <- exp(c(lka[k], lV[k], lvmax[k], lkm[k]))
    times <- sort(unique(c(0, time[rows])))
    events <- data.frame(var = "gut", time = c(12, 24), value = dose[k],
                         method = "add")
    sol <- lsoda(c(gut = dose[k], cent = 0), times, rhs, p, rtol = rtol, atol = atol,
                 events = list(data = events))
    out[rows] <- sol[match(time[rows], sol[, 1]), 3]
  }
  out
}

# The stiff TMDD model (validation/models/tmdd_sund.txt): log free drug, per subject
# with lsoda (which switches to its stiff method).
tmdd_fixed <- c(kon = 100, koff = 1, kdeg = 0.1)
tmdd_rhs <- function(t, s, p) {
  binding <- tmdd_fixed[["kon"]] * s[1] * (p[3] + s[2]) - tmdd_fixed[["koff"]] * s[3]
  list(c(-p[1] * s[1] - binding, -tmdd_fixed[["kdeg"]] * s[2] - binding,
         binding - p[2] * s[3]))
}
tmdd_conc <- function(lkel, lkint, lR0, dose, time, id) {
  out <- numeric(length(time))
  for (subject in unique(id)) {
    rows <- which(id == subject)
    k <- rows[1]
    p <- exp(c(lkel[k], lkint[k], lR0[k]))
    times <- sort(unique(c(0, time[rows])))
    sol <- lsoda(c(dose[k], 0, 0), times, tmdd_rhs, p, rtol = rtol, atol = atol)
    out[rows] <- log(sol[match(time[rows], sol[, 1]), 2])
  }
  out
}

nlme_fit <- function(name, d) {
  if (name == "oral") {
    g <- groupedData(conc ~ time | id, data = d)
    nlme(conc ~ oral_conc(lKe, lKa, lCl, dose, time), data = g,
         fixed = lKe + lKa + lCl ~ 1, random = pdDiag(lKa + lCl ~ 1),
         start = c(lKe = -2.5, lKa = 0.5, lCl = -3.2))
  } else if (name == "tmdd") {
    g <- groupedData(conc ~ time | id, data = d)
    nlme(conc ~ tmdd_conc(lkel, lkint, lR0, dose, time, id), data = g,
         fixed = lkel + lkint + lR0 ~ 1, random = pdDiag(lkel + lkint ~ 1),
         start = log(c(lkel = 0.15, lkint = 0.07, lR0 = 8)))
  } else if (name == "mmd") {
    g <- groupedData(conc ~ time | id, data = d)
    nlme(conc ~ mmd_conc(lka, lV, lvmax, lkm, dose, time, id), data = g,
         fixed = lka + lV + lvmax + lkm ~ 1, random = pdDiag(lV + lvmax ~ 1),
         start = log(c(lka = 0.8, lV = 12, lvmax = 2.5, lkm = 1)))
  } else {
    g <- groupedData(conc ~ time | id, data = d)
    nlme(conc ~ mm_conc(lka, lV, lvmax, lkm, dose, time, id), data = g,
         fixed = lka + lV + lvmax + lkm ~ 1, random = pdDiag(lV + lvmax ~ 1),
         start = log(c(lka = 0.8, lV = 12, lvmax = 2.5, lkm = 1)))
  }
}

# --- saemix (closed form) -------------------------------------------------------
saemix_fit <- function(d) {
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
  saemix(smodel, sdata, control = list(seed = 1, displayProgress = FALSE, save = FALSE,
                                       save.graphs = FALSE, print = FALSE,
                                       warnings = FALSE))
}

# --- nlmixr2 -------------------------------------------------------------------
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

mm_model <- function() {
  ini({
    tka <- log(0.8); tv <- log(12); tvmax <- log(2.5); tkm <- log(1)
    eta.v ~ 0.1; eta.vmax ~ 0.1
    add.sd <- 0.3
  })
  model({
    ka <- exp(tka)
    v <- exp(tv + eta.v)
    vmax <- exp(tvmax + eta.vmax)
    km <- exp(tkm)
    d/dt(gut) <- -ka * gut
    d/dt(cent) <- ka * gut / v - vmax * cent / (km + cent)
    cent ~ add(add.sd)
  })
}

tmdd_model <- function() {
  ini({
    tkel <- log(0.15); tkint <- log(0.07); tR0 <- log(8)
    eta.kel ~ 0.1; eta.kint ~ 0.1
    add.sd <- 0.2
  })
  model({
    kel <- exp(tkel + eta.kel)
    kint <- exp(tkint + eta.kint)
    R0 <- exp(tR0)
    kon <- 100; koff <- 1; kdeg <- 0.1
    receptor <- R0 + Rdev
    binding <- kon * L * receptor - koff * Cx
    d/dt(L) <- -kel * L - binding
    d/dt(Rdev) <- -kdeg * Rdev - binding
    d/dt(Cx) <- binding - kint * Cx
    lL <- log(L)
    lL ~ add(add.sd)
  })
}

nlmixr2_data <- function(name, d) {
  obs <- data.frame(ID = d$id, TIME = d$time, DV = d$conc, dosemg = d$dose,
                    AMT = 0, EVID = 0, CMT = 2)
  if (name == "oral") {
    return(obs[, c("ID", "TIME", "DV", "dosemg")])
  }
  first <- !duplicated(d$id)
  dose_times <- if (name == "mmd") c(0, 12, 24) else 0
  doses <- do.call(rbind, lapply(dose_times, function(t0) {
    data.frame(ID = d$id[first], TIME = t0, DV = NA, dosemg = d$dose[first],
               AMT = d$dose[first], EVID = 1, CMT = 1)
  }))
  out <- rbind(doses, obs)
  out[order(out$ID, out$TIME, -out$EVID), ]
}

nlmixr2_fit <- function(name, d, est) {
  control <- if (est == "focei") {
    foceiControl(print = 0, rxControl = rxControl(rtol = rtol, atol = atol))
  } else {
    saemControl(print = 0, seed = 1, rxControl = rxControl(rtol = rtol, atol = atol))
  }
  model <- switch(name, oral = oral_model, tmdd = tmdd_model, mm_model)
  suppressMessages(suppressWarnings(
    nlmixr2(model, nlmixr2_data(name, d), est = est, control = control)
  ))
}

# The log-likelihood each program reports (its own approximation), as a check
# that the fits are good ones.
loglik <- function(fit) {
  value <- tryCatch(
    if (inherits(fit, "SaemixObject")) fit@results@ll.is else as.numeric(logLik(fit)),
    error = function(e) NA
  )
  if (is.null(value) || length(value) == 0) NA else value
}

# --- timing --------------------------------------------------------------------
methods <- list(
  oral = list(
    "LME, nlme" = function(d) nlme_fit("oral", d),
    "SAEM, saemix" = function(d) saemix_fit(d),
    "FOCEi, nlmixr2" = function(d) nlmixr2_fit("oral", d, "focei"),
    "SAEM, nlmixr2" = function(d) nlmixr2_fit("oral", d, "saem")
  ),
  mm = list(
    "LME, nlme" = function(d) nlme_fit("mm", d),
    "FOCEi, nlmixr2" = function(d) nlmixr2_fit("mm", d, "focei"),
    "SAEM, nlmixr2" = function(d) nlmixr2_fit("mm", d, "saem")
  ),
  tmdd = list(
    "LME, nlme" = function(d) nlme_fit("tmdd", d),
    "FOCEi, nlmixr2" = function(d) nlmixr2_fit("tmdd", d, "focei"),
    "SAEM, nlmixr2" = function(d) nlmixr2_fit("tmdd", d, "saem")
  ),
  mmd = list(
    "LME, nlme" = function(d) nlme_fit("mmd", d),
    "FOCEi, nlmixr2" = function(d) nlmixr2_fit("mmd", d, "focei"),
    "SAEM, nlmixr2" = function(d) nlmixr2_fit("mmd", d, "saem")
  )
)

# The fallback, as in run_pynlme.py: a fit that fails or misses the optimum at the
# common tolerance runs at a tighter one, marked in the report. Here: nlme on tmdd,
# whose PNLS step halving fails near the optimum at relative 1e-6 to 1e-10.
tight <- list(tmdd = list("LME, nlme" = c(1e-12, 1e-14)))
with_tol <- function(tol, f) {
  force(f)
  function(d) {
    old <- c(rtol, atol)
    on.exit({ rtol <<- old[1]; atol <<- old[2] })
    rtol <<- tol[1]; atol <<- tol[2]
    f(d)
  }
}

# SCALING_MODELS=mmd runs only these models; other results already in r.json stay.
models <- strsplit(Sys.getenv("SCALING_MODELS", paste(names(methods), collapse = ",")),
                   ",")[[1]]
path <- "validation/scaling/results/r.json"
results <- if (file.exists(path)) fromJSON(path, simplifyVector = FALSE) else list()
for (name in models) {
  results[[name]] <- list()
  too_slow <- character(0)
  for (n in sizes) {
    d <- read_case(name, n)
    for (method in names(methods[[name]])) {
      key <- as.character(n)
      if (method %in% too_slow) {
        results[[name]][[method]][[key]] <- list(skipped = TRUE)
        next
      }
      fit <- methods[[name]][[method]]
      tol <- tight[[name]][[method]]
      if (!is.null(tol)) fit <- with_tol(tol, fit)
      outcome <- tryCatch({
        if (n == sizes[1]) fit(d)  # warm-up
        times <- c()
        for (rep in 1:3) {
          times <- c(times, system.time(result <- fit(d))[["elapsed"]])
          if (tail(times, 1) > 5) break
        }
        list(time_s = min(times), logl = loglik(result))
      }, error = function(e) list(error = conditionMessage(e)))
      if (!is.null(tol)) outcome[c("rtol", "atol")] <- as.list(tol)
      results[[name]][[method]][[key]] <- outcome
      if (!is.null(outcome$time_s)) {
        cat(sprintf("%s %5d %-16s %9.3f s\n", name, n, method, outcome$time_s))
        # As in run_pynlme.py: SAEM is not skipped on its time at the smallest size.
        if (outcome$time_s > limit && !(grepl("^SAEM", method) && n == sizes[1])) {
          too_slow <- c(too_slow, method)
        }
      } else {
        cat(sprintf("%s %5d %-16s error: %s\n", name, n, method, outcome$error))
      }
    }
  }
}
results$versions <- list(
  r = R.version.string,
  nlme = as.character(packageVersion("nlme")),
  saemix = as.character(packageVersion("saemix")),
  nlmixr2 = as.character(packageVersion("nlmixr2")),
  deSolve = as.character(packageVersion("deSolve"))
)
dir.create("validation/scaling/results", showWarnings = FALSE)
write(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 6),
      "validation/scaling/results/r.json")
