# Export the validation datasets to CSV.
#
# Run from the repository root:  Rscript validation/r/datasets.R
# The CSVs are committed, so this only needs re-running if a dataset is added.

out <- "validation/datasets"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

num <- function(f) as.integer(as.character(f))

# Indomethacin: same data as MATLAB's indomethacin.mat (6 subjects x 11 times).
indometh <- data.frame(subject = num(Indometh$Subject),
                       time = Indometh$time,
                       conc = Indometh$conc)
indometh <- indometh[order(indometh$subject, indometh$time), ]
write.csv(indometh, file.path(out, "indometh.csv"), row.names = FALSE)

# Theophylline: one-compartment oral PK, dose and weight are per-subject covariates.
theoph <- data.frame(subject = num(Theoph$Subject),
                     wt = Theoph$Wt,
                     dose = Theoph$Dose,
                     time = Theoph$Time,
                     conc = Theoph$conc)
theoph <- theoph[order(theoph$subject, theoph$time), ]
write.csv(theoph, file.path(out, "theoph.csv"), row.names = FALSE)

# Orange trees: logistic growth, 5 trees x 7 ages.
orange <- data.frame(tree = num(Orange$Tree),
                     age = Orange$age,
                     circumference = Orange$circumference)
orange <- orange[order(orange$tree, orange$age), ]
write.csv(orange, file.path(out, "orange.csv"), row.names = FALSE)

# Simulated linear mixed model with correlated random intercept and slope.
# A linear model function in nlmefit must reproduce the exact LME answer.
set.seed(20261004)
m <- 20
x <- rep(0:5, m)
id <- rep(seq_len(m), each = 6)
psi <- matrix(c(0.5^2, 0.3 * 0.5 * 0.2, 0.3 * 0.5 * 0.2, 0.2^2), 2)
b <- matrix(rnorm(2 * m), m) %*% chol(psi)
y <- (2 + b[id, 1]) + (0.5 + b[id, 2]) * x + rnorm(length(x), sd = 0.3)
write.csv(data.frame(id = id, x = x, y = round(y, 6)),
          file.path(out, "linear_sim.csv"), row.names = FALSE)
