#!/usr/bin/env bash
# Run all benchmarks with what is installed, then print the tables.
#
#   bash validation/run_benchmarks.sh            # everything (about two hours)
#   SCALING_SIZES=10,100 bash validation/run_benchmarks.sh   # skip 1000 subjects
#
# Run from the repository root, on an otherwise idle machine (other load changes
# the timings). pynlme always runs. R (nlme, saemix, nlmixr2, deSolve, jsonlite) runs
# when Rscript finds these packages; MATLAB when PYNLME_MATLAB names the MATLAB
# command (as for the tests). Programs that are missing keep their stored results in
# validation/reference/ and validation/scaling/results/, measured on the machine
# described in the README.
set -euo pipefail

have_r() {
    command -v Rscript >/dev/null && Rscript -e \
        'if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths()));
         ok <- all(sapply(c("nlme", "saemix", "nlmixr2", "deSolve", "jsonlite"),
                          requireNamespace, quietly = TRUE)); quit(status = !ok)' \
        >/dev/null 2>&1
}

run_matlab() {
    # PYNLME_MATLAB may start with environment assignments (e.g. LD_PRELOAD=...).
    # Output goes to a file: MATLAB's helper processes keep pipes open.
    env ${PYNLME_MATLAB} -batch "$1" >"validation/matlab_run.log" 2>&1 </dev/null
}

echo "== Validation cases =="
if have_r; then
    Rscript validation/r/benchmark.R >/dev/null
    Rscript validation/r/nlmixr2_benchmark.R >/dev/null
else
    echo "R or its packages not found: using the stored R and nlmixr2 timings"
fi
if [ -n "${PYNLME_MATLAB:-}" ]; then
    run_matlab "addpath validation/matlab; benchmark"
else
    echo "PYNLME_MATLAB not set: using the stored MATLAB timings"
fi
uv run python validation/benchmark.py --markdown
uv run python validation/nlmixr2_compare.py

echo "== Accuracy (bias, RMSE, coverage; not timing) =="
uv run python validation/accuracy/simulate.py
uv run python validation/accuracy/run_pynlme.py
if have_r; then
    Rscript validation/accuracy/run_r.R
fi
if [ -n "${PYNLME_MATLAB:-}" ]; then
    run_matlab "addpath validation/accuracy; run_matlab"
fi
uv run python validation/accuracy/report.py

echo "== Scaling =="
uv run python validation/scaling/make_data.py
uv run python validation/scaling/run_pynlme.py
if have_r; then
    Rscript validation/scaling/run_r.R
fi
if [ -n "${PYNLME_MATLAB:-}" ]; then
    run_matlab "addpath validation/scaling; run_matlab"
fi
uv run python validation/scaling/report.py
