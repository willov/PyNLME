import shutil
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow simulation studies")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def r_available() -> bool:
    """Whether Rscript with the nlme and jsonlite packages is available."""
    if shutil.which("Rscript") is None:
        return False
    check = (
        'if (dir.exists("~/R/library")) .libPaths(c("~/R/library", .libPaths())); '
        'quit(status = !all(sapply(c("nlme", "jsonlite"), requireNamespace, quietly = TRUE)))'
    )
    return subprocess.run(["Rscript", "-e", check], capture_output=True).returncode == 0
