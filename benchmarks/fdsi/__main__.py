"""Entry point: python -m benchmarks.fdsi <command>, from the repository root."""

import os

# One thread per run, fixed before numpy and torch load, so results don't depend on the machine
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

from benchmarks.fdsi.cli import app  # noqa: E402

app()
