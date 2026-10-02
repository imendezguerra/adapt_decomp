"""Regenerate the reference for test_tutorial_adaptation.py.

Only do this when results are *meant* to change (and say so in CHANGELOG.md):

    python tests/reproducibility/make_reference.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.reproducibility.fingerprint import (
    PATH_REFERENCE,
    data_available,
    environment_info,
    run_tutorial_adaptation,
    save_fingerprint,
)

if __name__ == "__main__":
    if not data_available():
        sys.exit("NeuroMotion data missing -- run `make data` first.")
    fingerprint = run_tutorial_adaptation()
    save_fingerprint(fingerprint, PATH_REFERENCE)
    roa = fingerprint["roa_gt"] * 100
    print(f"Wrote {PATH_REFERENCE.relative_to(Path.cwd())} ({environment_info()})")
    print(f"RoA vs ground truth: {roa.mean():.2f} ± {roa.std():.2f} (med: {np.median(roa):.2f}) %")
