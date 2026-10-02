"""Cross-platform reproducibility of the tutorial adaptation.

Re-runs notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb's
adaptation (section 2.2, no parameter saving) and checks it against a stored
reference. Run in CI on Linux, macOS and Windows from the pinned environment.yaml.

Different platforms use different BLAS libraries (MKL, OpenBLAS, Accelerate), so
results are not bit-identical -- the checks are tolerance-based but tight enough
that any change in the algorithm's behaviour fails them.

Needs the NeuroMotion data (`make data`, ~1.6 GB); skipped when it is absent.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from adapt_decomp.spikes import rate_of_agreement_paired
from tests.reproducibility.fingerprint import (
    PATH_REFERENCE,
    TOL_SPIKE_MS,
    data_available,
    environment_info,
    load_fingerprint,
    run_tutorial_adaptation,
    save_fingerprint,
)

pytestmark = [
    pytest.mark.repro,
    pytest.mark.skipif(not data_available(), reason="NeuroMotion data missing (run `make data`)"),
]

MIN_ROA_TO_REFERENCE = 0.99  # per unit, run's spikes vs reference spikes
MAX_ROA_GT_DIFF = 0.005  # per unit, |RoA vs ground truth - reference's| (0.5 pp)
NOTEBOOK_MEAN_ROA_GT = 0.9064  # published in the tutorial notebook
LOSS_RTOL = 1e-3
# sv_loss has entries near zero (median |sv_loss| ~ 2), where a relative tolerance alone
# is meaningless -- MKL vs OpenBLAS differ by up to ~3e-3 in absolute terms.
SV_LOSS_ATOL = 1e-2


@pytest.fixture(scope="module")
def run():
    fingerprint = run_tutorial_adaptation()
    yield fingerprint
    # Keep the run's fingerprint for diffing against the reference (uploaded by CI).
    out_dir = os.environ.get("REPRO_ARTIFACT_DIR")
    if out_dir:
        save_fingerprint(fingerprint, Path(out_dir) / "neuromotion_adapt.npz")


@pytest.fixture(scope="module")
def reference():
    if not PATH_REFERENCE.exists():
        pytest.fail(f"{PATH_REFERENCE} missing -- run tests/reproducibility/make_reference.py")
    return load_fingerprint(PATH_REFERENCE)


def test_spike_trains_match_reference(run, reference):
    assert run["spikes"].shape == reference["spikes"].shape
    roa, _, _ = rate_of_agreement_paired(
        reference["spikes"], run["spikes"], fs=int(run["fs"]), tol_spike_ms=TOL_SPIKE_MS
    )
    worst = int(np.argmin(roa))
    assert roa.min() >= MIN_ROA_TO_REFERENCE, (
        f"unit {worst}: RoA to reference {roa[worst]:.4f} < {MIN_ROA_TO_REFERENCE} "
        f"(reference: {reference['environment']}; this run: {environment_info()})"
    )


def test_accuracy_matches_reference(run, reference):
    np.testing.assert_allclose(run["roa_gt"], reference["roa_gt"], atol=MAX_ROA_GT_DIFF)
    assert run["roa_gt"].mean() == pytest.approx(NOTEBOOK_MEAN_ROA_GT, abs=MAX_ROA_GT_DIFF)


def test_losses_match_reference(run, reference):
    np.testing.assert_allclose(run["wh_loss"], reference["wh_loss"], rtol=LOSS_RTOL)
    np.testing.assert_allclose(
        run["sv_loss"], reference["sv_loss"], rtol=LOSS_RTOL, atol=SV_LOSS_ATOL
    )


def test_final_parameters_match_reference(run, reference):
    for key in ("whitening_norm", "sep_vectors_norm"):
        np.testing.assert_allclose(run[key], reference[key], rtol=LOSS_RTOL, err_msg=key)
