"""Tests for spikes/comparison.py: pair_ground_truth()."""

import numpy as np

from adapt_decomp.spikes.comparison import pair_ground_truth


def test_pair_ground_truth_reorders_gt_to_match_calibration_units():
    """pair_ground_truth() should reindex ground truth to the calibration's own
    unit order, via rate_of_agreement's greedy (non-Hungarian) matching."""
    n_samples, fs = 200, 1000
    dec0 = np.zeros(n_samples, dtype=np.int32)
    dec0[10:200:20] = 1  # 10, 30, 50, ...
    dec1 = np.zeros(n_samples, dtype=np.int32)
    dec1[20:200:20] = 1  # 20, 40, 60, ...
    spikes_calib = np.column_stack([dec0, dec1])

    gt_unrelated = np.zeros(n_samples, dtype=np.int32)
    gt_unrelated[[3, 150]] = 1
    spikes_gt = np.column_stack([dec1.copy(), gt_unrelated, dec0.copy()])  # gt idx 0->dec1, 2->dec0

    gt_full_bin, roa_calib = pair_ground_truth(spikes_gt, spikes_calib, fs=fs, tol_spike_ms=2.0)

    assert gt_full_bin is not None
    assert gt_full_bin.shape[0] == n_samples
    # Column order follows spikes_calib's own unit order (dec0, dec1) --
    # dec0 was paired with gt unit 2, dec1 with gt unit 0.
    np.testing.assert_array_equal(gt_full_bin[:, 0], spikes_gt[:, 2])
    np.testing.assert_array_equal(gt_full_bin[:, 1], spikes_gt[:, 0])
    assert roa_calib is not None
    np.testing.assert_allclose(roa_calib, np.ones_like(roa_calib), atol=1e-6)


def test_pair_ground_truth_returns_none_when_no_calibration_units():
    """No decomposed units -> nothing to align to -> (None, None)."""
    spikes_gt = np.zeros((50, 2), dtype=np.int32)
    spikes_calib = np.zeros((50, 0), dtype=np.int32)
    gt_full_bin, roa_calib = pair_ground_truth(spikes_gt, spikes_calib, fs=1000)
    assert gt_full_bin is None
    assert roa_calib is None
