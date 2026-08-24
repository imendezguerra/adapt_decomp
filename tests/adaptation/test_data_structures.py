"""Tests for adaptation/data_structures.py: Decomposition's calibration-time
state and AdaptationResult's save/load contract.
"""

import torch
import pytest
from torch.testing import assert_close

from adapt_decomp.preprocessing import extend_data


# ---------------------------------------------------------------------------
# Centroid initialisation from calibration
# ---------------------------------------------------------------------------

def test_centroid_init_from_calibration(make_decomposition):
    """After init_sd_update(), adaptive centroids equal calibration centroids."""
    decomp, _ = make_decomposition(
        M=3, ext_fact=10, raw_chs=2, n_cal=500, spike_stride=50, orthonormal_sv=False,
    )
    assert_close(decomp.spikes_centr, decomp.spikes_centr_cal)
    assert_close(decomp.base_centr, decomp.base_centr_cal)


# ---------------------------------------------------------------------------
# FIFO covariance full rank when batch < D
# ---------------------------------------------------------------------------

def test_fifo_cov_full_rank(make_decomposition):
    """With fifo_length = D and batch_size < D, Rz from FIFO is full rank."""
    ext_fact, raw_chs, M = 10, 2, 3
    D = raw_chs * ext_fact   # extended channels = 20
    decomp, _ = make_decomposition(
        M=M, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=300, spike_stride=50,
        fifo_length=D,   # exactly D samples
    )

    # Push one small batch (batch_size < D)
    batch = torch.randn(10, D)   # 10 << D=20
    decomp._update_fifo_cov(batch)
    Rz = decomp._compute_Rz_from_fifo()

    sign, logdet = torch.linalg.slogdet(Rz)
    assert sign.item() > 0, "Rz must be positive definite (full rank)"
    rank = torch.linalg.matrix_rank(Rz)
    assert rank.item() == D


# ---------------------------------------------------------------------------
# wh_mode = "kl_to_cal": KL(Rz_cal ‖ Rz_cal) = 0, and > 0 when drifted
# ---------------------------------------------------------------------------

def test_wh_mode_kl_to_cal_zero_at_calibration(make_decomposition):
    """KL(Rz_cal ‖ Rz_cal) must be exactly zero; Rz_cal_inv @ Rz_cal must equal I."""
    ext_fact, raw_chs, M = 2, 3, 2
    D = raw_chs * ext_fact
    decomp, _ = make_decomposition(
        M=M, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=500, spike_stride=40,
        wh_mode="kl_to_cal",
    )

    # Recompute Rz_cal exactly as init_wh_update does
    X_cal_ext = extend_data(decomp.emg_calib, ext_fact)
    X_cal_ext = X_cal_ext - X_cal_ext.mean(0, keepdim=True)
    Z_cal = X_cal_ext @ decomp.whitening.T
    N = Z_cal.shape[0]
    Rz_cal = (Z_cal.T @ Z_cal) / N
    Rz_cal = 0.5 * (Rz_cal + Rz_cal.T)
    Rz_cal = (1 - decomp.shrinkage) * Rz_cal + decomp.shrinkage * torch.eye(D)

    # A = Rz_cal_inv @ Rz_cal must equal I
    A = decomp.Rz_cal_inv @ Rz_cal
    assert_close(A, torch.eye(D), atol=1e-5, rtol=0)

    # KL(Rz_cal ‖ Rz_cal): logdet_A = logdet(Rz_cal) - logdet_cal = 0
    _, logdet_Rz_cal = torch.linalg.slogdet(Rz_cal)
    logdet_A = logdet_Rz_cal - decomp.logdet_cal
    K_rel = 0.5 * (A.trace() - logdet_A - D)
    assert_close(K_rel, torch.tensor(0.0), atol=1e-5, rtol=0)


def test_wh_mode_kl_to_cal_nonzero_on_drift(make_decomposition):
    """KL(Rz_drift ‖ Rz_cal) > 0 when the online covariance has drifted."""
    ext_fact, raw_chs, M, N_cal = 2, 3, 2, 500
    D = raw_chs * ext_fact
    decomp, _ = make_decomposition(
        M=M, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=N_cal, spike_stride=40,
        wh_mode="kl_to_cal",
    )

    # Build a clearly drifted Rz (3× variance scale → covariance ~9× larger)
    X_drift = extend_data(torch.randn(N_cal, raw_chs) * 3.0, ext_fact)
    X_drift = X_drift - X_drift.mean(0, keepdim=True)
    Z_drift = X_drift @ decomp.whitening.T
    N = Z_drift.shape[0]
    Rz_drift = (Z_drift.T @ Z_drift) / N
    Rz_drift = 0.5 * (Rz_drift + Rz_drift.T)
    Rz_drift = (1 - decomp.shrinkage) * Rz_drift + decomp.shrinkage * torch.eye(D)

    _, logdet_drift = torch.linalg.slogdet(Rz_drift)
    logdet_A = logdet_drift - decomp.logdet_cal
    A = decomp.Rz_cal_inv @ Rz_drift
    K_rel = 0.5 * (A.trace() - logdet_A - D)

    assert K_rel.item() > 0.5, f"Expected large KL for 3× variance drift, got {K_rel.item()}"


# ---------------------------------------------------------------------------
# Whitening error computation
#
# NOTE: this test doesn't exercise any adapt_decomp code -- it just replays
# the KL/whitening-error formula inline, and its main assertion
# (assert_close(e_v_raw, K - K_cal)) is tautological by construction. Kept
# as-is (unit-test-shaped documentation of the formula) rather than rewired
# to call Decomposition directly, since that's a behavioural change beyond
# this reorganisation -- worth revisiting separately.
# ---------------------------------------------------------------------------

def test_whitening_error_computation():
    """Verify e_v_raw = K - K_cal and that K >= 0 for near-identity Rz."""
    D = 8
    I = torch.eye(D)
    # Build a Rz slightly off identity
    noise = torch.randn(D, D) * 0.05
    Rz = I + 0.5 * (noise + noise.T)
    Rz = (1 - 1e-3) * Rz + 1e-3 * I   # shrinkage

    sign, logdet = torch.linalg.slogdet(Rz)
    K = 0.5 * (Rz.trace() - logdet - D)
    K_cal = torch.tensor(0.1)

    e_v_raw = K - K_cal
    assert_close(e_v_raw, K - K_cal)
    # K should be small positive for near-identity Rz
    assert K.item() >= 0.0


# ---------------------------------------------------------------------------
# ext_mode is honoured by Decomposition, not silently defaulted to "block"
# ---------------------------------------------------------------------------

def test_decomposition_uses_configured_ext_mode(make_decomposition):
    """Decomposition.init_wh_update must extend emg_calib with AdaptConfig.ext_mode,
    not silently fall back to 'block' -- verified via the FIFO buffer it seeds."""
    ext_fact = 2
    decomp, _ = make_decomposition(
        M=2, ext_fact=ext_fact, raw_chs=3, n_cal=300, spike_stride=40,
        ext_mode="toeplitz",
    )

    expected = extend_data(decomp.emg_calib, ext_fact, ext_mode="toeplitz")
    expected = expected - expected.mean(0, keepdim=True)
    assert_close(decomp.fifo_cov, expected[-decomp.fifo_samples:])


# ---------------------------------------------------------------------------
# AdaptationResult save/load (pickle round-trip)
# ---------------------------------------------------------------------------

def test_adaptation_result_save_load_roundtrip(tmp_path):
    """AdaptationResult.save()/.load() should round-trip tensors via pickle."""
    from adapt_decomp.adaptation.data_structures import AdaptationResult

    batches, M = 5, 3
    result = AdaptationResult(
        spikes=torch.zeros(batches, M, dtype=torch.int32),
        ipts=torch.randn(batches, M),
        wh_time_ms=torch.rand(batches),
        sv_time_ms=torch.rand(batches),
        sd_time_ms=torch.rand(batches),
        total_time_ms=torch.rand(batches),
    )

    path = tmp_path / "adaptation_result.pkl"
    result.save(path)
    loaded = AdaptationResult.load(path)

    assert isinstance(loaded, AdaptationResult)
    assert_close(loaded.ipts, result.ipts)
    assert loaded.wh_loss is None


def test_adaptation_result_load_rejects_wrong_type(tmp_path):
    """AdaptationResult.load() must raise ValueError if the pickle holds a different type."""
    import pickle
    from adapt_decomp.adaptation.data_structures import AdaptationResult

    path = tmp_path / "not_a_result.pkl"
    with open(path, "wb") as f:
        pickle.dump([1, 2, 3], f)

    with pytest.raises(ValueError):
        AdaptationResult.load(path)
