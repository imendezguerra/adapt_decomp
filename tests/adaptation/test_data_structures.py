"""Tests for adaptation/data_structures.py: Decomposition's calibration-time
state, Data's channel selection, RawData's minimal batch-serving contract,
and AdaptationResult's save/load contract.
"""

import numpy as np
import torch
import pytest
from torch.testing import assert_close

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import Data, RawData
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
# Data channel selection (ch_mask/ch_map/replace_bad_channels)
# ---------------------------------------------------------------------------

def test_data_ch_mask_drop_shrinks_emg_ext_width():
    """ch_mask set (drop mode): Data.emg_ext's channel width reflects the
    surviving (good) channels only, not the raw channel count. preprocess=True,
    since channel selection only runs alongside filtering (see Data.__init__)."""
    ext_fact, raw_chs = 2, 4
    emg = torch.randn(100, raw_chs)
    ch_mask = np.array([True, False, True, True])   # 3 good channels
    config = AdaptConfig(ext_fact=ext_fact, ch_mask=ch_mask)
    data = Data(emg, preprocess=True, config=config)
    assert data.emg_ext.shape[1] == 3 * ext_fact


def test_data_ch_mask_none_keeps_raw_width():
    """ch_mask=None (default): Data.emg_ext's width is unaffected -- the
    existing no-selection behaviour is unchanged."""
    ext_fact, raw_chs = 2, 4
    emg = torch.randn(100, raw_chs)
    config = AdaptConfig(ext_fact=ext_fact)
    data = Data(emg, preprocess=False, config=config)
    assert data.emg_ext.shape[1] == raw_chs * ext_fact


def test_data_replace_bad_channels_without_ch_map_raises():
    """replace_bad_channels=True with ch_map unset must raise ValueError --
    interpolation is impossible without the electrode grid. preprocess=True,
    since channel selection only runs alongside filtering (see Data.__init__)."""
    emg = torch.randn(50, 3)
    config = AdaptConfig(ext_fact=2, replace_bad_channels=True, ch_map=None)
    with pytest.raises(ValueError, match="ch_map"):
        Data(emg, preprocess=True, config=config)


def test_data_ch_mask_length_mismatch_raises():
    """ch_mask whose length disagrees with the raw emg's channel count must
    raise ValueError, not silently misalign or crash deep in extend_data.
    preprocess=True, since channel selection only runs alongside filtering
    (see Data.__init__)."""
    emg = torch.randn(50, 4)
    ch_mask = np.array([True, False, True])   # length 3, emg has 4 channels
    config = AdaptConfig(ext_fact=2, ch_mask=ch_mask)
    with pytest.raises(ValueError, match="ch_mask"):
        Data(emg, preprocess=True, config=config)


# ---------------------------------------------------------------------------
# RawData: minimal Dataset contract for the streaming mode
# ---------------------------------------------------------------------------

def test_raw_data_length_matches_emg_rows():
    emg = torch.randn(37, 4)
    raw = RawData(emg, AdaptConfig(device="cpu"))
    assert len(raw) == 37


def test_raw_data_getitem_returns_raw_row_and_label():
    emg = torch.randn(10, 3)
    raw = RawData(emg, AdaptConfig(device="cpu"))
    row, label = raw[4]
    assert_close(row, emg[4, :].float())
    assert label.item() == 4


def test_raw_data_dtype_and_device():
    emg = torch.randn(5, 2, dtype=torch.float64)
    raw = RawData(emg, AdaptConfig(device="cpu"))
    assert raw.emg.dtype == torch.float32
    assert raw.emg.device.type == "cpu"


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
        preprocess_time_ms=torch.zeros(batches),
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
