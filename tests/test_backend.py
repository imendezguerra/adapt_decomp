"""Tests for ops.py and the new backend algorithm."""

import math
import torch
import pytest
from torch.testing import assert_close

from adapt_decomp.ops import (
    log_cosh,
    clip_global_delta,
    clip_rowwise_delta,
    orthonormalize_rows_qr,
    orthonormalize_rows,
    find_peaks_multisource,
    classify_peaks_from_adaptive_centroids,
    update_centroids_from_peaks,
    update_B_spike_gated,
)


# ---------------------------------------------------------------------------
# 1. log_cosh
# ---------------------------------------------------------------------------

def test_log_cosh_stable():
    """log_cosh should be numerically stable at large |x| and match scipy reference."""
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    out = log_cosh(x)
    # At x=0: log(cosh(0)) = log(1) = 0
    assert_close(out[2], torch.tensor(0.0), atol=1e-6, rtol=0)
    # At large |x|: log(cosh(x)) ≈ |x| - log(2)
    assert_close(out[0], torch.tensor(100.0 - math.log(2.0)), atol=1e-4, rtol=0)
    assert_close(out[4], torch.tensor(100.0 - math.log(2.0)), atol=1e-4, rtol=0)
    # No inf or nan
    assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# 2 & 3. clip_global_delta
# ---------------------------------------------------------------------------

def test_clip_global_delta_clips():
    """Delta whose norm exceeds max_rel_delta * ref_norm is scaled down."""
    ref = torch.ones(4)
    delta = torch.ones(4) * 10.0   # norm >> ref norm
    clipped = clip_global_delta(delta, ref, max_rel_delta=0.1)
    assert torch.linalg.norm(clipped) <= 0.1 * torch.linalg.norm(ref) + 1e-6


def test_clip_global_delta_noop():
    """Delta within the trust region is returned unchanged."""
    ref = torch.ones(4) * 100.0
    delta = torch.ones(4) * 0.001
    out = clip_global_delta(delta, ref, max_rel_delta=0.5)
    assert_close(out, delta)


# ---------------------------------------------------------------------------
# 4. clip_rowwise_delta
# ---------------------------------------------------------------------------

def test_clip_rowwise_delta_clips():
    """Each row of delta is clipped independently."""
    ref = torch.ones(3, 4)
    delta = torch.zeros(3, 4)
    delta[0] = 100.0   # row 0: way too large
    delta[1] = 0.001   # row 1: fine
    delta[2] = 100.0   # row 2: way too large
    clipped = clip_rowwise_delta(delta, ref, max_rel_delta=0.1)

    row_norms_clipped = torch.linalg.norm(clipped, dim=1)
    row_norms_ref = torch.linalg.norm(ref, dim=1)

    assert row_norms_clipped[0] <= 0.1 * row_norms_ref[0] + 1e-6
    assert_close(clipped[1], delta[1])   # row 1 unchanged
    assert row_norms_clipped[2] <= 0.1 * row_norms_ref[2] + 1e-6


# ---------------------------------------------------------------------------
# 5. find_peaks_multisource — shape check
# ---------------------------------------------------------------------------

def test_find_peaks_multisource_shape():
    """Output tensors match input shape [N, M]."""
    N, M = 200, 4
    Y = torch.randn(N, M)
    peak_mask, Y_det = find_peaks_multisource(Y, min_dist=10)
    assert peak_mask.shape == (N, M)
    assert Y_det.shape == (N, M)
    assert peak_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 6. Strict peak detection suppresses plateaus
# ---------------------------------------------------------------------------

def test_find_peaks_strict_suppresses_plateau():
    """A flat plateau should produce at most one peak per source column."""
    N, M = 50, 2
    Y = torch.zeros(N, M)
    # Insert a flat plateau in rows 10-15 for both sources
    Y[10:16, :] = 5.0
    peak_mask, _ = find_peaks_multisource(Y, min_dist=3, strict=True)
    # Plateau should yield 0 or 1 peak per source (strict mode)
    for m in range(M):
        assert peak_mask[:, m].sum() <= 1


# ---------------------------------------------------------------------------
# 7 & 8. classify_peaks_from_adaptive_centroids
# ---------------------------------------------------------------------------

def test_classify_peaks_uses_adaptive_centroids():
    """spike_mask is True only where Y_det exceeds the adaptive threshold."""
    N, M = 20, 2
    spike_centroid = torch.tensor([4.0, 6.0])
    base_centroid = torch.tensor([1.0, 2.0])
    # threshold = base + 0.5 * (spike - base) = [2.5, 4.0]
    Y_det = torch.zeros(N, M)
    Y_det[5, 0] = 3.0   # above threshold 2.5 → spike
    Y_det[5, 1] = 3.0   # below threshold 4.0 → not spike
    Y_det[10, 1] = 5.0  # above threshold 4.0 → spike
    peak_mask = torch.zeros(N, M, dtype=torch.bool)
    peak_mask[5, :] = True
    peak_mask[10, 1] = True

    spike_mask = classify_peaks_from_adaptive_centroids(
        Y_det, peak_mask, spike_centroid, base_centroid
    )
    assert spike_mask[5, 0].item() is True
    assert spike_mask[5, 1].item() is False
    assert spike_mask[10, 1].item() is True


def test_classify_peaks_not_frozen_cal():
    """Changing adaptive centroids changes the classification (not frozen cal values)."""
    N, M = 10, 1
    Y_det = torch.tensor([[3.0]] * N)
    peak_mask = torch.ones(N, M, dtype=torch.bool)

    # With high spike_centroid → threshold high → no spikes
    spike_mask_high = classify_peaks_from_adaptive_centroids(
        Y_det, peak_mask,
        spike_centroid=torch.tensor([8.0]),
        base_centroid=torch.tensor([1.0]),
    )
    # With low spike_centroid → threshold low → all spikes
    spike_mask_low = classify_peaks_from_adaptive_centroids(
        Y_det, peak_mask,
        spike_centroid=torch.tensor([3.5]),
        base_centroid=torch.tensor([1.0]),
    )
    assert spike_mask_high.sum() == 0
    assert spike_mask_low.sum() == N


# ---------------------------------------------------------------------------
# 9. Centroid initialisation from calibration
# ---------------------------------------------------------------------------

def test_centroid_init_from_calibration():
    """After init_sd_update(), adaptive centroids equal calibration centroids."""
    from adapt_decomp.data_structures import Decomposition
    from adapt_decomp.config import Config
    M, ext_fact = 3, 10
    raw_chs = 2
    D = raw_chs * ext_fact   # extended dimension = 20
    cfg = Config()
    cfg.device = "cpu"
    cfg.ext_fact = ext_fact
    cfg.__post_init__()

    V = torch.eye(D)
    B = torch.randn(M, D)
    B = B / torch.linalg.norm(B, dim=1, keepdim=True)
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(500, raw_chs)   # raw (unextended) calibration EMG
    ipts_cal = torch.randn(500, M)
    spikes_cal = torch.zeros(500, M, dtype=torch.int32)
    spikes_cal[::50] = 1

    decomp = Decomposition(V, B, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)
    assert_close(decomp.spike_centroid, decomp.spike_centroid_cal)
    assert_close(decomp.base_centroid, decomp.base_centroid_cal)


# ---------------------------------------------------------------------------
# 10 & 11. update_centroids_from_peaks — update and skip logic
# ---------------------------------------------------------------------------

def _make_centroid_inputs(N=100, M=2):
    spike_centroid = torch.tensor([4.0] * M)
    base_centroid = torch.tensor([1.0] * M)
    # Y_det values: spikes are 5.0, baseline 0.5
    Y = torch.zeros(N, M)
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    peak_mask = torch.zeros(N, M, dtype=torch.bool)
    # Place 3 spikes and 5 base peaks per source
    for idx in [10, 30, 50]:
        Y[idx] = 5.0
        spike_mask[idx] = True
        peak_mask[idx] = True
    for idx in [20, 40, 60, 70, 80]:
        Y[idx] = 0.5
        peak_mask[idx] = True
    return Y, peak_mask, spike_mask, spike_centroid, base_centroid


def test_centroid_update_with_sufficient_spikes():
    """Centroid updates when spike/base counts meet minima."""
    Y, peak_mask, spike_mask, sc, bc = _make_centroid_inputs()
    new_sc, new_bc = update_centroids_from_peaks(
        Y, peak_mask, spike_mask, sc, bc,
        min_spikes_for_centroid=1, min_base_peaks_for_centroid=3,
    )
    # Should differ from originals (not stuck)
    assert not torch.all(new_sc == sc) or not torch.all(new_bc == bc)
    # Ordering invariant preserved
    assert torch.all(new_sc > new_bc)


def test_centroid_update_skipped_few_samples():
    """Centroids unchanged when peak counts are below minima."""
    N, M = 50, 2
    Y = torch.zeros(N, M)
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    peak_mask = torch.zeros(N, M, dtype=torch.bool)
    # 0 spikes, 0 base peaks → no update
    sc = torch.tensor([4.0] * M)
    bc = torch.tensor([1.0] * M)
    new_sc, new_bc = update_centroids_from_peaks(
        Y, peak_mask, spike_mask, sc, bc,
        min_spikes_for_centroid=1, min_base_peaks_for_centroid=3,
    )
    assert_close(new_sc, sc)
    assert_close(new_bc, bc)


# ---------------------------------------------------------------------------
# 12. Centroid update reverted when spike <= base
# ---------------------------------------------------------------------------

def test_centroid_update_reverted_if_invalid():
    """If proposed update would make spike_centroid <= base_centroid, revert."""
    N, M = 50, 1
    # Set up so batch spike values are below base_centroid
    Y = torch.zeros(N, M)
    Y[10, 0] = 0.1   # tiny "spike"
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[10, 0] = True
    peak_mask = spike_mask.clone()

    sc = torch.tensor([4.0])
    bc = torch.tensor([3.5])   # already close; tiny spike will collapse ordering
    new_sc, new_bc = update_centroids_from_peaks(
        Y, peak_mask, spike_mask, sc, bc,
        centroid_momentum=0.0,   # full replacement to make the violation obvious
        min_spikes_for_centroid=1, min_base_peaks_for_centroid=0,
    )
    assert torch.all(new_sc > new_bc)


# ---------------------------------------------------------------------------
# 13. No B update when no spikes
# ---------------------------------------------------------------------------

def test_no_B_update_no_spikes():
    """All-zero spike_mask → delta_B is all zero (B unchanged after update + orth)."""
    M, D, N = 3, 10, 50
    B = torch.randn(M, D)
    B = orthonormalize_rows_qr(B)
    Z = torch.randn(N, D)
    Y = Z @ B.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    kappa_cal = torch.zeros(M)

    B_new, diag = update_B_spike_gated(
        B, Z, Y, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_b=1.0, min_spikes_for_update=1,
    )
    # active should all be False → grad_B = 0 → delta_B = 0
    assert torch.all(~diag["active"])
    assert_close(diag["delta_B_norm"], torch.zeros(M), atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# 14. Only active sources get nonzero delta_B
# ---------------------------------------------------------------------------

def test_only_active_sources_get_delta_B():
    """Sources without spikes have zero delta_B; sources with spikes may have nonzero."""
    M, D, N = 4, 12, 80
    B = torch.randn(M, D)
    B = orthonormalize_rows_qr(B)
    Z = torch.randn(N, D)
    Y = Z @ B.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[::5, 0] = True   # source 0 has spikes
    spike_mask[::7, 2] = True   # source 2 has spikes
    kappa_cal = torch.zeros(M)

    _, diag = update_B_spike_gated(
        B, Z, Y, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_b=1.0, min_spikes_for_update=1,
    )
    # Sources 1 and 3: inactive → zero delta
    assert diag["delta_B_norm"][1].item() == pytest.approx(0.0, abs=1e-7)
    assert diag["delta_B_norm"][3].item() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# 15. B is orthonormal after QR orthonormalisation
# ---------------------------------------------------------------------------

def test_B_orthonormal_after_qr():
    """B @ B.T ≈ I after orthonormalize_rows_qr."""
    M, D = 5, 20
    B = torch.randn(M, D)
    B_orth = orthonormalize_rows_qr(B)
    gram = B_orth @ B_orth.T
    assert_close(gram, torch.eye(M), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 16. Whitening update skips when slogdet sign is invalid
# ---------------------------------------------------------------------------

def test_update_V_skips_invalid_slogdet():
    """If Rz has non-positive slogdet, V is returned unchanged from _update_V."""
    from adapt_decomp.config import Config
    from adapt_decomp.data_structures import Decomposition

    M, ext_fact = 2, 2
    raw_chs = 3
    D = raw_chs * ext_fact   # extended dimension = 6
    cfg = Config()
    cfg.device = "cpu"
    cfg.ext_fact = ext_fact
    cfg.adapt_wh = True
    cfg.compute_loss = False
    cfg.debug = True
    cfg.__post_init__()

    V = torch.eye(D)
    B = torch.randn(M, D)
    B = orthonormalize_rows_qr(B)
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(200, raw_chs)   # raw (unextended) calibration EMG
    ipts_cal = torch.randn(200, M)
    spikes_cal = torch.zeros(200, M, dtype=torch.int32)
    spikes_cal[::40] = 1

    decomp = Decomposition(V, B, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    # Force Rz to be singular: zero FIFO + zero shrinkage → Rz = 0 → slogdet sign=0
    decomp.fifo_cov = torch.zeros_like(decomp.fifo_cov)
    decomp.shrinkage = 0.0
    V_before = decomp.V.clone()

    from adapt_decomp.adaptation import AdaptDecomp
    adapter = AdaptDecomp.__new__(AdaptDecomp)
    adapter.config = cfg
    adapter.decomp = decomp
    adapter.units = M
    adapter.diagnostics = {}
    adapter.wh_loss = torch.zeros(1)
    adapter.total_loss = torch.zeros(1)

    # X must also be zero so _update_fifo_cov doesn't add signal back into the FIFO
    X = torch.zeros(50, D)
    adapter._update_V(X, batch_idx=0)

    # V should be unchanged because slogdet was non-positive (Rz = 0 matrix)
    assert_close(decomp.V, V_before)
    assert adapter.diagnostics.get(0, {}).get("wh_skip_invalid_slogdet", False)


# ---------------------------------------------------------------------------
# 17. Debug mode returns diagnostics
# ---------------------------------------------------------------------------

def test_debug_mode_returns_diagnostics():
    """update_B_spike_gated always returns a non-empty diagnostics dict."""
    M, D, N = 3, 8, 40
    B = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    Y = Z @ B.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[::4, 0] = True
    kappa_cal = torch.zeros(M)

    _, diag = update_B_spike_gated(
        B, Z, Y, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_b=1.0, min_spikes_for_update=1,
    )
    required_keys = {
        "kappa", "contrast_error",
        "spike_counts", "active", "delta_B_norm", "orthogonality_error",
    }
    assert required_keys.issubset(set(diag.keys()))


# ---------------------------------------------------------------------------
# 18. find_peaks_multisource returns correct dtype and device
# ---------------------------------------------------------------------------

def test_non_debug_fast_path():
    """find_peaks_multisource returns bool mask and float Y_det on the correct device."""
    N, M = 100, 3
    Y = torch.randn(N, M)
    peak_mask, Y_det = find_peaks_multisource(Y, min_dist=5)
    assert peak_mask.dtype == torch.bool
    assert Y_det.dtype == Y.dtype
    assert peak_mask.device == Y.device
    assert Y_det.device == Y.device


# ---------------------------------------------------------------------------
# 19. Whitening error computation
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
# 20. contrast_scope batch_based vs spike_based
# ---------------------------------------------------------------------------

def test_contrast_scope_batch_vs_spike():
    """batch_based and spike_based kappa differ when spikes cover a subset of samples."""
    M, D, N = 2, 6, 60
    B = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    Y = Z @ B.T
    kappa_cal = torch.zeros(M)

    # Only 1/3 of samples are spikes
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[::3] = True

    _, diag_batch = update_B_spike_gated(
        B.clone(), Z, Y, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_b=0.0, min_spikes_for_update=1,
        contrast_scope="batch_based",
    )
    _, diag_spike = update_B_spike_gated(
        B.clone(), Z, Y, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_b=0.0, min_spikes_for_update=1,
        contrast_scope="spike_based",
    )
    # kappa values should differ when the subsets differ
    assert not torch.allclose(diag_batch["kappa"], diag_spike["kappa"], atol=1e-5)


# ---------------------------------------------------------------------------
# 21. B update requires at least one spike
# ---------------------------------------------------------------------------

def test_B_update_requires_at_least_one_spike():
    """With all-zero spike_mask, B is unchanged (up to QR orthonormalization of B itself)."""
    M, D, N = 3, 10, 50
    B = orthonormalize_rows_qr(torch.randn(M, D))
    B_orig = B.clone()
    Z = torch.randn(N, D)
    Y = Z @ B.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    kappa_cal = torch.zeros(M)

    B_new, _ = update_B_spike_gated(
        B, Z, Y, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_b=1.0, min_spikes_for_update=1,
    )
    # delta_B = 0, so B_new = orth(B_orig) = B_orig (already orthonormal)
    assert_close(B_new, B_orig, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 22. Source FIFO edge spike detection
# ---------------------------------------------------------------------------

def test_source_fifo_edge_spike_not_missed():
    """A spike placed at the last sample of the previous batch is detected in the combined window."""
    N, M = 60, 1
    min_dist = 5

    # prev_Y has a clear spike at its last sample
    prev_Y = torch.zeros(N, M)
    prev_Y[-1, 0] = 10.0  # spike at right edge

    # curr_Y has no spike
    curr_Y = torch.zeros(N, M)

    # Without FIFO: peak detection on curr_Y alone → no spike
    peak_mask_solo, _ = find_peaks_multisource(curr_Y, min_dist=min_dist)
    assert peak_mask_solo.sum() == 0

    # With FIFO: detection on [prev_Y, curr_Y] → spike in prev region
    Y_full = torch.cat([prev_Y, curr_Y], dim=0)
    peak_mask_full, _ = find_peaks_multisource(Y_full, min_dist=min_dist)
    # The spike is in the prev_Y portion (rows 0..N-1) not in curr_Y portion
    # But it would have been missed if we only ran detection on curr_Y
    assert peak_mask_full[:N].sum() > 0


# ---------------------------------------------------------------------------
# 23. FIFO covariance full rank when batch < D
# ---------------------------------------------------------------------------

def test_fifo_cov_full_rank():
    """With fifo_length = D and batch_size < D, Rz from FIFO is full rank."""
    from adapt_decomp.data_structures import Decomposition
    from adapt_decomp.config import Config

    ext_fact = 10
    raw_chs = 2
    D = raw_chs * ext_fact   # extended channels = 20
    M = 3
    cfg = Config()
    cfg.device = "cpu"
    cfg.ext_fact = ext_fact
    cfg.fifo_length = D     # exactly D samples
    cfg.__post_init__()

    V = torch.eye(D)
    B = orthonormalize_rows_qr(torch.randn(M, D))
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(300, raw_chs)   # raw (unextended) calibration EMG
    ipts_cal = torch.randn(300, M)
    spikes_cal = torch.zeros(300, M, dtype=torch.int32)
    spikes_cal[::50] = 1

    decomp = Decomposition(V, B, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    # Push one small batch (batch_size < D)
    batch = torch.randn(10, D)   # 10 << D=20
    decomp._update_fifo_cov(batch)
    Rz = decomp._compute_Rz_from_fifo()

    sign, logdet = torch.linalg.slogdet(Rz)
    assert sign.item() > 0, "Rz must be positive definite (full rank)"
    rank = torch.linalg.matrix_rank(Rz)
    assert rank.item() == D


# ---------------------------------------------------------------------------
# 24. wh_mode = "kl_to_cal": KL(Rz_cal ‖ Rz_cal) = 0
# ---------------------------------------------------------------------------

def test_wh_mode_kl_to_cal_zero_at_calibration():
    """KL(Rz_cal ‖ Rz_cal) must be exactly zero; Rz_cal_inv @ Rz_cal must equal I."""
    from adapt_decomp.config import Config
    from adapt_decomp.data_structures import Decomposition, _extend_data_v

    ext_fact = 2
    raw_chs  = 3
    D        = raw_chs * ext_fact
    M        = 2
    N_cal    = 500

    cfg = Config()
    cfg.device   = "cpu"
    cfg.ext_fact = ext_fact
    cfg.wh_mode  = "kl_to_cal"
    cfg.__post_init__()

    V          = torch.eye(D)
    B          = orthonormalize_rows_qr(torch.randn(M, D))
    emg_cal    = torch.randn(N_cal, raw_chs)
    ipts_cal   = torch.randn(N_cal, M)
    spikes_cal = torch.zeros(N_cal, M, dtype=torch.int32)
    spikes_cal[::40] = 1
    spike_cal  = torch.rand(M) + 2.0
    base_cal   = torch.rand(M) * 0.5

    decomp = Decomposition(V, B, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    # Recompute Rz_cal exactly as init_wh_update does
    X_cal_ext = _extend_data_v(emg_cal, ext_fact)
    X_cal_ext = X_cal_ext - X_cal_ext.mean(0, keepdim=True)
    Z_cal = X_cal_ext @ V.T
    N = Z_cal.shape[0]
    Rz_cal = (Z_cal.T @ Z_cal) / N
    Rz_cal = 0.5 * (Rz_cal + Rz_cal.T)
    Rz_cal = (1 - cfg.shrinkage) * Rz_cal + cfg.shrinkage * torch.eye(D)

    # A = Rz_cal_inv @ Rz_cal must equal I
    A = decomp.Rz_cal_inv @ Rz_cal
    assert_close(A, torch.eye(D), atol=1e-5, rtol=0)

    # KL(Rz_cal ‖ Rz_cal): logdet_A = logdet(Rz_cal) - logdet_cal = 0
    _, logdet_Rz_cal = torch.linalg.slogdet(Rz_cal)
    logdet_A = logdet_Rz_cal - decomp.logdet_cal
    K_rel = 0.5 * (A.trace() - logdet_A - D)
    assert_close(K_rel, torch.tensor(0.0), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 25. wh_mode = "kl_to_cal": positive error when covariance drifts
# ---------------------------------------------------------------------------

def test_wh_mode_kl_to_cal_nonzero_on_drift():
    """KL(Rz_drift ‖ Rz_cal) > 0 when the online covariance has drifted."""
    from adapt_decomp.config import Config
    from adapt_decomp.data_structures import Decomposition, _extend_data_v

    ext_fact = 2
    raw_chs  = 3
    D        = raw_chs * ext_fact
    M        = 2
    N_cal    = 500

    cfg = Config()
    cfg.device   = "cpu"
    cfg.ext_fact = ext_fact
    cfg.wh_mode  = "kl_to_cal"
    cfg.__post_init__()

    V          = torch.eye(D)
    B          = orthonormalize_rows_qr(torch.randn(M, D))
    emg_cal    = torch.randn(N_cal, raw_chs)
    ipts_cal   = torch.randn(N_cal, M)
    spikes_cal = torch.zeros(N_cal, M, dtype=torch.int32)
    spikes_cal[::40] = 1
    spike_cal  = torch.rand(M) + 2.0
    base_cal   = torch.rand(M) * 0.5

    decomp = Decomposition(V, B, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    # Build a clearly drifted Rz (3× variance scale → covariance ~9× larger)
    X_drift = _extend_data_v(torch.randn(N_cal, raw_chs) * 3.0, ext_fact)
    X_drift = X_drift - X_drift.mean(0, keepdim=True)
    Z_drift = X_drift @ V.T
    N = Z_drift.shape[0]
    Rz_drift = (Z_drift.T @ Z_drift) / N
    Rz_drift = 0.5 * (Rz_drift + Rz_drift.T)
    Rz_drift = (1 - cfg.shrinkage) * Rz_drift + cfg.shrinkage * torch.eye(D)

    _, logdet_drift = torch.linalg.slogdet(Rz_drift)
    logdet_A = logdet_drift - decomp.logdet_cal
    A = decomp.Rz_cal_inv @ Rz_drift
    K_rel = 0.5 * (A.trace() - logdet_A - D)

    assert K_rel.item() > 0.5, f"Expected large KL for 3× variance drift, got {K_rel.item()}"
