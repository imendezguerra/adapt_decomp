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
    orthonormalize_rows_gram_schmidt,
    find_peaks_multisource,
    classify_peaks_from_adaptive_centroids,
    update_centroids_from_peaks,
    update_sv_spike_gated,
    gate_spikes_by_iqr,
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
    sources = torch.randn(N, M)
    peak_mask, sources_det = find_peaks_multisource(sources, min_dist=10)
    assert peak_mask.shape == (N, M)
    assert sources_det.shape == (N, M)
    assert peak_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 6. Strict peak detection suppresses plateaus
# ---------------------------------------------------------------------------

def test_find_peaks_strict_suppresses_plateau():
    """A flat plateau should produce at most one peak per source column."""
    N, M = 50, 2
    sources = torch.zeros(N, M)
    # Insert a flat plateau in rows 10-15 for both sources
    sources[10:16, :] = 5.0
    peak_mask, _ = find_peaks_multisource(sources, min_dist=3, strict=True)
    # Plateau should yield 0 or 1 peak per source (strict mode)
    for m in range(M):
        assert peak_mask[:, m].sum() <= 1


# ---------------------------------------------------------------------------
# 7 & 8. classify_peaks_from_adaptive_centroids
# ---------------------------------------------------------------------------

def test_classify_peaks_uses_adaptive_centroids():
    """spike_mask is True only where sources_det exceeds the adaptive threshold."""
    N, M = 20, 2
    spike_centroids = torch.tensor([4.0, 6.0])
    base_centroids = torch.tensor([1.0, 2.0])
    # threshold = base + 0.5 * (spike - base) = [2.5, 4.0]
    sources_det = torch.zeros(N, M)
    sources_det[5, 0] = 3.0   # above threshold 2.5 → spike
    sources_det[5, 1] = 3.0   # below threshold 4.0 → not spike
    sources_det[10, 1] = 5.0  # above threshold 4.0 → spike
    peak_mask = torch.zeros(N, M, dtype=torch.bool)
    peak_mask[5, :] = True
    peak_mask[10, 1] = True

    spike_mask = classify_peaks_from_adaptive_centroids(
        sources_det, peak_mask, spike_centroids, base_centroids
    )
    assert spike_mask[5, 0].item() is True
    assert spike_mask[5, 1].item() is False
    assert spike_mask[10, 1].item() is True


def test_classify_peaks_not_frozen_cal():
    """Changing adaptive centroids changes the classification (not frozen cal values)."""
    N, M = 10, 1
    sources_det = torch.tensor([[3.0]] * N)
    peak_mask = torch.ones(N, M, dtype=torch.bool)

    # With high spike_centroids → threshold high → no spikes
    spike_mask_high = classify_peaks_from_adaptive_centroids(
        sources_det, peak_mask,
        spike_centroids=torch.tensor([8.0]),
        base_centroids=torch.tensor([1.0]),
    )
    # With low spike_centroids → threshold low → all spikes
    spike_mask_low = classify_peaks_from_adaptive_centroids(
        sources_det, peak_mask,
        spike_centroids=torch.tensor([3.5]),
        base_centroids=torch.tensor([1.0]),
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

    wh = torch.eye(D)
    sv = torch.randn(M, D)
    sv = sv / torch.linalg.norm(sv, dim=1, keepdim=True)
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(500, raw_chs)   # raw (unextended) calibration EMG
    ipts_cal = torch.randn(500, M)
    spikes_cal = torch.zeros(500, M, dtype=torch.int32)
    spikes_cal[::50] = 1

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)
    assert_close(decomp.spikes_centr, decomp.spikes_centr_cal)
    assert_close(decomp.base_centr, decomp.base_centr_cal)


# ---------------------------------------------------------------------------
# 10 & 11. update_centroids_from_peaks — update and skip logic
# ---------------------------------------------------------------------------

def _make_centroid_inputs(N=100, M=2):
    spike_centroids = torch.tensor([4.0] * M)
    base_centroids = torch.tensor([1.0] * M)
    # sources_det values: spikes are 5.0, baseline 0.5
    sources = torch.zeros(N, M)
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    peak_mask = torch.zeros(N, M, dtype=torch.bool)
    # Place 3 spikes and 5 base peaks per source
    for idx in [10, 30, 50]:
        sources[idx] = 5.0
        spike_mask[idx] = True
        peak_mask[idx] = True
    for idx in [20, 40, 60, 70, 80]:
        sources[idx] = 0.5
        peak_mask[idx] = True
    return sources, peak_mask, spike_mask, spike_centroids, base_centroids


def test_centroid_update_with_sufficient_spikes():
    """Centroid updates when spike/base counts meet minima."""
    sources, peak_mask, spike_mask, sc, bc = _make_centroid_inputs()
    new_sc, new_bc = update_centroids_from_peaks(
        sources, peak_mask, spike_mask, sc, bc,
        min_spikes_for_centroid=1, min_base_peaks_for_centroid=3,
    )
    # Should differ from originals (not stuck)
    assert not torch.all(new_sc == sc) or not torch.all(new_bc == bc)
    # Ordering invariant preserved
    assert torch.all(new_sc > new_bc)


def test_centroid_update_skipped_few_samples():
    """Centroids unchanged when peak counts are below minima."""
    N, M = 50, 2
    sources = torch.zeros(N, M)
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    peak_mask = torch.zeros(N, M, dtype=torch.bool)
    # 0 spikes, 0 base peaks → no update
    sc = torch.tensor([4.0] * M)
    bc = torch.tensor([1.0] * M)
    new_sc, new_bc = update_centroids_from_peaks(
        sources, peak_mask, spike_mask, sc, bc,
        min_spikes_for_centroid=1, min_base_peaks_for_centroid=3,
    )
    assert_close(new_sc, sc)
    assert_close(new_bc, bc)


# ---------------------------------------------------------------------------
# 12. Centroid update reverted when spike <= base
# ---------------------------------------------------------------------------

def test_centroid_update_reverted_if_invalid():
    """If proposed update would make spike_centroids <= base_centroids, revert."""
    N, M = 50, 1
    # Set up so batch spike values are below base_centroids
    sources = torch.zeros(N, M)
    sources[10, 0] = 0.1   # tiny "spike"
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[10, 0] = True
    peak_mask = spike_mask.clone()

    sc = torch.tensor([4.0])
    bc = torch.tensor([3.5])   # already close; tiny spike will collapse ordering
    new_sc, new_bc = update_centroids_from_peaks(
        sources, peak_mask, spike_mask, sc, bc,
        centroid_momentum=0.0,   # full replacement to make the violation obvious
        min_spikes_for_centroid=1, min_base_peaks_for_centroid=0,
    )
    assert torch.all(new_sc > new_bc)


# ---------------------------------------------------------------------------
# 13. No sv update when no spikes
# ---------------------------------------------------------------------------

def test_no_sv_update_no_spikes():
    """All-zero spike_mask → delta_sv is all zero (sv unchanged after update + orth)."""
    M, D, N = 3, 10, 50
    sv = torch.randn(M, D)
    sv = orthonormalize_rows_qr(sv)
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    kappa_cal = torch.zeros(M)

    sv_new, diag = update_sv_spike_gated(
        sv, Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=1.0, min_spikes_for_update=1,
        contrast_scope="spike_based",
    )
    # active should all be False → grad_sv = 0 → delta_sv = 0
    assert torch.all(~diag["active"])
    assert_close(diag["delta_sv_norm"], torch.zeros(M), atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# 14. Only active sources get nonzero delta_sv
# ---------------------------------------------------------------------------

def test_only_active_sources_get_delta_sv():
    """Sources without spikes have zero delta_sv; sources with spikes may have nonzero."""
    M, D, N = 4, 12, 80
    sv = torch.randn(M, D)
    sv = orthonormalize_rows_qr(sv)
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[::5, 0] = True   # source 0 has spikes
    spike_mask[::7, 2] = True   # source 2 has spikes
    kappa_cal = torch.zeros(M)

    _, diag = update_sv_spike_gated(
        sv, Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=1.0, min_spikes_for_update=1,
        contrast_scope="spike_based",
    )
    # Sources 1 and 3: inactive → zero delta
    assert diag["delta_sv_norm"][1].item() == pytest.approx(0.0, abs=1e-7)
    assert diag["delta_sv_norm"][3].item() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# 15. sv is orthonormal after QR orthonormalisation
# ---------------------------------------------------------------------------

def test_sv_orthonormal_after_qr():
    """sv @ sv.T ≈ I after orthonormalize_rows_qr."""
    M, D = 5, 20
    sv = torch.randn(M, D)
    sv_orth = orthonormalize_rows_qr(sv)
    gram = sv_orth @ sv_orth.T
    assert_close(gram, torch.eye(M), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 16. sv is orthonormal after Gram-Schmidt orthonormalization
# ---------------------------------------------------------------------------

def test_sv_orthonormal_after_gram_schmidt():
    """sv @ sv.T ≈ I after orthonormalize_rows_gram_schmidt."""
    M, D = 5, 20
    sv = torch.randn(M, D)
    sv_orth = orthonormalize_rows_gram_schmidt(sv)
    gram = sv_orth @ sv_orth.T
    assert_close(gram, torch.eye(M), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 17. Whitening update skips when slogdet sign is invalid
# ---------------------------------------------------------------------------

def test_update_wh_skips_invalid_slogdet():
    """If Rz has non-positive slogdet, wh is returned unchanged from _update_wh."""
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

    wh = torch.eye(D)
    sv = torch.randn(M, D)
    sv = orthonormalize_rows_qr(sv)
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(200, raw_chs)   # raw (unextended) calibration EMG
    ipts_cal = torch.randn(200, M)
    spikes_cal = torch.zeros(200, M, dtype=torch.int32)
    spikes_cal[::40] = 1

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    # Force Rz to be singular: zero FIFO + zero shrinkage → Rz = 0 → slogdet sign=0
    decomp.fifo_cov = torch.zeros_like(decomp.fifo_cov)
    decomp.shrinkage = 0.0
    wh_before = decomp.whitening.clone()

    from adapt_decomp.adaptation import AdaptDecomp
    adapter = AdaptDecomp.__new__(AdaptDecomp)
    adapter.config = cfg
    adapter.decomp = decomp
    adapter.units = M
    adapter.diagnostics = {}
    adapter.wh_loss = torch.zeros(1)
    adapter.wh_trace = torch.zeros(1)
    adapter.total_loss = torch.zeros(1)

    # X must also be zero so _update_fifo_cov doesn't add signal back into the FIFO
    X = torch.zeros(50, D)
    adapter._update_wh(X, batch_idx=0)

    # wh should be unchanged because slogdet was non-positive (Rz = 0 matrix)
    assert_close(decomp.whitening, wh_before)
    assert adapter.diagnostics.get(0, {}).get("wh_skip_invalid_slogdet", False)


# ---------------------------------------------------------------------------
# 17. Debug mode returns diagnostics
# ---------------------------------------------------------------------------

def test_debug_mode_returns_diagnostics():
    """update_sv_spike_gated always returns a non-empty diagnostics dict."""
    M, D, N = 3, 8, 40
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[::4, 0] = True
    kappa_cal = torch.zeros(M)

    _, diag = update_sv_spike_gated(
        sv, Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=1.0, min_spikes_for_update=1,
    )
    required_keys = {
        "kappa", "contrast_error",
        "spike_counts", "active", "delta_sv_norm", "orthogonality_error",
    }
    assert required_keys.issubset(set(diag.keys()))


# ---------------------------------------------------------------------------
# 18. find_peaks_multisource returns correct dtype and device
# ---------------------------------------------------------------------------

def test_non_debug_fast_path():
    """find_peaks_multisource returns bool mask and float sources_det on the correct device."""
    N, M = 100, 3
    sources = torch.randn(N, M)
    peak_mask, sources_det = find_peaks_multisource(sources, min_dist=5)
    assert peak_mask.dtype == torch.bool
    assert sources_det.dtype == sources.dtype
    assert peak_mask.device == sources.device
    assert sources_det.device == sources.device


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
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    kappa_cal = torch.zeros(M)

    # Only 1/3 of samples are spikes
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[::3] = True

    _, diag_batch = update_sv_spike_gated(
        sv.clone(), Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=0.0, min_spikes_for_update=1,
        contrast_scope="batch_based",
    )
    _, diag_spike = update_sv_spike_gated(
        sv.clone(), Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=0.0, min_spikes_for_update=1,
        contrast_scope="spike_based",
    )
    # kappa values should differ when the subsets differ
    assert not torch.allclose(diag_batch["kappa"], diag_spike["kappa"], atol=1e-5)


# ---------------------------------------------------------------------------
# 21. sv update requires at least one spike
# ---------------------------------------------------------------------------

def test_sv_update_requires_at_least_one_spike():
    """With all-zero spike_mask, sv is unchanged (up to QR orthonormalization of sv itself)."""
    M, D, N = 3, 10, 50
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    sv_orig = sv.clone()
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    kappa_cal = torch.zeros(M)

    sv_new, _ = update_sv_spike_gated(
        sv, Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=1.0, min_spikes_for_update=1,
        contrast_scope="spike_based",
    )
    # delta_sv = 0, so sv_new = orth(sv_orig) = sv_orig (already orthonormal)
    assert_close(sv_new, sv_orig, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 22. Source FIFO edge spike detection
# ---------------------------------------------------------------------------

def test_source_fifo_edge_spike_not_missed():
    """A spike placed at the last sample of the previous batch is detected in the combined window."""
    N, M = 60, 1
    min_dist = 5

    # prev_sources has a clear spike at its last sample
    prev_sources = torch.zeros(N, M)
    prev_sources[-1, 0] = 10.0  # spike at right edge

    # curr_sources has no spike
    curr_sources = torch.zeros(N, M)

    # Without FIFO: peak detection on curr_sources alone → no spike
    peak_mask_solo, _ = find_peaks_multisource(curr_sources, min_dist=min_dist)
    assert peak_mask_solo.sum() == 0

    # With FIFO: detection on [prev_sources, curr_sources] → spike in prev region
    sources_full = torch.cat([prev_sources, curr_sources], dim=0)
    peak_mask_full, _ = find_peaks_multisource(sources_full, min_dist=min_dist)
    # The spike is in the prev_sources portion (rows 0..N-1) not in curr_sources portion
    # But it would have been missed if we only ran detection on curr_sources
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

    wh = torch.eye(D)
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(300, raw_chs)   # raw (unextended) calibration EMG
    ipts_cal = torch.randn(300, M)
    spikes_cal = torch.zeros(300, M, dtype=torch.int32)
    spikes_cal[::50] = 1

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

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
    from adapt_decomp.data_structures import Decomposition, _extend_data_wh

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

    wh          = torch.eye(D)
    sv          = orthonormalize_rows_qr(torch.randn(M, D))
    emg_cal    = torch.randn(N_cal, raw_chs)
    ipts_cal   = torch.randn(N_cal, M)
    spikes_cal = torch.zeros(N_cal, M, dtype=torch.int32)
    spikes_cal[::40] = 1
    spike_cal  = torch.rand(M) + 2.0
    base_cal   = torch.rand(M) * 0.5

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    # Recompute Rz_cal exactly as init_wh_update does
    X_cal_ext = _extend_data_wh(emg_cal, ext_fact)
    X_cal_ext = X_cal_ext - X_cal_ext.mean(0, keepdim=True)
    Z_cal = X_cal_ext @ wh.T
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
    from adapt_decomp.data_structures import Decomposition, _extend_data_wh

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

    wh          = torch.eye(D)
    sv          = orthonormalize_rows_qr(torch.randn(M, D))
    emg_cal    = torch.randn(N_cal, raw_chs)
    ipts_cal   = torch.randn(N_cal, M)
    spikes_cal = torch.zeros(N_cal, M, dtype=torch.int32)
    spikes_cal[::40] = 1
    spike_cal  = torch.rand(M) + 2.0
    base_cal   = torch.rand(M) * 0.5

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    # Build a clearly drifted Rz (3× variance scale → covariance ~9× larger)
    X_drift = _extend_data_wh(torch.randn(N_cal, raw_chs) * 3.0, ext_fact)
    X_drift = X_drift - X_drift.mean(0, keepdim=True)
    Z_drift = X_drift @ wh.T
    N = Z_drift.shape[0]
    Rz_drift = (Z_drift.T @ Z_drift) / N
    Rz_drift = 0.5 * (Rz_drift + Rz_drift.T)
    Rz_drift = (1 - cfg.shrinkage) * Rz_drift + cfg.shrinkage * torch.eye(D)

    _, logdet_drift = torch.linalg.slogdet(Rz_drift)
    logdet_A = logdet_drift - decomp.logdet_cal
    A = decomp.Rz_cal_inv @ Rz_drift
    K_rel = 0.5 * (A.trace() - logdet_A - D)

    assert K_rel.item() > 0.5, f"Expected large KL for 3× variance drift, got {K_rel.item()}"


# ---------------------------------------------------------------------------
# gate_spikes_by_iqr
# ---------------------------------------------------------------------------

class TestGateSpikesByIqr:
    def _make_inputs(self, N=50, M=3):
        torch.manual_seed(0)
        sources = torch.randn(N, M)
        spike_mask = torch.zeros(N, M, dtype=torch.bool)
        spike_mask[5::10] = True
        Q75_cal = torch.full((M,), 2.0)
        IQR_cal = torch.full((M,), 0.5)
        return sources, spike_mask, Q75_cal, IQR_cal

    def test_output_shape_and_dtype(self):
        sources, spike_mask, Q75_cal, IQR_cal = self._make_inputs()
        out = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
        assert out.shape == spike_mask.shape
        assert out.dtype == torch.bool

    def test_no_outliers_returns_unchanged_mask(self):
        """When no spikes exceed the gate, trusted_mask == spike_mask."""
        N, M = 50, 3
        sources = torch.zeros(N, M)
        spike_mask = torch.zeros(N, M, dtype=torch.bool)
        spike_mask[5] = True
        sources[5] = 1.0  # sources_det = 1.0^2 = 1.0, well below upper_gate = 4.0 + 3*0.5 = 5.5
        Q75_cal = torch.full((M,), 4.0)
        IQR_cal = torch.full((M,), 0.5)
        out = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
        assert torch.equal(out, spike_mask)

    def test_outlier_spike_excluded(self):
        """A spike above the upper fence is excluded from the trusted mask."""
        N, M = 10, 1
        sources = torch.zeros(N, M)
        spike_mask = torch.zeros(N, M, dtype=torch.bool)
        spike_mask[3, 0] = True
        # peak_power=2, use_abs=True: sources_det = |sources|^2
        # upper_gate = 4.0 + 3*0.5 = 5.5; set sources so sources_det = 6.0 > 5.5
        sources[3, 0] = 6.0 ** 0.5
        Q75_cal = torch.tensor([4.0])
        IQR_cal = torch.tensor([0.5])
        out = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
        assert not out[3, 0]
        assert out.sum() == 0

    def test_trusted_mask_is_subset_of_spike_mask(self):
        sources, spike_mask, Q75_cal, IQR_cal = self._make_inputs()
        trusted = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
        assert torch.all(~trusted | spike_mask)  # trusted ⊆ spike_mask

    def test_gate_disabled_returns_full_mask(self):
        """With a very large gate_factor nothing is excluded."""
        sources, spike_mask, Q75_cal, IQR_cal = self._make_inputs()
        trusted = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=1e9)
        assert torch.equal(trusted, spike_mask)

    def test_non_spike_samples_never_added(self):
        """Samples not in spike_mask are never added to the trusted mask."""
        N, M = 20, 2
        sources = torch.zeros(N, M)
        spike_mask = torch.zeros(N, M, dtype=torch.bool)  # no spikes
        Q75_cal = torch.ones(M)
        IQR_cal = torch.ones(M)
        trusted = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
        assert trusted.sum() == 0


# ---------------------------------------------------------------------------
# wh_learning_rate/lr_sv direction-normalized update: proportionality, safety-net-is-rare,
# EMA smoothing, wh_b_coupling consistency, multi-batch stability
# ---------------------------------------------------------------------------

def test_delta_sv_scales_with_error_magnitude():
    """With a loose safety clip, delta_sv_norm scales linearly with |e_b| -- the
    property the lr_sv/direction-normalization reorder restores (the old
    max_rel_delta_sv scheme discarded |e_b| entirely once the clip engaged)."""
    torch.manual_seed(0)
    M, D, N = 1, 8, 60
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.ones(N, M, dtype=torch.bool)  # batch_based ignores spike_mask
    sigma_kappa_cal = torch.ones(M)
    kappa = log_cosh(sources).mean(dim=0)

    def delta_norm_for(e_b_target: torch.Tensor) -> float:
        kappa_cal = kappa - e_b_target
        _, diag = update_sv_spike_gated(
            sv.clone(), Z, sources, kappa_cal, spike_mask=spike_mask,
            max_rel_delta_sv=1e6, min_spikes_for_update=1,   # effectively unclipped
            contrast_scope="batch_based", sigma_kappa_cal=sigma_kappa_cal,
            lr_sv=1e-3,
        )
        return diag["delta_sv_norm"][0].item()

    d1 = delta_norm_for(torch.tensor([1.0]))
    d3 = delta_norm_for(torch.tensor([3.0]))
    assert d3 == pytest.approx(3.0 * d1, rel=1e-4)


def test_safety_clip_engages_only_for_extreme_error():
    """A typical-magnitude e_b produces an (almost) unclipped step; a deliberately
    extreme e_b is capped by the safety net -- confirms the clip is now a rare
    guard rather than the routine, always-on transform it used to be."""
    torch.manual_seed(1)
    M, D, N = 1, 8, 60
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.ones(N, M, dtype=torch.bool)
    sigma_kappa_cal = torch.ones(M)
    kappa = log_cosh(sources).mean(dim=0)
    lr_sv = 1e-3
    safety_ceiling = 20.0 * lr_sv   # matches Config.safety_clip_multiplier_sv default

    def run(e_b_target: torch.Tensor):
        kappa_cal = kappa - e_b_target
        return update_sv_spike_gated(
            sv.clone(), Z, sources, kappa_cal, spike_mask=spike_mask,
            max_rel_delta_sv=safety_ceiling, min_spikes_for_update=1,
            contrast_scope="batch_based", sigma_kappa_cal=sigma_kappa_cal,
            lr_sv=lr_sv,
        )

    _, diag_typical = run(torch.tensor([2.0]))     # well within the ~20-sigma margin
    _, diag_extreme = run(torch.tensor([1000.0]))  # deliberately pathological

    assert diag_typical["delta_sv_norm"][0].item() == pytest.approx(
        diag_typical["delta_sv_raw_norm"][0].item(), rel=1e-4
    )
    # Extreme case: the raw (unclipped) target is far above the ceiling, and the
    # applied step is pinned exactly at the ceiling rather than scaling with it.
    assert diag_extreme["delta_sv_raw_norm"][0].item() > 10 * safety_ceiling
    assert diag_extreme["delta_sv_norm"][0].item() == pytest.approx(safety_ceiling, rel=1e-3)


def test_ema_gradnorm_cold_start_seeds_directly():
    """First call (ema_gradnorm_sv=None) seeds the EMA to exactly the instantaneous
    grad_sv norm -- no blending on the first batch of a fresh trial."""
    torch.manual_seed(2)
    M, D, N = 2, 8, 40
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.ones(N, M, dtype=torch.bool)
    kappa_cal = torch.zeros(M)

    _, diag = update_sv_spike_gated(
        sv, Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=1.0, min_spikes_for_update=1,
        contrast_scope="batch_based", ema_gradnorm_sv=None,
    )
    G = torch.tanh(sources)
    grad_sv = (G.T @ Z) / N
    expected = torch.linalg.norm(grad_sv, dim=1)
    assert_close(diag["ema_gradnorm_sv"], expected, atol=1e-5, rtol=1e-5)


def test_ema_gradnorm_blends_on_subsequent_call():
    """A prior EMA value blends with the new instantaneous norm via ema_alpha."""
    torch.manual_seed(3)
    M, D, N = 2, 8, 40
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.ones(N, M, dtype=torch.bool)
    kappa_cal = torch.zeros(M)
    prior_ema = torch.tensor([5.0, 7.0])
    alpha = 0.8

    _, diag = update_sv_spike_gated(
        sv, Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=1.0, min_spikes_for_update=1,
        contrast_scope="batch_based", ema_gradnorm_sv=prior_ema, ema_alpha=alpha,
    )
    G = torch.tanh(sources)
    grad_sv = (G.T @ Z) / N
    instantaneous = torch.linalg.norm(grad_sv, dim=1)
    expected = alpha * prior_ema + (1 - alpha) * instantaneous
    assert_close(diag["ema_gradnorm_sv"], expected, atol=1e-5, rtol=1e-5)


def test_wh_b_coupling_matches_frame_correction_identity():
    """coupling_matrix must equal -delta_wh @ wh^-1 (the first-order frame correction
    implied by the wh step) under the new wh_learning_rate/direction-normalized formula -- this
    identity previously held because clip_global_delta is a pure scalar rescale of
    delta_V_raw; the reorder requires reintroducing the extra wh_learning_rate/ema_dirnorm_wh
    factor into coupling_matrix's own formula (see adaptation.py::_update_wh)."""
    from adapt_decomp.config import Config
    from adapt_decomp.data_structures import Decomposition
    from adapt_decomp.adaptation import AdaptDecomp

    M, ext_fact, raw_chs = 2, 2, 3
    D = raw_chs * ext_fact
    cfg = Config()
    cfg.device = "cpu"
    cfg.ext_fact = ext_fact
    cfg.adapt_wh = True
    cfg.wh_b_coupling = True
    cfg.debug = False
    cfg.wh_learning_rate = 5e-3
    cfg.__post_init__()

    wh = torch.eye(D) * 1.3   # non-identity, trivially invertible
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(300, raw_chs)
    ipts_cal = torch.randn(300, M)
    spikes_cal = torch.zeros(300, M, dtype=torch.int32)
    spikes_cal[::40] = 1

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    adapter = AdaptDecomp.__new__(AdaptDecomp)
    adapter.config = cfg
    adapter.decomp = decomp
    adapter.units = M
    adapter.diagnostics = {}
    adapter.wh_loss = torch.zeros(1)
    adapter.wh_trace = torch.zeros(1)
    adapter.total_loss = torch.zeros(1)

    # Real (nonzero) signal so Rz is a genuine, positive-definite, drifted
    # covariance -- otherwise e_v ~ 0 and both delta_wh and coupling_matrix would
    # be trivially ~0, which wouldn't exercise the identity meaningfully.
    X = torch.randn(50, D) * 2.0
    wh_before = decomp.whitening.clone()
    _, coupling_matrix = adapter._update_wh(X, batch_idx=0)

    assert coupling_matrix is not None
    delta_wh = decomp.whitening - wh_before
    expected_coupling = -delta_wh @ torch.linalg.inv(wh_before)
    assert_close(coupling_matrix, expected_coupling, atol=1e-4, rtol=1e-3)


def test_multibatch_stability_and_rare_safety_clip():
    """Run AdaptDecomp over many synthetic batches with the new lr-based update:
    no NaN/Inf, sv stays orthonormal, wh stays finite/invertible, and the safety
    clip engages rarely (not on ~100% of batches like the old max_rel_delta
    scheme, verified empirically on real data before this change)."""
    from adapt_decomp.config import Config
    from adapt_decomp.adaptation import AdaptDecomp

    torch.manual_seed(42)
    raw_chs, ext_fact, M = 3, 2, 2
    D = raw_chs * ext_fact
    fs = 200

    cfg = Config()
    cfg.device = "cpu"
    cfg.fs = fs
    cfg.ext_fact = ext_fact
    cfg.batch_ms = 100
    cfg.adapt_wh = True
    cfg.adapt_sv = True
    cfg.adapt_sd = True
    cfg.debug = True
    cfg.wh_learning_rate = 5e-3
    cfg.sv_learning_rate = 1e-3
    cfg.__post_init__()

    wh = torch.eye(D)
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    base_centroids = torch.rand(M) * 0.5
    spike_centroids = torch.rand(M) + 2.0
    emg_calib = torch.randn(500, raw_chs)
    ipts_calib = torch.randn(500, M)
    spikes_calib = torch.zeros(500, M, dtype=torch.int32)
    spikes_calib[::20] = 1

    emg_online = torch.randn(600, raw_chs)

    adapter = AdaptDecomp(
        emg=emg_online,
        whitening=wh,
        sep_vectors=sv,
        base_centr=base_centroids,
        spikes_centr=spike_centroids,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        preprocess=False,
        config=cfg,
    )
    outputs = adapter.run()

    assert torch.isfinite(adapter.decomp.whitening).all()
    assert torch.isfinite(adapter.decomp.sep_vectors).all()
    gram = adapter.decomp.sep_vectors @ adapter.decomp.sep_vectors.T
    assert_close(gram, torch.eye(M), atol=1e-3, rtol=0)
    assert torch.isfinite(torch.linalg.inv(adapter.decomp.whitening)).all()

    diagnostics = outputs["diagnostics"]

    def clip_fraction(raw_key: str, applied_key: str) -> float:
        # Values are scalars for wh (delta_wh_*) but per-unit [M] tensors for sv
        # (delta_sv_*) -- normalize both to flat 1-D tensors before stacking.
        raw_vals = [d[raw_key] for d in diagnostics.values() if raw_key in d]
        applied_vals = [d[applied_key] for d in diagnostics.values() if applied_key in d]
        assert len(raw_vals) > 5, "expected multiple valid batches for this check"
        raw = torch.stack([torch.as_tensor(v).flatten() for v in raw_vals]).flatten()
        applied = torch.stack([torch.as_tensor(v).flatten() for v in applied_vals]).flatten()
        ratio = applied / raw.clamp_min(1e-12)
        return (ratio < 0.99).float().mean().item()

    # Both should engage the safety clip rarely, not on ~100% of batches.
    assert clip_fraction("delta_wh_raw_norm", "delta_wh_norm") < 0.5
    assert clip_fraction("delta_sv_raw_norm", "delta_sv_norm") < 0.5


# ---------------------------------------------------------------------------
# lr_alone ablation: drops the signed e_v/e_b factor entirely, leaving a
# constant direction-normalized step -- the closest available reproduction of
# main (v1)'s fixed-learning-rate update (wh's sign is unaffected; sv's flips
# from an error-correcting descent to an unconditional ascent).
# ---------------------------------------------------------------------------

def test_lr_alone_ignores_error_magnitude_sv():
    """With lr_alone=True, delta_sv_norm is identical regardless of e_b's magnitude --
    the defining property of a genuine fixed learning rate (main's v1 update had no
    error term at all). Contrast with test_delta_B_scales_with_error_magnitude, which
    shows the default (error-weighted) branch scales linearly with |e_b|."""
    torch.manual_seed(4)
    M, D, N = 1, 8, 60
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.ones(N, M, dtype=torch.bool)
    sigma_kappa_cal = torch.ones(M)
    kappa = log_cosh(sources).mean(dim=0)

    def delta_norm_for(e_b_target: torch.Tensor) -> float:
        kappa_cal = kappa - e_b_target
        _, diag = update_sv_spike_gated(
            sv.clone(), Z, sources, kappa_cal, spike_mask=spike_mask,
            max_rel_delta_sv=1e6, min_spikes_for_update=1,   # effectively unclipped
            contrast_scope="batch_based", sigma_kappa_cal=sigma_kappa_cal,
            lr_sv=1e-3, lr_alone=True,
        )
        return diag["delta_sv_norm"][0].item()

    d1 = delta_norm_for(torch.tensor([1.0]))
    d3 = delta_norm_for(torch.tensor([3.0]))
    assert d1 == pytest.approx(d3, rel=1e-5)


def test_lr_alone_is_natural_gradient_ascent_for_sv():
    """lr_alone's delta_sv_target must equal +lr_sv*sv_row_norm*grad_sv/ema (an ascent),
    not -lr_sv*...*grad_sv (a descent) -- the sign flip that reproduces main (v1)'s
    unconditional contrast-maximizing update instead of an error-correcting descent."""
    torch.manual_seed(5)
    M, D, N = 2, 8, 50
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    Z = torch.randn(N, D)
    sources = Z @ sv.T
    spike_mask = torch.ones(N, M, dtype=torch.bool)
    kappa_cal = torch.zeros(M)
    lr_sv = 2e-3

    _, diag = update_sv_spike_gated(
        sv, Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=1e6, min_spikes_for_update=1,
        contrast_scope="batch_based", lr_sv=lr_sv, lr_alone=True,
    )
    G = torch.tanh(sources)
    grad_sv = (G.T @ Z) / N
    ema_gradnorm_sv = torch.linalg.norm(grad_sv, dim=1)   # cold-start seed (ema_gradnorm_sv=None)
    sv_row_norm = torch.linalg.norm(sv, dim=1, keepdim=True)
    expected_delta = lr_sv * sv_row_norm * grad_sv / ema_gradnorm_sv[:, None]
    assert_close(diag["delta_sv_norm"], torch.linalg.norm(expected_delta, dim=1), atol=1e-5, rtol=1e-4)
    # Sign check: applying sv + delta_sv moves each row TOWARD grad_sv (ascent), not away.
    sv_new = sv + expected_delta
    assert ((sv_new * grad_sv).sum(dim=1) > (sv * grad_sv).sum(dim=1)).all()


def test_lr_alone_ignores_error_magnitude_wh():
    """With cfg.lr_alone=True, delta_wh is identical regardless of the calibration
    reference K_cal (which drives e_v's magnitude under the default branch) -- same
    fixed-learning-rate property as the sv-side test above, applied to whitening."""
    from adapt_decomp.config import Config
    from adapt_decomp.data_structures import Decomposition
    from adapt_decomp.adaptation import AdaptDecomp

    M, ext_fact, raw_chs = 2, 2, 3
    D = raw_chs * ext_fact

    def run_with_K_cal(k_cal_value: float) -> torch.Tensor:
        torch.manual_seed(6)   # identical wh/sv/calib/X across calls; only K_cal differs
        cfg = Config()
        cfg.device = "cpu"
        cfg.ext_fact = ext_fact
        cfg.adapt_wh = True
        cfg.lr_alone = True
        cfg.wh_learning_rate = 5e-3
        cfg.__post_init__()

        wh = torch.eye(D) * 1.3
        sv = orthonormalize_rows_qr(torch.randn(M, D))
        spike_cal = torch.rand(M) + 2.0
        base_cal = torch.rand(M) * 0.5
        emg_cal = torch.randn(300, raw_chs)
        ipts_cal = torch.randn(300, M)
        spikes_cal = torch.zeros(300, M, dtype=torch.int32)
        spikes_cal[::40] = 1
        decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)
        decomp.kl_div_calib_mean = torch.tensor(k_cal_value)   # only knob that changes e_v's magnitude

        adapter = AdaptDecomp.__new__(AdaptDecomp)
        adapter.config = cfg
        adapter.decomp = decomp
        adapter.units = M
        adapter.diagnostics = {}
        adapter.wh_loss = torch.zeros(1)
        adapter.wh_trace = torch.zeros(1)
        adapter.total_loss = torch.zeros(1)

        X = torch.randn(50, D) * 2.0
        wh_before = decomp.whitening.clone()
        adapter._update_wh(X, batch_idx=0)
        return decomp.whitening - wh_before

    delta_v_1 = run_with_K_cal(0.0)
    delta_v_2 = run_with_K_cal(50.0)   # would drive e_v far from the first run's value
    assert_close(delta_v_1, delta_v_2, atol=1e-6, rtol=1e-5)


def test_wh_b_coupling_matches_frame_correction_identity_lr_alone():
    """Same identity as test_wh_b_coupling_matches_frame_correction_identity, but
    under cfg.lr_alone=True -- confirms `weight` was substituted symmetrically into
    both delta_wh_target and coupling_matrix's formula, not just one of them."""
    from adapt_decomp.config import Config
    from adapt_decomp.data_structures import Decomposition
    from adapt_decomp.adaptation import AdaptDecomp

    M, ext_fact, raw_chs = 2, 2, 3
    D = raw_chs * ext_fact
    cfg = Config()
    cfg.device = "cpu"
    cfg.ext_fact = ext_fact
    cfg.adapt_wh = True
    cfg.wh_b_coupling = True
    cfg.debug = False
    cfg.lr_alone = True
    cfg.wh_learning_rate = 5e-3
    cfg.__post_init__()

    wh = torch.eye(D) * 1.3
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5
    emg_cal = torch.randn(300, raw_chs)
    ipts_cal = torch.randn(300, M)
    spikes_cal = torch.zeros(300, M, dtype=torch.int32)
    spikes_cal[::40] = 1

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    adapter = AdaptDecomp.__new__(AdaptDecomp)
    adapter.config = cfg
    adapter.decomp = decomp
    adapter.units = M
    adapter.diagnostics = {}
    adapter.wh_loss = torch.zeros(1)
    adapter.wh_trace = torch.zeros(1)
    adapter.total_loss = torch.zeros(1)

    X = torch.randn(50, D) * 2.0
    wh_before = decomp.whitening.clone()
    _, coupling_matrix = adapter._update_wh(X, batch_idx=0)

    assert coupling_matrix is not None
    delta_wh = decomp.whitening - wh_before
    expected_coupling = -delta_wh @ torch.linalg.inv(wh_before)
    assert_close(coupling_matrix, expected_coupling, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 27. ext_mode: "block" vs "toeplitz"
# ---------------------------------------------------------------------------

def test_extend_data_wh_toeplitz_matches_manual_construction():
    """toeplitz mode must place each channel's own delays in contiguous columns,
    forming a per-channel Toeplitz (constant-diagonal) block -- i.e. column
    c*ext_fact+i holds channel c delayed by i samples."""
    from adapt_decomp.data_structures import _extend_data_wh

    samples, chs, ext_fact = 20, 3, 4
    data = torch.randn(samples, chs)

    out = _extend_data_wh(data, ext_fact, ext_mode="toeplitz")
    assert out.shape == (samples, chs * ext_fact)

    expected = torch.zeros_like(out)
    for c in range(chs):
        for i in range(ext_fact):
            col = c * ext_fact + i
            expected[i:, col] = data[: samples - i, c]
    assert_close(out, expected)


def test_extend_data_wh_toeplitz_is_permutation_of_block():
    """block and toeplitz modes must carry the same information, just reordered
    (delay-major vs channel-major) -- neither mode invents or drops content."""
    from adapt_decomp.data_structures import _extend_data_wh

    samples, chs, ext_fact = 15, 2, 3
    data = torch.randn(samples, chs)

    block = _extend_data_wh(data, ext_fact, ext_mode="block")
    toeplitz = _extend_data_wh(data, ext_fact, ext_mode="toeplitz")

    reordered = (
        block.view(samples, ext_fact, chs).permute(0, 2, 1).reshape(samples, chs * ext_fact)
    )
    assert_close(toeplitz, reordered)


def test_extend_data_wh_unknown_mode_raises():
    from adapt_decomp.data_structures import _extend_data_wh

    with pytest.raises(ValueError):
        _extend_data_wh(torch.randn(10, 2), 2, ext_mode="bogus")


def test_extend_emg_toeplitz_matches_extend_data_wh():
    """cbss.whitening.extend_emg (calibration) and data_structures._extend_data_wh
    (online adaptation) must agree on both modes, so a Config.ext_mode that
    matches CBSSConfig.ext_mode really does keep them in sync."""
    from adapt_decomp.cbss.whitening import extend_emg
    from adapt_decomp.data_structures import _extend_data_wh

    samples, chs, ext_fact = 12, 3, 3
    data = torch.randn(samples, chs)

    for mode in ("block", "toeplitz"):
        a = _extend_data_wh(data, ext_fact, ext_mode=mode)
        b = extend_emg(data, ext_fact, ext_mode=mode)
        assert_close(a, b)


def test_decomposition_uses_configured_ext_mode():
    """Decomposition.init_wh_update must extend emg_calib with Config.ext_mode,
    not silently fall back to 'block' -- verified via the FIFO buffer it seeds."""
    from adapt_decomp.config import Config
    from adapt_decomp.data_structures import Decomposition, _extend_data_wh

    ext_fact, raw_chs, M, N_cal = 2, 3, 2, 300

    cfg = Config()
    cfg.device = "cpu"
    cfg.ext_fact = ext_fact
    cfg.ext_mode = "toeplitz"
    cfg.__post_init__()

    D = raw_chs * ext_fact
    wh = torch.eye(D)
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    emg_cal = torch.randn(N_cal, raw_chs)
    ipts_cal = torch.randn(N_cal, M)
    spikes_cal = torch.zeros(N_cal, M, dtype=torch.int32)
    spikes_cal[::40] = 1
    spike_cal = torch.rand(M) + 2.0
    base_cal = torch.rand(M) * 0.5

    decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)

    expected = _extend_data_wh(emg_cal, ext_fact, ext_mode="toeplitz")
    expected = expected - expected.mean(0, keepdim=True)
    assert_close(decomp.fifo_cov, expected[-decomp.fifo_samples:])
