"""Pure-tensor unit tests for adaptation/ops.py's per-batch update primitives.

No torch device/IO/Decomposition setup needed -- these are the cheapest,
fastest tests in the suite and the ones to run on every change.
"""

import torch
import pytest
from torch.testing import assert_close

from adapt_decomp.cbss.ica import log_cosh
from adapt_decomp.adaptation.ops import (
    clip_global_delta,
    clip_rowwise_delta,
    orthonormalize_rows_qr,
    orthonormalize_rows_gram_schmidt,
    classify_peaks_from_adaptive_centroids,
    update_centroids_from_peaks,
    update_sv_spike_gated,
    gate_spikes_by_iqr,
)


# ---------------------------------------------------------------------------
# clip_global_delta
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
# clip_rowwise_delta
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
# classify_peaks_from_adaptive_centroids
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
# update_centroids_from_peaks — update and skip logic
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
# update_sv_spike_gated
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
        max_rel_delta_sv=1.0,
        contrast_scope="spike_based",
    )
    # active should all be False → grad_sv = 0 → delta_sv = 0
    assert torch.all(~diag["active"])
    assert_close(diag["delta_sv_norm"], torch.zeros(M), atol=1e-7, rtol=0)


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
        max_rel_delta_sv=1.0,
        contrast_scope="spike_based",
    )
    # Sources 1 and 3: inactive → zero delta
    assert diag["delta_sv_norm"][1].item() == pytest.approx(0.0, abs=1e-7)
    assert diag["delta_sv_norm"][3].item() == pytest.approx(0.0, abs=1e-7)


def test_sv_orthonormal_after_qr():
    """sv @ sv.T ≈ I after orthonormalize_rows_qr."""
    M, D = 5, 20
    sv = torch.randn(M, D)
    sv_orth = orthonormalize_rows_qr(sv)
    gram = sv_orth @ sv_orth.T
    assert_close(gram, torch.eye(M), atol=1e-5, rtol=0)


def test_sv_orthonormal_after_gram_schmidt():
    """sv @ sv.T ≈ I after orthonormalize_rows_gram_schmidt."""
    M, D = 5, 20
    sv = torch.randn(M, D)
    sv_orth = orthonormalize_rows_gram_schmidt(sv)
    gram = sv_orth @ sv_orth.T
    assert_close(gram, torch.eye(M), atol=1e-5, rtol=0)


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
        max_rel_delta_sv=1.0,
    )
    required_keys = {
        "kappa", "contrast_error",
        "spike_counts", "active", "delta_sv_norm", "orthogonality_error",
    }
    assert required_keys.issubset(set(diag.keys()))


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
        max_rel_delta_sv=0.0,
        contrast_scope="batch_based",
    )
    _, diag_spike = update_sv_spike_gated(
        sv.clone(), Z, sources, kappa_cal, spike_mask=spike_mask,
        max_rel_delta_sv=0.0,
        contrast_scope="spike_based",
    )
    # kappa values should differ when the subsets differ
    assert not torch.allclose(diag_batch["kappa"], diag_spike["kappa"], atol=1e-5)


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
        max_rel_delta_sv=1.0,
        contrast_scope="spike_based",
    )
    # delta_sv = 0, so sv_new = orth(sv_orig) = sv_orig (already orthonormal)
    assert_close(sv_new, sv_orig, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# wh_learning_rate/lr_sv direction-normalized update: proportionality,
# safety-net-is-rare, EMA smoothing
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
            max_rel_delta_sv=1e6,   # effectively unclipped
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
    safety_ceiling = 20.0 * lr_sv   # matches AdaptConfig.safety_clip_multiplier_sv default

    def run(e_b_target: torch.Tensor):
        kappa_cal = kappa - e_b_target
        return update_sv_spike_gated(
            sv.clone(), Z, sources, kappa_cal, spike_mask=spike_mask,
            max_rel_delta_sv=safety_ceiling,
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
        max_rel_delta_sv=1.0,
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
        max_rel_delta_sv=1.0,
        contrast_scope="batch_based", ema_gradnorm_sv=prior_ema, ema_alpha=alpha,
    )
    G = torch.tanh(sources)
    grad_sv = (G.T @ Z) / N
    instantaneous = torch.linalg.norm(grad_sv, dim=1)
    expected = alpha * prior_ema + (1 - alpha) * instantaneous
    assert_close(diag["ema_gradnorm_sv"], expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# lr_alone ablation: drops the signed e_v/e_b factor entirely, leaving a
# constant direction-normalized step -- the closest available reproduction of
# main (v1)'s fixed-learning-rate update (sv's sign flips from an
# error-correcting descent to an unconditional ascent).
# ---------------------------------------------------------------------------

def test_lr_alone_ignores_error_magnitude_sv():
    """With lr_alone=True, delta_sv_norm is identical regardless of e_b's magnitude --
    the defining property of a genuine fixed learning rate (main's v1 update had no
    error term at all). Contrast with test_delta_sv_scales_with_error_magnitude, which
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
            max_rel_delta_sv=1e6,   # effectively unclipped
            contrast_scope="batch_based", sigma_kappa_cal=sigma_kappa_cal,
            lr_sv=1e-3, lr_mode="fixed",
        )
        return diag["delta_sv_norm"][0].item()

    d1 = delta_norm_for(torch.tensor([1.0]))
    d3 = delta_norm_for(torch.tensor([3.0]))
    assert d1 == pytest.approx(d3, rel=1e-5)


def test_lr_alone_is_natural_gradient_ascent_for_sv():
    """lr_mode="fixed"'s delta_sv_target must equal +lr_sv*grad_sv -- the raw,
    un-normalized natural-gradient ascent (no sv_row_norm/ema_gradnorm_sv
    factor; those only apply in the default "rel_error" branch, see
    ops.py::update_sv_spike_gated) -- not -lr_sv*...*grad_sv (a descent), the
    sign flip that reproduces main (v1)'s unconditional contrast-maximizing
    update instead of an error-correcting descent."""
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
        max_rel_delta_sv=1e6,
        contrast_scope="batch_based", lr_sv=lr_sv, lr_mode="fixed",
    )
    G = torch.tanh(sources)
    grad_sv = (G.T @ Z) / N
    expected_delta = lr_sv * grad_sv
    assert_close(diag["delta_sv_norm"], torch.linalg.norm(expected_delta, dim=1), atol=1e-5, rtol=1e-4)
    # Sign check: applying sv + delta_sv moves each row TOWARD grad_sv (ascent), not away.
    sv_new = sv + expected_delta
    assert ((sv_new * grad_sv).sum(dim=1) > (sv * grad_sv).sum(dim=1)).all()


# ---------------------------------------------------------------------------
# gate_spikes_by_iqr
# ---------------------------------------------------------------------------

def _make_iqr_gate_inputs(N=50, M=3):
    torch.manual_seed(0)
    sources = torch.randn(N, M)
    spike_mask = torch.zeros(N, M, dtype=torch.bool)
    spike_mask[5::10] = True
    Q75_cal = torch.full((M,), 2.0)
    IQR_cal = torch.full((M,), 0.5)
    return sources, spike_mask, Q75_cal, IQR_cal


def test_gate_spikes_by_iqr_output_shape_and_dtype():
    sources, spike_mask, Q75_cal, IQR_cal = _make_iqr_gate_inputs()
    out = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
    assert out.shape == spike_mask.shape
    assert out.dtype == torch.bool


def test_gate_spikes_by_iqr_no_outliers_returns_unchanged_mask():
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


def test_gate_spikes_by_iqr_outlier_spike_excluded():
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


def test_gate_spikes_by_iqr_trusted_mask_is_subset_of_spike_mask():
    sources, spike_mask, Q75_cal, IQR_cal = _make_iqr_gate_inputs()
    trusted = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
    assert torch.all(~trusted | spike_mask)  # trusted ⊆ spike_mask


def test_gate_spikes_by_iqr_gate_disabled_returns_full_mask():
    """With a very large gate_factor nothing is excluded."""
    sources, spike_mask, Q75_cal, IQR_cal = _make_iqr_gate_inputs()
    trusted = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=1e9)
    assert torch.equal(trusted, spike_mask)


def test_gate_spikes_by_iqr_non_spike_samples_never_added():
    """Samples not in spike_mask are never added to the trusted mask."""
    N, M = 20, 2
    sources = torch.zeros(N, M)
    spike_mask = torch.zeros(N, M, dtype=torch.bool)  # no spikes
    Q75_cal = torch.ones(M)
    IQR_cal = torch.ones(M)
    trusted = gate_spikes_by_iqr(sources, spike_mask, Q75_cal, IQR_cal, gate_factor=3.0)
    assert trusted.sum() == 0
