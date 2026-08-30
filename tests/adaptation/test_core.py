"""Tests for adaptation/core.py's AdaptDecomp: the per-batch whitening/
separation/spike-detection update loop.

test_multibatch_stability_and_rare_safety_clip is marked slow -- it runs a
real multi-batch adaptation loop end to end, unlike the other tests here
which exercise a single _whiten() call via the make_adapter fixture.
"""

import warnings

import torch
import pytest
from torch.testing import assert_close

from adapt_decomp.adaptation.ops import orthonormalize_rows_qr


# ---------------------------------------------------------------------------
# Whitening update skips when slogdet sign is invalid
# ---------------------------------------------------------------------------

def test_update_wh_skips_invalid_slogdet(make_decomposition, make_adapter):
    """If Rz has non-positive slogdet, wh is returned unchanged from _update_wh."""
    decomp, cfg = make_decomposition(
        M=2, ext_fact=2, raw_chs=3, n_cal=200, spike_stride=40,
        adapt_wh=True, compute_loss=False, debug=True,
    )
    adapter = make_adapter(decomp, cfg)

    # Force Rz to be singular: zero FIFO + zero shrinkage → Rz = 0 → slogdet sign=0
    decomp.fifo_cov = torch.zeros_like(decomp.fifo_cov)
    decomp.shrinkage = 0.0
    wh_before = decomp.whitening.clone()

    # X must also be zero so _update_fifo_cov doesn't add signal back into the FIFO
    X = torch.zeros(50, decomp.n)
    adapter._whiten(X, batch_idx=0)

    # wh should be unchanged because slogdet was non-positive (Rz = 0 matrix)
    assert_close(decomp.whitening, wh_before)
    assert adapter.diagnostics.get(0, {}).get("wh_skip_invalid_slogdet", False)


# ---------------------------------------------------------------------------
# wh_sv_coupling: coupling_matrix must equal -delta_wh @ wh^-1 (first-order
# frame correction identity implied by the wh step)
# ---------------------------------------------------------------------------

def test_wh_sv_coupling_matches_frame_correction_identity(make_decomposition, make_adapter):
    """coupling_matrix must equal -delta_wh @ wh^-1 (the first-order frame correction
    implied by the wh step) under the lr_learning_rate/direction-normalized formula."""
    ext_fact, raw_chs = 2, 3
    D = raw_chs * ext_fact
    decomp, cfg = make_decomposition(
        M=2, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=300, spike_stride=40,
        whitening=torch.eye(D) * 1.3,   # non-identity, trivially invertible
        adapt_wh=True, wh_sv_coupling=True, debug=False, wh_learning_rate=5e-3,
    )
    adapter = make_adapter(decomp, cfg)

    # Real (nonzero) signal so Rz is a genuine, positive-definite, drifted
    # covariance -- otherwise e_v ~ 0 and both delta_wh and coupling_matrix would
    # be trivially ~0, which wouldn't exercise the identity meaningfully.
    X = torch.randn(50, D) * 2.0
    wh_before = decomp.whitening.clone()
    _, coupling_matrix = adapter._whiten(X, batch_idx=0)

    assert coupling_matrix is not None
    delta_wh = decomp.whitening - wh_before
    expected_coupling = -delta_wh @ torch.linalg.inv(wh_before)
    assert_close(coupling_matrix, expected_coupling, atol=1e-4, rtol=1e-3)


def test_wh_sv_coupling_matches_frame_correction_identity_lr_alone(make_decomposition, make_adapter):
    """Same identity as test_wh_sv_coupling_matches_frame_correction_identity, but
    under cfg.lr_mode="fixed" (lr_alone) -- confirms `weight` was substituted
    symmetrically into both delta_wh_target and coupling_matrix's formula, not
    just one of them."""
    ext_fact, raw_chs = 2, 3
    D = raw_chs * ext_fact
    decomp, cfg = make_decomposition(
        M=2, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=300, spike_stride=40,
        whitening=torch.eye(D) * 1.3,
        adapt_wh=True, wh_sv_coupling=True, debug=False,
        lr_mode="fixed", wh_learning_rate=5e-3,
    )
    adapter = make_adapter(decomp, cfg)

    X = torch.randn(50, D) * 2.0
    wh_before = decomp.whitening.clone()
    _, coupling_matrix = adapter._whiten(X, batch_idx=0)

    assert coupling_matrix is not None
    delta_wh = decomp.whitening - wh_before
    expected_coupling = -delta_wh @ torch.linalg.inv(wh_before)
    assert_close(coupling_matrix, expected_coupling, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# lr_alone ablation, wh side: drops the signed e_v factor entirely
# ---------------------------------------------------------------------------

def test_lr_alone_ignores_error_magnitude_wh(make_decomposition, make_adapter):
    """With cfg.lr_mode="fixed" (lr_alone), delta_wh is identical regardless of the
    calibration reference K_cal (which drives e_v's magnitude under the default
    branch) -- same fixed-learning-rate property as the sv-side
    test_lr_alone_ignores_error_magnitude_sv, applied to whitening."""
    ext_fact, raw_chs = 2, 3
    D = raw_chs * ext_fact

    def run_with_K_cal(k_cal_value: float) -> torch.Tensor:
        torch.manual_seed(6)   # identical wh/sv/calib/X across calls; only K_cal differs
        decomp, cfg = make_decomposition(
            M=2, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=300, spike_stride=40,
            whitening=torch.eye(D) * 1.3,
            adapt_wh=True, lr_mode="fixed", wh_learning_rate=5e-3,
        )
        decomp.kl_div_calib_mean = torch.tensor(k_cal_value)   # only knob that changes e_v's magnitude
        adapter = make_adapter(decomp, cfg)

        X = torch.randn(50, D) * 2.0
        wh_before = decomp.whitening.clone()
        adapter._whiten(X, batch_idx=0)
        return decomp.whitening - wh_before

    delta_v_1 = run_with_K_cal(0.0)
    delta_v_2 = run_with_K_cal(50.0)   # would drive e_v far from the first run's value
    assert_close(delta_v_1, delta_v_2, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Full multi-batch adaptation loop: stability + rare safety clip
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_multibatch_stability_and_rare_safety_clip():
    """Run AdaptDecomp over many synthetic batches with the lr-based update:
    no NaN/Inf, sv stays orthonormal, wh stays finite/invertible, and the safety
    clip engages rarely (not on ~100% of batches like the old max_rel_delta
    scheme, verified empirically on real data before this change)."""
    from adapt_decomp.adaptation.config import AdaptConfig
    from adapt_decomp.adaptation import AdaptDecomp

    torch.manual_seed(42)
    raw_chs, ext_fact, M = 3, 2, 2
    D = raw_chs * ext_fact
    fs = 200

    cfg = AdaptConfig()
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
        adapt_config=cfg,
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
# _compute_losses: guarded per-run wh_loss_median/sv_loss_median/total_loss
# ---------------------------------------------------------------------------

def test_compute_losses_sums_wh_and_sv_medians(make_decomposition, make_adapter):
    """Normal (non-diverged) case: total_loss == wh_loss_median + sv_loss_median,
    each the median/nanmedian of the per-batch wh_loss/sv_loss tensors."""
    decomp, cfg = make_decomposition(M=2, ext_fact=2, raw_chs=3)
    adapter = make_adapter(decomp, cfg)

    adapter.wh_loss = torch.tensor([0.1, 0.3, 0.2])
    adapter.sv_loss = torch.tensor([[0.4, 0.6], [0.5, float("nan")], [0.2, 0.8]])
    adapter.wh_trace = decomp.trace_cal.expand(3).clone()  # ratio == 1, well within the guard

    wh_loss_median, sv_loss_median, total_loss = adapter._compute_losses()

    assert_close(wh_loss_median, torch.tensor(0.2))
    assert_close(sv_loss_median, adapter.sv_loss.nanmedian())
    assert_close(total_loss, wh_loss_median + sv_loss_median)


def test_compute_losses_guards_against_nan_and_divergence(make_decomposition, make_adapter):
    """NaN anywhere in wh_loss (or a wh_trace/trace_cal ratio outside (0.1, 50))
    forces all three losses to the 1e10 divergence sentinel."""
    decomp, cfg = make_decomposition(M=2, ext_fact=2, raw_chs=3)
    adapter = make_adapter(decomp, cfg)

    adapter.wh_loss = torch.tensor([0.1, float("nan")])
    adapter.sv_loss = torch.zeros(2, 2)
    adapter.wh_trace = decomp.trace_cal.expand(2).clone()

    wh_loss_median, sv_loss_median, total_loss = adapter._compute_losses()

    for value in (wh_loss_median, sv_loss_median, total_loss):
        assert value.item() == pytest.approx(1e10)


# ---------------------------------------------------------------------------
# SharedCalibFields / reconcile_with_calib_config
# ---------------------------------------------------------------------------

def test_shared_calib_fields_from_cbss_config_round_trips():
    """from_cbss_config() reads every _SHARED_CBSS_ADAPT_FIELDS entry, and
    nothing else, off a real CBSSConfig."""
    from adapt_decomp.adaptation.core import SharedCalibFields, _SHARED_CBSS_ADAPT_FIELDS
    from adapt_decomp.cbss.config import CBSSConfig

    cbss_config = CBSSConfig(ext_fact=3, spike_det_exp=1.5, ext_mode="toeplitz")
    shared = SharedCalibFields.from_cbss_config(cbss_config)

    for field in _SHARED_CBSS_ADAPT_FIELDS:
        assert getattr(shared, field) == getattr(cbss_config, field)


def test_reconcile_with_calib_config_overwrites_disagreeing_fields_and_warns():
    """shared wins on every disagreeing field, warns exactly once, and never
    mutates the caller's adapt_config in place."""
    from adapt_decomp.adaptation.config import AdaptConfig
    from adapt_decomp.adaptation.core import SharedCalibFields, reconcile_with_calib_config
    from adapt_decomp.cbss.config import CBSSConfig

    shared = SharedCalibFields.from_cbss_config(CBSSConfig(ext_fact=2, spike_det_exp=1.5))
    adapt_config = AdaptConfig(ext_fact=2, spike_det_exp=9.0, device="cpu")

    with pytest.warns(UserWarning, match="spike_det_exp"):
        reconciled = reconcile_with_calib_config(adapt_config, shared)

    assert reconciled.spike_det_exp == 1.5   # shared won
    assert adapt_config.spike_det_exp == 9.0  # caller's instance untouched


def test_reconcile_with_calib_config_agreeing_fields_stay_silent():
    """No warning, and adapt_config's own values pass through, when every
    shared field already agrees."""
    from adapt_decomp.adaptation.config import AdaptConfig
    from adapt_decomp.adaptation.core import SharedCalibFields, reconcile_with_calib_config
    from adapt_decomp.cbss.config import CBSSConfig

    cbss_config = CBSSConfig(ext_fact=2, spike_det_exp=1.5)
    shared = SharedCalibFields.from_cbss_config(cbss_config)
    adapt_config = AdaptConfig(ext_fact=2, spike_det_exp=1.5, device="cpu")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reconciled = reconcile_with_calib_config(adapt_config, shared)

    assert not any("disagreed" in str(w.message) for w in caught)
    assert reconciled.spike_det_exp == 1.5
