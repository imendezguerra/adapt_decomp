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
        whitening=wh,
        sep_vectors=sv,
        base_centr=base_centroids,
        spikes_centr=spike_centroids,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        adapt_config=cfg,
    )
    outputs = adapter.process_data(emg_online, preprocess=False)

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
# _compute_losses: guarded per-run wh_loss_total/sv_loss_total/total_loss
# ---------------------------------------------------------------------------

def test_compute_losses_sums_wh_and_sv_losses(make_decomposition, make_adapter):
    """Normal (non-diverged) case: total_loss == wh_loss_total + sv_loss_total,
    wh_loss_total the sum of the per-batch wh_loss tensor and sv_loss_total the
    sum of sv_loss summed across units per batch, then across batches."""
    decomp, cfg = make_decomposition(M=2, ext_fact=2, raw_chs=3)
    adapter = make_adapter(decomp, cfg)

    adapter.wh_loss = torch.tensor([0.1, 0.3, 0.2])
    adapter.sv_loss = torch.tensor([[0.4, 0.6], [0.5, float("nan")], [0.2, 0.8]])
    adapter.wh_trace = decomp.trace_cal.expand(3).clone()  # ratio == 1, well within the guard

    wh_loss_total, sv_loss_total, total_loss = adapter._compute_losses()

    assert_close(wh_loss_total, torch.tensor(0.6))
    assert_close(sv_loss_total, adapter.sv_loss.nansum(dim=1).nansum())
    assert_close(total_loss, wh_loss_total + sv_loss_total)


def test_compute_losses_guards_against_nan_and_divergence(make_decomposition, make_adapter):
    """NaN anywhere in wh_loss (or a wh_trace/trace_cal ratio outside (0.1, 50))
    forces all three losses to the 1e10 divergence sentinel."""
    decomp, cfg = make_decomposition(M=2, ext_fact=2, raw_chs=3)
    adapter = make_adapter(decomp, cfg)

    adapter.wh_loss = torch.tensor([0.1, float("nan")])
    adapter.sv_loss = torch.zeros(2, 2)
    adapter.wh_trace = decomp.trace_cal.expand(2).clone()

    wh_loss_total, sv_loss_total, total_loss = adapter._compute_losses()

    for value in (wh_loss_total, sv_loss_total, total_loss):
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


# ---------------------------------------------------------------------------
# Construction: process_data(emg, ...) is the only place emg ever enters
# ---------------------------------------------------------------------------

def _make_construction_kwargs():
    """Small, valid AdaptDecomp.__init__ kwargs (no emg) plus emg separately --
    mirrors test_multibatch_stability_and_rare_safety_clip's own setup, just
    without emg baked into the kwargs, for testing construction paths
    independently of the full calibration machinery."""
    from adapt_decomp.adaptation.config import AdaptConfig

    raw_chs, ext_fact, M = 3, 2, 2
    D = raw_chs * ext_fact
    cfg = AdaptConfig()
    cfg.device = "cpu"
    cfg.ext_fact = ext_fact
    cfg.batch_ms = 100
    cfg.fs = 200
    cfg.__post_init__()

    wh = torch.eye(D)
    sv = orthonormalize_rows_qr(torch.randn(M, D))
    base_centroids = torch.rand(M) * 0.5
    spike_centroids = torch.rand(M) + 2.0
    emg_calib = torch.randn(500, raw_chs)
    ipts_calib = torch.randn(500, M)
    spikes_calib = torch.zeros(500, M, dtype=torch.int32)
    spikes_calib[::20] = 1
    emg_online = torch.randn(300, raw_chs)

    kwargs = dict(
        whitening=wh, sep_vectors=sv, base_centr=base_centroids,
        spikes_centr=spike_centroids, emg_calib=emg_calib,
        ipts_calib=ipts_calib, spikes_calib=spikes_calib, adapt_config=cfg,
    )
    return kwargs, emg_online


def test_init_without_emg_does_not_warn():
    """Normal construction (no emg) must not warn -- it's the recommended
    pattern, not the deprecated v1-compatible one."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, _ = _make_construction_kwargs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = AdaptDecomp(**kwargs)
    assert not any(issubclass(w.category, FutureWarning) for w in caught)
    assert adapter._emg_raw is None


def test_init_with_emg_warns_future_warning_and_stores_it():
    """Passing emg to __init__ (the deprecated v1-compatible pattern) must
    warn FutureWarning and store it on self._emg_raw, without building
    self.data (process_data()/init_data() are never called here -- only
    .run() consumes it)."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, emg_online = _make_construction_kwargs()
    with pytest.warns(FutureWarning, match="deprecated"):
        adapter = AdaptDecomp(emg=emg_online, **kwargs)
    assert_close(adapter._emg_raw, emg_online)
    assert not hasattr(adapter, "data")


def test_run_without_emg_raises_clear_error():
    """.run() on an instance built without emg must raise a clear
    ValueError, not AttributeError."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, _ = _make_construction_kwargs()
    adapter = AdaptDecomp(**kwargs)
    with pytest.raises(ValueError, match="emg"):
        adapter.run()


def test_run_warns_and_matches_process_data_offline():
    """.run() must warn FutureWarning and return the same result as calling
    process_data(emg, processing_mode="offline") directly, using the emg
    passed to __init__."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, emg_online = _make_construction_kwargs()
    # fs=200 needs valid filter bounds since .run() always preprocesses
    # (no way to pass preprocess=False through .run()).
    kwargs["adapt_config"].highcut = 80.0
    kwargs["adapt_config"].powerline = False

    torch.manual_seed(3)
    via_run = AdaptDecomp(emg=emg_online, **kwargs)
    with pytest.warns(FutureWarning, match="deprecated"):
        out_run = via_run.run()

    torch.manual_seed(3)
    via_process_data = AdaptDecomp(**kwargs)
    out_direct = via_process_data.process_data(emg_online, processing_mode="offline")

    assert_close(out_run.spikes, out_direct.spikes)
    assert_close(out_run.ipts, out_direct.ipts)


def test_process_data_requires_emg():
    """process_data() with no emg must raise TypeError, not run over stale
    or absent state."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, _ = _make_construction_kwargs()
    adapter = AdaptDecomp(**kwargs)
    with pytest.raises(TypeError):
        adapter.process_data()


def test_process_data_matches_separate_init_data_call():
    """process_data(emg, preprocess=...) must produce the same output as
    calling .init_data(emg, preprocess) and .process_data(emg, preprocess)
    separately, for the same inputs."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, emg_online = _make_construction_kwargs()

    torch.manual_seed(1)
    adapter_a = AdaptDecomp(**kwargs)
    out_a = adapter_a.process_data(emg_online, preprocess=False)

    torch.manual_seed(1)
    adapter_b = AdaptDecomp(**kwargs)
    adapter_b.init_data(emg_online, preprocess=False)
    out_b = adapter_b.process_data(emg_online, preprocess=False)

    assert_close(out_a.spikes, out_b.spikes)
    assert_close(out_a.ipts, out_b.ipts)


def test_process_data_offline_mode_builds_data_online_mode_builds_raw_data():
    """process_data(emg, processing_mode=...) must dispatch to Data
    (processing_mode="offline", the default) or RawData ("online"), and set
    data_preprocessed to match."""
    from adapt_decomp.adaptation import AdaptDecomp
    from adapt_decomp.adaptation.data_structures import Data, RawData

    kwargs, emg_online = _make_construction_kwargs()
    # fs=200 needs valid filter bounds for the online branch below, which
    # always filters (no preprocess=False escape hatch); see
    # test_process_data_streaming_end_to_end_matches_eager_shape's own note.
    kwargs["adapt_config"].highcut = 80.0
    kwargs["adapt_config"].powerline = False

    offline = AdaptDecomp(**kwargs)
    offline.process_data(emg_online, preprocess=False)
    assert isinstance(offline.data, Data)
    assert offline.data_preprocessed is True
    assert offline.processing_mode == "offline"

    online = AdaptDecomp(**kwargs)
    online.process_data(emg_online, preprocess=False, processing_mode="online")
    assert isinstance(online.data, RawData)
    assert online.data_preprocessed is False
    assert online.processing_mode == "online"


def test_process_batch_callable_directly_after_plain_construction():
    """Full online mode: process_batch() must be callable right after plain
    construction, with no process_data()/init_data() call and no manual
    pre-seeding of the per-batch accumulators."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, _ = _make_construction_kwargs()
    kwargs["adapt_config"].highcut = 80.0   # valid for fs=200; process_batch always filters
    kwargs["adapt_config"].powerline = False
    D = kwargs["adapt_config"].ext_fact * 3   # raw_chs=3 (see _make_construction_kwargs)

    adapter = AdaptDecomp(**kwargs)
    adapter.data_preprocessed = False   # full online mode

    spikes, ipts = adapter.process_batch(torch.randn(20, 3))
    assert spikes.shape == (20, kwargs["sep_vectors"].shape[0])
    assert ipts.shape == (20, kwargs["sep_vectors"].shape[0])


# ---------------------------------------------------------------------------
# Streaming mode (data_preprocessed=False): _center_and_extend_batch,
# _preprocess_batch_raw, process_batch's uniform output, and end-to-end
# parity with the offline path.
# ---------------------------------------------------------------------------

def test_center_and_extend_batch_matches_extend_data_reference(make_decomposition, make_adapter):
    """Two consecutive _center_and_extend_batch calls must match extend_data
    run once on the centred concatenation of both batches, each sliced to
    its own span."""
    from adapt_decomp.preprocessing import extend_data

    ext_fact, raw_chs, M = 3, 2, 2
    decomp, cfg = make_decomposition(
        M=M, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=200, spike_stride=20,
    )
    adapter = make_adapter(decomp, cfg)

    torch.manual_seed(0)
    batch1 = torch.randn(10, raw_chs)
    batch2 = torch.randn(10, raw_chs)

    ext1, pad1 = adapter._center_and_extend_batch(batch1)
    ext2, pad2 = adapter._center_and_extend_batch(batch2)

    assert pad1 == ext_fact
    assert pad2 == 0
    assert ext1.shape[0] == batch1.shape[0] - ext_fact
    assert ext2.shape[0] == batch2.shape[0]

    mean1 = batch1.mean(0, keepdim=True)
    window1_ext = extend_data(batch1 - mean1, ext_fact, ext_mode=cfg.ext_mode)
    assert_close(ext1, window1_ext[ext_fact:])

    mean2 = cfg.ema_alpha * mean1 + (1 - cfg.ema_alpha) * batch2.mean(0, keepdim=True)
    window2 = torch.cat([batch1[-ext_fact:], batch2], dim=0)
    window2_ext = extend_data(window2 - mean2, ext_fact, ext_mode=cfg.ext_mode)
    assert_close(ext2, window2_ext[ext_fact:])


def test_ema_mean_online_seeds_from_first_batch_then_blends(make_decomposition, make_adapter):
    """decomp.ema_mean_online must seed to the first batch's own mean, then
    blend via config.ema_alpha on later calls."""
    ext_fact, raw_chs, M = 2, 2, 2
    decomp, cfg = make_decomposition(
        M=M, ext_fact=ext_fact, raw_chs=raw_chs, n_cal=200, spike_stride=20,
    )
    adapter = make_adapter(decomp, cfg)

    torch.manual_seed(1)
    batch1 = torch.randn(8, raw_chs)
    batch2 = torch.randn(8, raw_chs)

    adapter._center_and_extend_batch(batch1)
    assert_close(decomp.ema_mean_online, batch1.mean(0, keepdim=True))

    adapter._center_and_extend_batch(batch2)
    expected = cfg.ema_alpha * batch1.mean(0, keepdim=True) + (1 - cfg.ema_alpha) * batch2.mean(0, keepdim=True)
    assert_close(decomp.ema_mean_online, expected)


def _prep_batch_process_adapter(make_decomposition, make_adapter, data_preprocessed: bool):
    """Wire a make_adapter instance with the timing lists process_batch needs
    (make_adapter itself only seeds wh_loss/sv_loss/wh_trace)."""
    decomp, cfg = make_decomposition(M=2, ext_fact=3, raw_chs=2, n_cal=200, spike_stride=20)
    adapter = make_adapter(decomp, cfg)
    adapter.data_preprocessed = data_preprocessed
    adapter.time_wh_ms, adapter.time_sv_ms = [], []
    adapter.time_sd_ms, adapter.time_preprocess_ms = [], []
    return adapter, cfg


def test_process_batch_uniform_output_eager_mode_first_batch_zero_padded(make_decomposition, make_adapter):
    """process_batch's first call in eager mode must return emg_batch.shape[0]
    rows with the leading ext_fact rows zeroed; later calls are fully populated."""
    adapter, cfg = _prep_batch_process_adapter(make_decomposition, make_adapter, data_preprocessed=True)
    D = adapter.decomp.n
    N = 12

    spikes, sources = adapter.process_batch(torch.randn(N, D), batch_idx=0)
    assert spikes.shape == (N, adapter.units)
    assert sources.shape == (N, adapter.units)
    assert torch.all(spikes[:cfg.ext_fact] == 0)
    assert torch.all(sources[:cfg.ext_fact] == 0)
    assert adapter.time_preprocess_ms == [0.0]

    spikes2, _ = adapter.process_batch(torch.randn(N, D), batch_idx=1)
    assert spikes2.shape == (N, adapter.units)
    assert adapter.time_preprocess_ms == [0.0, 0.0]


def test_process_batch_uniform_output_streaming_mode_first_batch_zero_padded(make_decomposition, make_adapter):
    """process_batch's first call in streaming mode must return
    emg_batch.shape[0] rows with the leading rows zeroed, and log a
    non-negative preprocessing time each call; later calls are fully
    populated. (A near-zero synthetic batch can legitimately measure
    0.0ms on a coarse wall clock, so only non-negativity is checked here.)"""
    adapter, cfg = _prep_batch_process_adapter(make_decomposition, make_adapter, data_preprocessed=False)
    raw_chs = 2
    N = 20

    spikes, sources = adapter.process_batch(torch.randn(N, raw_chs), batch_idx=0)
    assert spikes.shape == (N, adapter.units)
    assert sources.shape == (N, adapter.units)
    assert torch.all(spikes[:cfg.ext_fact] == 0)
    assert torch.all(sources[:cfg.ext_fact] == 0)
    assert adapter.time_preprocess_ms[0] >= 0.0

    spikes2, _ = adapter.process_batch(torch.randn(N, raw_chs), batch_idx=1)
    assert spikes2.shape == (N, adapter.units)
    assert adapter.time_preprocess_ms[1] >= 0.0


def test_process_data_streaming_end_to_end_matches_eager_shape():
    """A processing_mode="online" run driven via process_data() must
    complete without error, match an equivalent offline-mode run's output
    shapes, keep preprocess_time_ms at zero throughout the offline run, and
    seed the streaming path's per-batch state (filter zi, EMA mean,
    extension FIFO)."""
    from adapt_decomp.adaptation import AdaptDecomp

    kwargs, emg_online = _make_construction_kwargs()
    # fs=200 (from _make_construction_kwargs) needs valid filter bounds since
    # the streaming path always filters (no preprocess=False escape hatch);
    # disable the notch filter too since its default harmonics (100, 150 Hz)
    # exceed this fs's Nyquist frequency (100 Hz).
    kwargs["adapt_config"].highcut = 80.0
    kwargs["adapt_config"].powerline = False

    torch.manual_seed(2)
    eager = AdaptDecomp(**kwargs)
    out_eager = eager.process_data(emg_online, preprocess=False)

    torch.manual_seed(2)
    streaming = AdaptDecomp(**kwargs)
    out_streaming = streaming.process_data(emg_online, preprocess=False, processing_mode="online")

    assert out_streaming.spikes.shape == out_eager.spikes.shape
    assert out_streaming.ipts.shape == out_eager.ipts.shape

    assert torch.all(out_eager.preprocess_time_ms == 0)
    # A synthetic batch this small can legitimately measure 0.0ms on a
    # coarse wall clock, so preprocess_time_ms itself is only checked for
    # shape/non-negativity here; that streaming mode's preprocessing step
    # actually ran is checked deterministically via its per-batch state below.
    assert torch.all(out_streaming.preprocess_time_ms >= 0)
    assert_close(
        out_streaming.total_time_ms,
        out_streaming.wh_time_ms + out_streaming.sv_time_ms
        + out_streaming.sd_time_ms + out_streaming.preprocess_time_ms,
    )

    # Streaming mode's per-batch state (filter zi, EMA mean, extension FIFO)
    # must have been seeded by the run.
    assert streaming.decomp.zi is not None
    assert streaming.decomp.ema_mean_online is not None
    assert streaming.decomp.ext_fifo is not None

    # adapter.spikes/ipts (read directly off the instance) must match
    # outputs.spikes/ipts exactly, since optimize.py's _run_one_dataset relies on this.
    assert_close(eager.spikes, out_eager.spikes)
    assert_close(eager.ipts, out_eager.ipts)
    assert_close(streaming.spikes, out_streaming.spikes)
    assert_close(streaming.ipts, out_streaming.ipts)
