"""Shared pytest fixtures for adapt_decomp's test suite.

Factory fixtures (make_adapt_config/make_decomposition/make_adapter/
make_optimize_kwargs) replace the near-identical AdaptConfig/Decomposition/
AdaptDecomp-wiring boilerplate that used to be hand-copied into almost every
test in the old tests/test_backend.py. Each is a plain factory function
returned from a fixture (the standard pytest "factory as fixture" pattern),
so a test can still call it more than once with different arguments (e.g.
test_lr_alone_ignores_error_magnitude_wh needs two independent decompositions
built under the same torch.manual_seed).
"""

from typing import Optional

import pytest
import torch

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import Decomposition
from adapt_decomp.adaptation.ops import orthonormalize_rows_qr


@pytest.fixture
def make_adapt_config():
    """Factory fixture: build an AdaptConfig for tests.

    Returns:
        Callable[..., AdaptConfig]: Call with field-name overrides as kwargs;
        device="cpu" is set first, then overrides, then __post_init__ is
        re-run so derived fields (spike_min_dist, batch_size) stay in sync.
    """
    def _make(**overrides) -> AdaptConfig:
        cfg = AdaptConfig()
        cfg.device = "cpu"
        for key, value in overrides.items():
            setattr(cfg, key, value)
        cfg.__post_init__()
        return cfg
    return _make


@pytest.fixture
def make_decomposition(make_adapt_config):
    """Factory fixture: build a Decomposition over synthetic calibration data.

    Returns:
        Callable[..., Tuple[Decomposition, AdaptConfig]]: Call with
        (M, ext_fact, raw_chs), optionally n_cal (default 500), spike_stride
        (default 50, i.e. spikes_cal[::spike_stride] = 1), whitening (default
        torch.eye(D)), orthonormal_sv (default True; False row-normalises sv
        instead of QR-orthonormalising it), config (an existing AdaptConfig
        to use as-is), or any AdaptConfig field override -- forwarded to
        make_adapt_config(ext_fact=ext_fact, **cfg_overrides) when config is
        not given. Returns (decomposition, the config it was built with) --
        the config is needed by make_adapter, since several of Decomposition's
        derived fields depend on it.
    """
    def _make(
        M: int,
        ext_fact: int,
        raw_chs: int,
        n_cal: int = 500,
        spike_stride: int = 50,
        whitening: Optional[torch.Tensor] = None,
        orthonormal_sv: bool = True,
        config: Optional[AdaptConfig] = None,
        **cfg_overrides,
    ):
        cfg = config if config is not None else make_adapt_config(ext_fact=ext_fact, **cfg_overrides)
        D = raw_chs * ext_fact

        wh = whitening if whitening is not None else torch.eye(D)
        sv = torch.randn(M, D)
        sv = (
            orthonormalize_rows_qr(sv) if orthonormal_sv
            else sv / torch.linalg.norm(sv, dim=1, keepdim=True)
        )
        spike_cal = torch.rand(M) + 2.0
        base_cal = torch.rand(M) * 0.5
        emg_cal = torch.randn(n_cal, raw_chs)
        ipts_cal = torch.randn(n_cal, M)
        spikes_cal = torch.zeros(n_cal, M, dtype=torch.int32)
        spikes_cal[::spike_stride] = 1

        decomp = Decomposition(wh, sv, base_cal, spike_cal, emg_cal, ipts_cal, spikes_cal, cfg)
        return decomp, cfg
    return _make


@pytest.fixture
def make_adapter():
    """Factory fixture: wire a bare AdaptDecomp directly to an existing
    Decomposition, bypassing __init__ -- for tests exercising a single
    internal step (e.g. _whiten) without running a full calibration-from-EMG
    pipeline.

    Returns:
        Callable[[Decomposition, AdaptConfig], "AdaptDecomp"]: Call with the
        decomposition and the AdaptConfig it was built with (matching config
        matters -- several of decomp's derived fields depend on it).
    """
    def _make(decomp: Decomposition, config: AdaptConfig):
        from adapt_decomp.adaptation import AdaptDecomp

        adapter = AdaptDecomp.__new__(AdaptDecomp)
        adapter.config = config
        adapter.decomp = decomp
        adapter.units = decomp.sep_vectors.shape[0]
        adapter.diagnostics = {}
        adapter.wh_loss = torch.zeros(1)
        adapter.wh_trace = torch.zeros(1)
        adapter.total_loss = torch.zeros(1)
        return adapter
    return _make


@pytest.fixture
def make_optimize_kwargs():
    """Factory fixture: tiny synthetic AdaptDecomp inputs for
    adaptation/optimize.py smoke tests -- no real EMG data needed.

    Returns:
        Callable[[], Tuple[Dict, int]]: Call with no arguments; reseeds
        torch.manual_seed(42) on every call, so repeated calls (e.g. building
        two pooled conditions) reproduce identical synthetic data. Returns
        (kwargs, M): kwargs is the dict of AdaptDecomp/optimize_adapt_decomp
        constructor arguments (emg, whitening, sep_vectors, base_centr,
        spikes_centr, emg_calib, ipts_calib, spikes_calib, preprocess,
        base_config), M is the number of motor units.
    """
    def _make():
        torch.manual_seed(42)
        raw_chs, ext_fact, M = 3, 2, 2
        D = raw_chs * ext_fact
        fs = 200

        cfg = AdaptConfig()
        cfg.device = "cpu"
        cfg.fs = fs
        cfg.ext_fact = ext_fact
        cfg.batch_ms = 100
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

        return dict(
            emg=emg_online, whitening=wh, sep_vectors=sv,
            base_centr=base_centroids, spikes_centr=spike_centroids,
            emg_calib=emg_calib, ipts_calib=ipts_calib, spikes_calib=spikes_calib,
            preprocess=False, base_config=cfg,
        ), M
    return _make
