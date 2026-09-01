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

import numpy as np
import pytest
import torch

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import Decomposition
from adapt_decomp.adaptation.ops import orthonormalize_rows_qr
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult


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
        matters -- several of decomp's derived fields depend on it). The
        returned adapter's wh_loss/sv_loss/wh_trace start as empty lists,
        ready for _whiten()/_update_sep_vectors() to append to (see
        core.py's growable-accumulator convention); for tests exercising
        _compute_losses() directly, overwrite them with tensors first.
    """
    def _make(decomp: Decomposition, config: AdaptConfig):
        from adapt_decomp.adaptation import AdaptDecomp

        adapter = AdaptDecomp.__new__(AdaptDecomp)
        adapter.config = config
        adapter.decomp = decomp
        adapter.units = decomp.sep_vectors.shape[0]
        adapter.diagnostics = {}
        adapter.wh_loss = []
        adapter.sv_loss = []
        adapter.wh_trace = []
        return adapter
    return _make


@pytest.fixture
def make_optimize_kwargs():
    """Factory fixture: tiny synthetic CBSSResult/CBSSConfig for
    adaptation/optimize.py smoke tests -- no real EMG data needed.

    Returns:
        Callable[[], Tuple[Dict, int]]: Call with no arguments; reseeds
        torch.manual_seed(42) on every call, so repeated calls (e.g. building
        two pooled datasets) reproduce identical synthetic data. Returns
        (kwargs, M): kwargs is emg/calibration/cbss_config/preprocess/
        base_config -- exactly the pieces needed to build a
        PooledDatasetMemory (plus base_config, forwarded separately to
        optimize_adapt_decomp_pooled_memory's own base_config parameter);
        M is the number of motor units.
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

        sv = orthonormalize_rows_qr(torch.randn(M, D))
        base_centroids = torch.rand(M) * 0.5
        spike_centroids = torch.rand(M) + 2.0
        emg_calib = torch.randn(500, raw_chs)
        ipts_calib = torch.randn(500, M)
        spikes_calib = torch.zeros(500, M, dtype=torch.int32)
        spikes_calib[::20] = 1
        emg_online = torch.randn(600, raw_chs)

        spikes_calib_np = spikes_calib.numpy()
        calibration = CBSSResult(
            sources=ipts_calib.numpy(),
            spikes=spikes_calib_np,
            spikes_dict={i: np.where(spikes_calib_np[:, i])[0] for i in range(M)},
            sep_vectors=sv.numpy().T,  # CBSSResult stores [dim, n_mu]; to_adapt_tensors() transposes back
            whitening=np.eye(D, dtype=np.float32),
            extension_mean=np.zeros((1, D), dtype=np.float32),
            spikes_centr=spike_centroids.numpy(),
            base_centr=base_centroids.numpy(),
            sil=np.full(M, 0.9, dtype=np.float32),
            cov_isi=np.full(M, 0.1, dtype=np.float32),
            ext_fact=ext_fact,
            emg=emg_calib.numpy(),
        )
        cbss_config = CBSSConfig(ext_fact=ext_fact, fs=fs, save_emg=True)

        return dict(
            emg=emg_online, calibration=calibration, cbss_config=cbss_config,
            preprocess=False, base_config=cfg,
        ), M
    return _make
