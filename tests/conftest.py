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
from adapt_decomp.cbss import CBSS
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult
from tests.synthetic import SyntheticRecording, make_synthetic_recording


@pytest.fixture(scope="session")
def synthetic_recording() -> SyntheticRecording:
    """One drifting synthetic recording, shared by the whole session (read-only)."""
    return make_synthetic_recording()


@pytest.fixture(scope="session")
def synthetic_cbss_config(synthetic_recording) -> CBSSConfig:
    """A small, fast CBSSConfig for synthetic_recording (read-only)."""
    return CBSSConfig(fs=synthetic_recording.fs, ext_fact=8, search_iter=15, random_seed=0)


@pytest.fixture(scope="session")
def synthetic_calibration(synthetic_recording, synthetic_cbss_config) -> CBSSResult:
    """CBSS on synthetic_recording's calibration window, narrowed to its GT-matched units."""
    rec = synthetic_recording
    calibration = CBSS(synthetic_cbss_config).decompose(rec.emg[: rec.n_cal])
    return calibration.select_supervised(rec.spikes[: rec.n_cal], fs=rec.fs, tol_spike_ms=1)


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
        cfg = (
            config if config is not None else make_adapt_config(ext_fact=ext_fact, **cfg_overrides)
        )
        D = raw_chs * ext_fact

        wh = whitening if whitening is not None else torch.eye(D)
        sv = torch.randn(M, D)
        sv = (
            orthonormalize_rows_qr(sv)
            if orthonormal_sv
            else sv / torch.linalg.norm(sv, dim=1, keepdim=True)
        )
        spike_cal = torch.rand(M) + 2.0
        base_cal = torch.rand(M) * 0.5
        emg_cal = torch.randn(n_cal, raw_chs)
        sources_cal = torch.randn(n_cal, M)
        spikes_cal = torch.zeros(n_cal, M, dtype=torch.int32)
        spikes_cal[::spike_stride] = 1

        decomp = Decomposition(
            wh,
            sv,
            base_cal,
            spike_cal,
            emg_cal,
            spikes_cal,
            cfg,
            sources_calib=sources_cal,
        )
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
    adaptation/optimize/ smoke tests -- no real EMG data needed.

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
        sources_calib = torch.randn(500, M)
        spikes_calib = torch.zeros(500, M, dtype=torch.int32)
        spikes_calib[::20] = 1
        emg_online = torch.randn(600, raw_chs)

        spikes_calib_np = spikes_calib.numpy()
        calibration = CBSSResult(
            sources=sources_calib.numpy(),
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
            emg=emg_online,
            calibration=calibration,
            cbss_config=cbss_config,
            preprocess=False,
            base_config=cfg,
        ), M

    return _make


@pytest.fixture
def make_memory_pool(make_optimize_kwargs):
    """Factory fixture: an in-memory search pool of identical synthetic datasets.

    Returns:
        Callable[..., Tuple[Dict[str, PooledDatasetMemory], AdaptConfig]]: Call
        with the dataset names (default ("dataset_a",)), optionally cov_isi (the
        calibration's per-unit CoV-ISI) and gt (True attaches paired ground
        truth). Returns (pool, base_config).
    """
    from adapt_decomp.utils.loaders import PooledDatasetMemory

    def _make(names=("dataset_a",), cov_isi=None, gt=False):
        pool = {}
        for name in names:
            common, n_units = make_optimize_kwargs()
            calibration = common["calibration"]
            if cov_isi is not None:
                calibration.cov_isi = np.asarray(cov_isi, dtype=np.float32)
            gt_paired_bin = None
            if gt:
                gt_paired_bin = np.zeros((common["emg"].shape[0], n_units), dtype=np.float32)
                gt_paired_bin[::30] = 1
            pool[name] = PooledDatasetMemory(
                emg=common["emg"],
                calibration=calibration,
                cbss_config=common["cbss_config"],
                preprocess=common["preprocess"],
                gt_paired_bin=gt_paired_bin,
            )
        return pool, common["base_config"]

    return _make


@pytest.fixture
def make_disk_dataset(tmp_path):
    """Factory fixture: one on-disk search dataset, written to tmp_path.

    Same shapes as make_optimize_kwargs (raw_chs=3, ext_fact=2, M=2, fs=200,
    500 calibration and 600 online samples): a CBSSResult pickle, a CBSSConfig
    YAML and an emg .npz, plus a "<name>_gt.npz" when gt_dense is given.

    Returns:
        Callable[..., PooledDatasetDisk]: Call with the dataset name, optionally
        M, fs, seed, spikes (calibration spike trains, (500, M); default every
        20th sample) and gt_dense (full-recording ground truth, (600, n_gt)).
    """
    from adapt_decomp.utils.loaders import PooledDatasetDisk

    def _make(name, M=2, fs=200, seed=0, spikes=None, gt_dense=None):
        raw_chs, ext_fact, n_cal, n_full = 3, 2, 500, 600
        rng = np.random.default_rng(seed)
        D = raw_chs * ext_fact
        if spikes is None:
            spikes = np.zeros((n_cal, M), dtype=np.int32)
            spikes[::20] = 1
        sv = orthonormalize_rows_qr(
            torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32))
        ).numpy()
        CBSSResult(
            sources=rng.standard_normal((n_cal, M)).astype(np.float32),
            spikes=spikes,
            spikes_dict={i: np.where(spikes[:, i])[0] for i in range(M)},
            sep_vectors=sv.T,  # CBSSResult stores [dim, n_mu]
            whitening=np.eye(D, dtype=np.float32),
            extension_mean=np.zeros((1, D), dtype=np.float32),
            spikes_centr=(rng.random(M) + 2.0).astype(np.float32),
            base_centr=(rng.random(M) * 0.5).astype(np.float32),
            sil=np.full(M, 0.9, dtype=np.float32),
            cov_isi=np.full(M, 0.1, dtype=np.float32),
            ext_fact=ext_fact,
            emg=rng.standard_normal((n_cal, raw_chs)).astype(np.float32),
        ).save(tmp_path / f"{name}_cbss.pkl")
        CBSSConfig(ext_fact=ext_fact, fs=fs, save_emg=True).to_yaml(
            tmp_path / f"{name}_cbss_config.yaml"
        )
        emg = rng.standard_normal((n_full, raw_chs)).astype(np.float32)
        np.savez(tmp_path / f"{name}_emg.npz", emg=emg)
        path_gt = None
        if gt_dense is not None:
            path_gt = tmp_path / f"{name}_gt.npz"
            np.savez(path_gt, spikes=gt_dense)

        return PooledDatasetDisk(
            path_calib=tmp_path / f"{name}_cbss.pkl",
            path_calib_config=tmp_path / f"{name}_cbss_config.yaml",
            path_emg=tmp_path / f"{name}_emg.npz",
            preprocess=False,  # fs=200 is below AdaptConfig's default highcut
            path_gt=path_gt,
            fs=fs,  # calibration.timestamps is never set, so calibration.fs would raise
        )

    return _make


@pytest.fixture(autouse=True)
def _one_core_by_default(monkeypatch):
    """Make optimize_adapt_decomp's default n_cores 1, so searches run in-process.

    Tests of the worker processes request cores explicitly (see
    allow_cores below), keeping the rest of the suite fast and independent of
    the machine's core count.
    """
    from adapt_decomp.adaptation.optimize import resources

    monkeypatch.setattr(resources, "available_cores", lambda: 1)


@pytest.fixture
def allow_cores(monkeypatch):
    """Factory fixture: let optimize_adapt_decomp use up to n cores in this test.

    Returns:
        Callable[[int], None]: Call with the number of cores to allow.
    """
    from adapt_decomp.adaptation.optimize import resources

    def _allow(n: int) -> None:
        monkeypatch.setattr(resources, "available_cores", lambda: n)

    return _allow
