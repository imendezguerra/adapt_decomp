"""Tests for utils/loaders.py: the CBSSResult-based data-loading contract.

CBSSResult.to_adapt_tensors() and spikes.comparison.pair_ground_truth() have
their own tests in tests/cbss/test_data_structure.py and
tests/spikes/test_comparison.py respectively -- both were promoted out of
this module (see CBSSResult.to_adapt_tensors()/pair_ground_truth()'s own
docstrings) and are exercised here only indirectly, through
TestLoadPooledCbssMemory's integration test.

_cbss_result_from_jne_decomp/load_example's real-file paths (.mat/.hdf5) are
integration-level and not covered here -- these target the pure/cheap-to-test
pieces (per CLAUDE.md's testing guidance), plus a tmp_path-based end-to-end
test of load_pooled_cbss_memory itself since building its on-disk inputs (a
CBSSResult pickle + two .npz files) is cheap.
"""

import h5py
import numpy as np
import pytest

from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.utils.loaders import (
    PooledDatasetMemory,
    _load_bin_spikes_muniverse,
    load_calib,
    load_data,
    load_emg,
    load_gt,
    load_pooled_cbss_disk,
    load_pooled_cbss_memory,
)


def _make_cbss_result(n_mu: int = 2, T: int = 20, D: int = 4, C: int = 2, ext_fact: int = 2) -> CBSSResult:
    """Small, valid CBSSResult for load_pooled_cbss_memory's integration test -- mirrors
    tests/cbss/test_data_structure.py's construction, with emg/timestamps set
    (to_adapt_tensors requires emg; timestamps enables .fs)."""
    spikes = np.zeros((T, n_mu), dtype=np.int32)
    spikes[::5] = 1
    return CBSSResult(
        sources=np.random.randn(T, n_mu).astype(np.float32),
        spikes=spikes,
        spikes_dict={i: np.where(spikes[:, i])[0] for i in range(n_mu)},
        sep_vectors=np.random.randn(D, n_mu).astype(np.float32),
        whitening=np.eye(D, dtype=np.float32),
        extension_mean=np.zeros((1, D), dtype=np.float32),
        spikes_centr=np.ones(n_mu, dtype=np.float32),
        base_centr=np.zeros(n_mu, dtype=np.float32),
        sil=np.full(n_mu, 0.9, dtype=np.float32),
        cov_isi=np.full(n_mu, 0.1, dtype=np.float32),
        ext_fact=ext_fact,
        emg=np.random.randn(T, C).astype(np.float32),
        timestamps=np.arange(T, dtype=np.float64) / 2048.0,  # fs=2048, so rate_of_agreement's
                                                              # tol_spike (round(tol_ms/1000*fs))
                                                              # doesn't round down to an empty kernel
    )


def _make_neuromotion_h5(tmp_path, name: str, emg: np.ndarray, spikes: np.ndarray, fs: float):
    """Write a minimal valid neuromotion-format HDF5 file (every key
    _load_neuromotion reads) for load_emg/load_gt's "neuromotion" branch."""
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("emg", data=emg)
        h5.create_dataset("spikes", data=spikes)
        h5.create_dataset("fs", data=fs)
        h5.create_dataset("rms", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("ch_map", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("ch_cols", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("bad_ch", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("timestamps", data=np.zeros(1, dtype=np.float64))
        h5.create_dataset("angle_profile", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("force_profile", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("muaps", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("muap_muscle_labels", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("muap_angle_labels", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("roa_0deg", data=np.zeros(1, dtype=np.float32))
        h5.create_dataset("lags_0deg", data=np.zeros(1, dtype=np.float32))
        h5.create_group("staircase_phases")
        h5.create_group("paired_units")
    return path


class TestLoadBinSpikes:
    def test_passes_through_an_already_dense_array(self, tmp_path):
        dense = np.zeros((30, 2), dtype=np.float32)
        dense[::3] = 1
        path = tmp_path / "dense_spikes.npz"
        np.savez(path, spikes=dense)

        out = _load_bin_spikes_muniverse(path, n_samples=30)
        np.testing.assert_array_equal(out, dense)

    def test_densifies_a_ragged_firing_index_object_array(self, tmp_path):
        n_samples = 40
        spikes_obj = np.empty(2, dtype=object)
        spikes_obj[0] = [1, 5, 9]
        spikes_obj[1] = [2, 100]  # 100 is out of range for n_samples=40, must be dropped
        path = tmp_path / "ragged_spikes.npz"
        np.savez(path, spikes=spikes_obj)

        out = _load_bin_spikes_muniverse(path, n_samples=n_samples)

        assert out.shape == (n_samples, 2)
        assert out[:, 0].sum() == 3
        assert out[1, 0] == 1 and out[5, 0] == 1 and out[9, 0] == 1
        assert out[:, 1].sum() == 1  # only sample 2 kept; 100 is out of range
        assert out[2, 1] == 1


class TestLoadData:
    def test_raises_on_unknown_loader(self):
        with pytest.raises(ValueError, match="Unknown data loader"):
            load_data({"loader": "not_a_real_loader"})


class TestLoadCalib:
    def test_class_loads_result_and_config(self, tmp_path):
        result = _make_cbss_result()
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        config_path = tmp_path / "cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(config_path)

        cbss_result, cbss_config = load_calib(calib_path, config_path)

        assert cbss_result.sources.shape == result.sources.shape
        assert cbss_config.ext_fact == result.ext_fact

    def test_class_with_no_config_path_returns_none_config(self, tmp_path):
        result = _make_cbss_result()
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)

        cbss_result, cbss_config = load_calib(calib_path, None)

        assert cbss_result.sources.shape == result.sources.shape
        assert cbss_config is None

    def test_unknown_calib_loader_raises(self, tmp_path):
        result = _make_cbss_result()
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        with pytest.raises(ValueError, match="Unknown calib_loader"):
            load_calib(calib_path, None, calib_loader="bogus")


class TestLoadEmg:
    def test_npz_default(self, tmp_path):
        emg = np.random.randn(30, 3).astype(np.float32)
        path = tmp_path / "emg.npz"
        np.savez(path, emg=emg)

        out = load_emg(path)

        np.testing.assert_array_equal(out, emg)

    def test_neuromotion(self, tmp_path):
        emg = np.random.randn(30, 3).astype(np.float32)
        spikes = np.zeros((30, 2), dtype=np.float32)
        path = _make_neuromotion_h5(tmp_path, "cond", emg, spikes, fs=2048.0)

        out = load_emg(path, emg_loader="neuromotion")

        np.testing.assert_array_almost_equal(out, emg)

    def test_unknown_emg_loader_raises(self, tmp_path):
        emg = np.random.randn(10, 2).astype(np.float32)
        path = tmp_path / "emg.npz"
        np.savez(path, emg=emg)
        with pytest.raises(ValueError, match="Unknown emg_loader"):
            load_emg(path, emg_loader="bogus")


class TestLoadGt:
    def test_npz_default(self, tmp_path):
        dense = np.zeros((30, 2), dtype=np.float32)
        dense[::3] = 1
        path = tmp_path / "spikes.npz"
        np.savez(path, spikes=dense)

        out = load_gt(path, n_samples=30)

        np.testing.assert_array_equal(out, dense)

    def test_neuromotion(self, tmp_path):
        emg = np.random.randn(30, 3).astype(np.float32)
        spikes = np.zeros((30, 2), dtype=np.float32)
        spikes[::5] = 1
        path = _make_neuromotion_h5(tmp_path, "cond", emg, spikes, fs=2048.0)

        # n_samples is ignored for the neuromotion branch.
        out = load_gt(path, n_samples=999, gt_loader="neuromotion")

        np.testing.assert_array_almost_equal(out, spikes)

    def test_unknown_gt_loader_raises(self, tmp_path):
        dense = np.zeros((10, 2), dtype=np.float32)
        path = tmp_path / "spikes.npz"
        np.savez(path, spikes=dense)
        with pytest.raises(ValueError, match="Unknown gt_loader"):
            load_gt(path, n_samples=10, gt_loader="bogus")


class TestLoadPooledCbssMemory:
    def test_builds_one_pooled_dataset_per_entry(self, tmp_path):
        n_mu, T, D, C = 2, 20, 4, 2
        result = _make_cbss_result(n_mu=n_mu, T=T, D=D, C=C)
        calib_path = tmp_path / "dataset_a_cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "dataset_a_cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)

        n_full = 60
        emg_full = np.random.randn(n_full, C).astype(np.float32)
        emg_path = tmp_path / "dataset_a_emg.npz"
        np.savez(emg_path, emg=emg_full)

        gt_dense = np.zeros((n_full, n_mu), dtype=np.float32)
        gt_dense[::5] = 1  # matches _make_cbss_result's own spike stride, so select_supervised matches
        gt_path = tmp_path / "dataset_a_spikes.npz"
        np.savez(gt_path, spikes=gt_dense)

        data_config = {
            "root": str(tmp_path),
            "loader": "load_pooled_cbss_memory",
            "datasets": [
                {
                    "name": "dataset_a",
                    "path_emg": emg_path.name,
                    "path_calib": calib_path.name,
                    "path_calib_config": cbss_config_path.name,
                    "path_gt": gt_path.name,
                },
                {
                    "name": "dataset_b_no_gt",
                    "path_emg": emg_path.name,
                    "path_calib": calib_path.name,
                    "path_calib_config": cbss_config_path.name,
                },
            ],
        }

        pool = load_data(data_config)

        assert set(pool.keys()) == {"dataset_a", "dataset_b_no_gt"}
        dataset_a = pool["dataset_a"]
        assert dataset_a.emg.shape == (n_full, C)
        assert dataset_a.calibration.sep_vectors.shape == (D, n_mu)  # CBSSResult's own [dim, n_mu] storage
        assert dataset_a.cbss_config.ext_fact == result.ext_fact
        assert dataset_a.preprocess is True  # default preprocess: true propagated
        assert dataset_a.gt_paired_bin is not None
        assert dataset_a.gt_paired_bin.shape[0] == n_full
        assert dataset_a.gt_paired_bin.shape[1] == dataset_a.calibration.spikes.shape[1]

        assert pool["dataset_b_no_gt"].gt_paired_bin is None

    def test_gt_match_narrows_calibration_and_gt_paired_bin_together(self, tmp_path):
        """A calibration unit with no correlated ground truth is dropped by
        select_supervised, and gt_paired_bin's column count follows it down --
        the bug this session's GT-pairing fix targets."""
        n_mu, T, D, C = 2, 20, 4, 2
        result = _make_cbss_result(n_mu=n_mu, T=T, D=D, C=C)
        result.spikes = result.spikes.copy()
        result.spikes[:, 1] = 0  # unit 1 never fires -- no GT will match it
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)

        n_full = 60
        emg_path = tmp_path / "emg.npz"
        np.savez(emg_path, emg=np.random.randn(n_full, C).astype(np.float32))

        gt_dense = np.zeros((n_full, 1), dtype=np.float32)  # only 1 GT unit, matching unit 0
        gt_dense[::5] = 1
        gt_path = tmp_path / "spikes.npz"
        np.savez(gt_path, spikes=gt_dense)

        data_config = {
            "root": str(tmp_path),
            "loader": "load_pooled_cbss_memory",
            "datasets": [{
                "name": "dataset_a",
                "path_emg": emg_path.name,
                "path_calib": calib_path.name,
                "path_calib_config": cbss_config_path.name,
                "path_gt": gt_path.name,
            }],
        }

        pool = load_data(data_config)

        dataset_a = pool["dataset_a"]
        assert dataset_a.calibration.spikes.shape[1] == 1  # unit 1 dropped
        assert dataset_a.gt_paired_bin.shape[1] == 1

    def test_load_pooled_cbss_memory_matches_load_data_dispatch(self, tmp_path):
        result = _make_cbss_result()
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)
        emg_path = tmp_path / "emg.npz"
        np.savez(emg_path, emg=np.random.randn(40, 2).astype(np.float32))

        data_config = {
            "root": str(tmp_path),
            "loader": "load_pooled_cbss_memory",
            "datasets": [{
                "name": "only", "path_emg": emg_path.name,
                "path_calib": calib_path.name, "path_calib_config": cbss_config_path.name,
            }],
        }

        direct = load_pooled_cbss_memory(data_config)
        via_dispatch = load_data(data_config)
        assert direct.keys() == via_dispatch.keys()

    def test_per_dataset_loader_override(self, tmp_path):
        """One dataset overrides emg_loader to "neuromotion"; both datasets
        still load successfully against the pool-level npz default."""
        n_mu, T, D, C = 2, 20, 4, 2
        result = _make_cbss_result(n_mu=n_mu, T=T, D=D, C=C)
        calib_path = tmp_path / "dataset_cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "dataset_cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)

        n_full = 60
        emg_npz = np.random.randn(n_full, C).astype(np.float32)
        emg_npz_path = tmp_path / "dataset_npz_emg.npz"
        np.savez(emg_npz_path, emg=emg_npz)

        emg_h5 = np.random.randn(n_full, C).astype(np.float32)
        spikes_h5 = np.zeros((n_full, n_mu), dtype=np.float32)
        emg_h5_path = _make_neuromotion_h5(tmp_path, "dataset_neuromotion", emg_h5, spikes_h5, fs=2048.0)

        data_config = {
            "root": str(tmp_path),
            "loader": "load_pooled_cbss_memory",
            "datasets": [
                {
                    "name": "dataset_npz", "path_emg": emg_npz_path.name,
                    "path_calib": calib_path.name, "path_calib_config": cbss_config_path.name,
                },
                {
                    "name": "dataset_neuromotion",
                    "path_emg": emg_h5_path.name,
                    "path_calib": calib_path.name,
                    "path_calib_config": cbss_config_path.name,
                    "emg_loader": "neuromotion",
                },
            ],
        }

        pool = load_data(data_config)

        assert pool["dataset_npz"].emg.shape == (n_full, C)
        assert pool["dataset_neuromotion"].emg.shape == (n_full, C)


class TestLoadPooledCbssDisk:
    def test_builds_one_pooled_dataset_disk_per_entry_without_touching_any_paths(self, tmp_path):
        """load_pooled_cbss_disk never reads a single file -- every path
        (path_gt included) is just stored, resolved fresh per trial by
        optimize_adapt_decomp_pooled_disk instead. Every path below is
        nonexistent, for both datasets, to prove this directly."""
        data_config = {
            "root": str(tmp_path),
            "loader": "load_pooled_cbss_disk",
            "datasets": [
                {
                    "name": "dataset_a",
                    "path_emg": "does_not_exist_emg.npz",
                    "path_calib": "does_not_exist_calib.pkl",
                    "path_calib_config": "does_not_exist_config.yaml",
                    "path_gt": "does_not_exist_spikes.npz",
                    "fs": 500,
                },
                {
                    "name": "dataset_b_no_gt",
                    "path_emg": "does_not_exist_emg_b.npz",
                    "path_calib": "does_not_exist_calib_b.pkl",
                    "path_calib_config": "does_not_exist_config_b.yaml",
                },
            ],
        }

        pool = load_data(data_config)

        assert set(pool.keys()) == {"dataset_a", "dataset_b_no_gt"}
        dataset_a = pool["dataset_a"]
        assert dataset_a.path_calib == tmp_path / "does_not_exist_calib.pkl"
        assert dataset_a.path_calib_config == tmp_path / "does_not_exist_config.yaml"
        assert dataset_a.path_emg == tmp_path / "does_not_exist_emg.npz"
        assert dataset_a.calib_loader == "class"
        assert dataset_a.emg_loader == "npz"
        assert dataset_a.gt_loader == "npz"
        assert dataset_a.preprocess is True  # default preprocess: true propagated
        assert dataset_a.path_gt == tmp_path / "does_not_exist_spikes.npz"
        assert dataset_a.fs == 500

        dataset_b = pool["dataset_b_no_gt"]
        assert dataset_b.path_gt is None
        assert dataset_b.fs is None
        assert dataset_b.path_calib == tmp_path / "does_not_exist_calib_b.pkl"

    def test_load_pooled_cbss_disk_matches_load_data_dispatch(self, tmp_path):
        result = _make_cbss_result()
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)
        emg_path = tmp_path / "emg.npz"
        np.savez(emg_path, emg=np.random.randn(40, 2).astype(np.float32))

        data_config = {
            "root": str(tmp_path),
            "loader": "load_pooled_cbss_disk",
            "datasets": [{
                "name": "only", "path_emg": emg_path.name,
                "path_calib": calib_path.name, "path_calib_config": cbss_config_path.name,
            }],
        }

        direct = load_pooled_cbss_disk(data_config)
        via_dispatch = load_data(data_config)
        assert direct.keys() == via_dispatch.keys()

    def test_per_dataset_loader_override_propagates(self, tmp_path):
        """A dataset-level emg_loader override lands on the resulting
        PooledDatasetDisk unresolved (stored, not consumed immediately)."""
        result = _make_cbss_result()
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)

        data_config = {
            "root": str(tmp_path),
            "loader": "load_pooled_cbss_disk",
            "datasets": [{
                "name": "only",
                "path_emg": "does_not_exist.h5",  # never opened -- no path_gt
                "path_calib": calib_path.name,
                "path_calib_config": cbss_config_path.name,
                "emg_loader": "neuromotion",
            }],
        }

        pool = load_pooled_cbss_disk(data_config)

        assert pool["only"].emg_loader == "neuromotion"


class TestPooledDatasetMemoryResolve:
    def test_resolve_passes_through_its_own_fields(self):
        result = _make_cbss_result()
        cbss_config = CBSSConfig(ext_fact=result.ext_fact)
        emg = np.random.randn(10, 2).astype(np.float32)
        gt_paired_bin = np.zeros((10, 2), dtype=np.float32)
        dataset = PooledDatasetMemory(
            emg=emg, calibration=result, cbss_config=cbss_config,
            preprocess=False, gt_paired_bin=gt_paired_bin,
        )

        out_emg, out_calibration, out_cbss_config, out_preprocess, out_gt = dataset.resolve()

        assert out_emg is emg
        assert out_calibration is result
        assert out_cbss_config is cbss_config
        assert out_preprocess is False
        assert out_gt is gt_paired_bin


class TestPooledDatasetDiskResolve:
    def _build_entry(self, tmp_path, n_mu=2, T=20, D=4, C=2, path_gt=None, fs=None):
        result = _make_cbss_result(n_mu=n_mu, T=T, D=D, C=C)
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)
        emg_path = tmp_path / "emg.npz"
        np.savez(emg_path, emg=np.random.randn(60, C).astype(np.float32))

        dataset = {
            "name": "only", "path_emg": emg_path.name,
            "path_calib": calib_path.name, "path_calib_config": cbss_config_path.name,
        }
        if path_gt is not None:
            dataset["path_gt"] = path_gt
        if fs is not None:
            dataset["fs"] = fs

        pool = load_pooled_cbss_disk({
            "root": str(tmp_path), "loader": "load_pooled_cbss_disk", "datasets": [dataset],
        })
        return pool["only"], result

    def test_resolve_loads_emg_calib_and_config_with_no_gt(self, tmp_path):
        entry, result = self._build_entry(tmp_path)

        emg, calibration, cbss_config, preprocess, gt_paired_bin = entry.resolve()

        assert emg.shape == (60, 2)
        assert calibration.sources.shape[1] == 2
        assert cbss_config.ext_fact == result.ext_fact
        assert preprocess is True
        assert gt_paired_bin is None

    def test_resolve_matches_ground_truth_when_path_gt_is_set(self, tmp_path):
        n_mu, C = 2, 2
        gt_dense = np.zeros((60, n_mu), dtype=np.float32)
        gt_dense[::5] = 1  # matches _make_cbss_result's own spike stride, so select_supervised matches
        gt_path = tmp_path / "spikes.npz"
        np.savez(gt_path, spikes=gt_dense)

        entry, _ = self._build_entry(tmp_path, n_mu=n_mu, C=C, path_gt=gt_path.name)

        emg, calibration, cbss_config, preprocess, gt_paired_bin = entry.resolve()

        assert gt_paired_bin is not None
        assert gt_paired_bin.shape[0] == 60
        assert gt_paired_bin.shape[1] == calibration.spikes.shape[1]

    def test_resolve_gt_match_narrows_calibration_and_gt_paired_bin_together(self, tmp_path):
        """A calibration unit with no correlated ground truth is dropped by
        select_supervised, and gt_paired_bin's column count follows it down --
        the disk-path counterpart of the memory-path partial-match test."""
        n_mu, T, D, C = 2, 20, 4, 2
        result = _make_cbss_result(n_mu=n_mu, T=T, D=D, C=C)
        result.spikes = result.spikes.copy()
        result.spikes[:, 1] = 0  # unit 1 never fires -- no GT will match it
        calib_path = tmp_path / "cbss.pkl"
        result.save(calib_path)
        cbss_config_path = tmp_path / "cbss_config.yaml"
        CBSSConfig(ext_fact=result.ext_fact).to_yaml(cbss_config_path)
        emg_path = tmp_path / "emg.npz"
        np.savez(emg_path, emg=np.random.randn(60, C).astype(np.float32))

        gt_dense = np.zeros((60, 1), dtype=np.float32)  # only 1 GT unit, matching unit 0
        gt_dense[::5] = 1
        gt_path = tmp_path / "spikes.npz"
        np.savez(gt_path, spikes=gt_dense)

        pool = load_pooled_cbss_disk({
            "root": str(tmp_path), "loader": "load_pooled_cbss_disk",
            "datasets": [{
                "name": "only", "path_emg": emg_path.name,
                "path_calib": calib_path.name, "path_calib_config": cbss_config_path.name,
                "path_gt": gt_path.name,
            }],
        })

        emg, calibration, cbss_config, preprocess, gt_paired_bin = pool["only"].resolve()

        assert calibration.spikes.shape[1] == 1  # unit 1 dropped
        assert gt_paired_bin.shape[1] == 1

    def test_resolve_touches_no_files_until_called(self, tmp_path):
        """load_pooled_cbss_disk itself still touches nothing -- resolve() is
        where loading actually happens, called fresh each time."""
        entry, _ = self._build_entry(tmp_path)
        assert entry.path_calib.exists()  # sanity: the files really are there

        # Calling resolve() twice re-reads from disk both times (nothing cached).
        first = entry.resolve()
        second = entry.resolve()
        assert first[1].sources.shape == second[1].sources.shape
