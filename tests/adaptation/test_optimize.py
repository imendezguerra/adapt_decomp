"""Tests for adaptation/optimize.py: optimize_adapt_decomp_pooled_memory / _pooled_disk.

All marked slow -- each spins up a real (if tiny) Optuna study end to end.
"""

import numpy as np
import pytest
import torch

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.adaptation.ops import orthonormalize_rows_qr
from adapt_decomp.adaptation.optimize import (
    optimize_adapt_decomp_pooled_memory,
    optimize_adapt_decomp_pooled_disk,
    DEFAULT_PARAM_SPACE,
)
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.utils.loaders import PooledDatasetMemory, PooledDatasetDisk

pytestmark = pytest.mark.slow


def test_optimize_adapt_decomp_pooled_memory_best_result_path_optional(tmp_path, make_optimize_kwargs):
    """best_result_path stays a fully optional opt-in: omitted -> (best_config,
    study); set -> (best_outputs, best_config, study) with the winning trial's
    per-dataset AdaptationResults and their files written to disk -- the
    one-entry-pool case, since a single dataset is just a pool of one."""
    common, _ = make_optimize_kwargs()
    pool = {"dataset_a": PooledDatasetMemory(
        emg=common["emg"], calibration=common["calibration"],
        cbss_config=common["cbss_config"], preprocess=common["preprocess"],
    )}

    result_no_path = optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2, base_config=common["base_config"],
    )
    assert len(result_no_path) == 2
    best_config, study = result_no_path
    assert isinstance(best_config, AdaptConfig)
    assert isinstance(study.best_value, float)

    best_dir = tmp_path / "best_trial"
    result_with_path = optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
        base_config=common["base_config"], best_result_path=str(best_dir),
    )
    assert len(result_with_path) == 3
    outputs, best_config2, study2 = result_with_path
    assert set(outputs.keys()) == {"dataset_a"}
    assert isinstance(outputs["dataset_a"], AdaptationResult)
    assert outputs["dataset_a"].wh_loss is not None
    for fname in ("dataset_a.pkl", "config.yaml", "study.pkl"):
        assert (best_dir / fname).exists()


def test_optimize_adapt_decomp_pooled_memory_compute_roa_writes_outputs_roa(tmp_path, make_optimize_kwargs):
    """compute_roa=True writes RoA onto the winning trial's per-dataset
    AdaptationResult.roa when best_result_path is set."""
    common, M = make_optimize_kwargs()
    n_samples = common["emg"].shape[0]
    gt_full_bin = np.zeros((n_samples, M), dtype=np.float32)
    gt_full_bin[::30] = 1
    # Default roa_kwargs tol_spike_ms=2 rounds to 0 samples at this fixture's low
    # fs=200 (round(2 * 200 / 1000) == 0), which rate_of_agreement_paired can't
    # convolve with -- widen it for this fs, same as a real low-fs caller would.
    roa_kwargs = {"tol_spike_ms": 25}
    pool = {"dataset_a": PooledDatasetMemory(
        emg=common["emg"], calibration=common["calibration"],
        cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        gt_paired_bin=gt_full_bin,
    )}

    best_dir = tmp_path / "best_trial_roa"
    outputs, _, _ = optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
        base_config=common["base_config"], compute_roa=True, roa_kwargs=roa_kwargs,
        best_result_path=str(best_dir),
    )
    assert outputs["dataset_a"].roa is not None
    assert outputs["dataset_a"].roa.shape == (M,)


def test_optimize_adapt_decomp_pooled_memory_on_trial_log_vars(make_optimize_kwargs):
    """optimize_adapt_decomp_pooled_memory's on_trial log dict carries the pooled SUM
    as "loss" plus a "per_dataset" entry (with its own loss/RoA) for every
    dataset in the pool -- the schema that fixed the original bug (an
    external wandb callback assuming the single-dataset user_attrs keys,
    which pooled never sets)."""
    common_a, M = make_optimize_kwargs()
    common_b, _ = make_optimize_kwargs()
    n_samples = common_a["emg"].shape[0]
    gt_full_bin = np.zeros((n_samples, M), dtype=np.float32)
    gt_full_bin[::30] = 1
    pool = {
        "dataset_a": PooledDatasetMemory(
            emg=common_a["emg"], calibration=common_a["calibration"],
            cbss_config=common_a["cbss_config"], preprocess=common_a["preprocess"],
            gt_paired_bin=gt_full_bin,
        ),
        "dataset_b": PooledDatasetMemory(
            emg=common_b["emg"], calibration=common_b["calibration"],
            cbss_config=common_b["cbss_config"], preprocess=common_b["preprocess"],
            gt_paired_bin=gt_full_bin,
        ),
    }
    seen = []

    optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
        base_config=common_a["base_config"],
        compute_roa=True, roa_kwargs={"tol_spike_ms": 25}, on_trial=seen.append,
    )

    assert len(seen) == 2
    for v in seen:
        assert set(v["per_dataset"]) == {"dataset_a", "dataset_b"}
        assert v["loss"] == pytest.approx(
            sum(d["loss"] for d in v["per_dataset"].values())
        )
        for dataset_vars in v["per_dataset"].values():
            assert "roa_mean" in dataset_vars and "roa_per_unit" in dataset_vars
        assert "roa_mean" in v  # pooled mean, present since compute_roa=True


@pytest.mark.parametrize("objective", ["sv_loss", "wh_loss", "total_loss"])
def test_optimize_adapt_decomp_pooled_memory_log_vars_carry_all_losses(objective, make_optimize_kwargs):
    """Pooled log dict carries sv_loss/wh_loss/total_loss (each the SUM across
    datasets) alongside "loss" == the sum for objective, plus per-dataset
    sv_loss/wh_loss/total_loss that sum to the same top-level values."""
    common_a, M = make_optimize_kwargs()
    common_b, _ = make_optimize_kwargs()
    pool = {
        "dataset_a": PooledDatasetMemory(
            emg=common_a["emg"], calibration=common_a["calibration"],
            cbss_config=common_a["cbss_config"], preprocess=common_a["preprocess"],
        ),
        "dataset_b": PooledDatasetMemory(
            emg=common_b["emg"], calibration=common_b["calibration"],
            cbss_config=common_b["cbss_config"], preprocess=common_b["preprocess"],
        ),
    }
    seen = []

    optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2, objective=objective,
        base_config=common_a["base_config"], on_trial=seen.append,
    )

    assert len(seen) == 2
    for v in seen:
        assert v["objective"] == objective
        assert v["loss"] == pytest.approx(v[objective])
        for key in ("sv_loss", "wh_loss", "total_loss"):
            assert v[key] == pytest.approx(
                sum(d[key] for d in v["per_dataset"].values())
            )


def test_optimize_adapt_decomp_pooled_memory_invalid_objective_raises(make_optimize_kwargs):
    """An objective outside {"sv_loss", "wh_loss", "total_loss"} raises ValueError
    up front, before any trial runs."""
    common, _ = make_optimize_kwargs()
    pool = {"dataset_a": PooledDatasetMemory(
        emg=common["emg"], calibration=common["calibration"],
        cbss_config=common["cbss_config"], preprocess=common["preprocess"],
    )}
    with pytest.raises(ValueError):
        optimize_adapt_decomp_pooled_memory(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1, objective="bogus",
            base_config=common["base_config"],
        )


def test_optimize_adapt_decomp_pooled_memory_objective_roa(make_optimize_kwargs):
    """Pooled objective="roa" sums each dataset's guarded, inverted "roa" loss
    (NOT the same convention as the roa_mean_pooled diagnostic, which is a mean)."""
    common_a, M = make_optimize_kwargs()
    common_b, _ = make_optimize_kwargs()
    n_samples = common_a["emg"].shape[0]
    gt_full_bin = np.zeros((n_samples, M), dtype=np.float32)
    gt_full_bin[::30] = 1
    pool = {
        "dataset_a": PooledDatasetMemory(
            emg=common_a["emg"], calibration=common_a["calibration"],
            cbss_config=common_a["cbss_config"], preprocess=common_a["preprocess"],
            gt_paired_bin=gt_full_bin,
        ),
        "dataset_b": PooledDatasetMemory(
            emg=common_b["emg"], calibration=common_b["calibration"],
            cbss_config=common_b["cbss_config"], preprocess=common_b["preprocess"],
            gt_paired_bin=gt_full_bin,
        ),
    }
    seen = []

    optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2, objective="roa",
        roa_kwargs={"tol_spike_ms": 25}, base_config=common_a["base_config"], on_trial=seen.append,
    )

    assert len(seen) == 2
    for v in seen:
        assert v["objective"] == "roa"
        assert v["loss"] == pytest.approx(v["roa"])
        assert v["roa"] == pytest.approx(
            sum(d["roa"] for d in v["per_dataset"].values())
        )
        # roa_mean_pooled (mean diagnostic) is a different aggregation than the pooled
        # "roa" sum -- they should not coincide in general.
        assert "roa_mean" in v


def test_optimize_adapt_decomp_pooled_memory_objective_roa_requires_gt_full_bin(make_optimize_kwargs):
    """Pooled objective="roa" without every dataset's gt_paired_bin raises ValueError
    up front, before any trial runs."""
    common, _ = make_optimize_kwargs()
    pool = {"dataset_a": PooledDatasetMemory(
        emg=common["emg"], calibration=common["calibration"],
        cbss_config=common["cbss_config"], preprocess=common["preprocess"],
    )}
    with pytest.raises(ValueError):
        optimize_adapt_decomp_pooled_memory(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1, objective="roa",
            base_config=common["base_config"],
        )


def test_optimize_adapt_decomp_pooled_memory_partial_gt_match(make_optimize_kwargs):
    """A dataset whose calibration has an unmatched unit -- select_supervised
    (what load_pooled_cbss_memory uses, exercised directly here) narrows both
    the calibration and its ground truth to the matched subset, so the search
    runs without rate_of_agreement_paired's shape-mismatch ValueError -- the
    bug this session's GT-pairing fix targets."""
    common, M = make_optimize_kwargs()
    calibration = common["calibration"]
    calibration.spikes = calibration.spikes.copy()
    calibration.spikes[:, 1] = 0  # unit 1 never fires in the calib window -- unmatchable

    n_full = common["emg"].shape[0]
    spikes_gt_full = np.zeros((n_full, 1), dtype=np.float32)  # only 1 GT unit, matching unit 0
    spikes_gt_full[::20] = 1  # matches unit 0's own stride-20 calibration pattern
    # select_supervised()'s own default tol_spike_ms=0.5 rounds to 0 samples at
    # this fixture's low fs=200 -- widen it, same as this module's roa_kwargs convention.
    matched = calibration.select_supervised(
        spikes_gt_full[: calibration.sources.shape[0]],
        fs=common["base_config"].fs, tol_spike_ms=25,
    )
    gt_paired_bin = spikes_gt_full[:, matched.gt_matched_indices]
    assert matched.spikes.shape[1] == 1  # confirms the partial match actually happened

    pool = {"dataset_a": PooledDatasetMemory(
        emg=common["emg"], calibration=matched, cbss_config=common["cbss_config"],
        preprocess=common["preprocess"], gt_paired_bin=gt_paired_bin,
    )}

    best_config, study = optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1,
        base_config=common["base_config"], compute_roa=True, roa_kwargs={"tol_spike_ms": 25},
    )
    assert isinstance(study.best_value, float)


def test_optimize_adapt_decomp_pooled_memory_n_jobs_completes_with_best_result_path(tmp_path, make_optimize_kwargs):
    """n_jobs>1 runs trials concurrently (Optuna's thread-based study.optimize
    n_jobs) -- resolve()/_run_one_dataset() add no new shared state across
    concurrent trials, so the search still completes every trial and
    best_result_path still ends up holding a single, correctly-promoted best
    trial, exercising the best_lock guarding best_loss/best_outputs."""
    common_a, _ = make_optimize_kwargs()
    common_b, _ = make_optimize_kwargs()
    pool = {
        "dataset_a": PooledDatasetMemory(
            emg=common_a["emg"], calibration=common_a["calibration"],
            cbss_config=common_a["cbss_config"], preprocess=common_a["preprocess"],
        ),
        "dataset_b": PooledDatasetMemory(
            emg=common_b["emg"], calibration=common_b["calibration"],
            cbss_config=common_b["cbss_config"], preprocess=common_b["preprocess"],
        ),
    }
    best_dir = tmp_path / "best_trial"

    outputs, best_config, study = optimize_adapt_decomp_pooled_memory(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=6, n_jobs=2,
        base_config=common_a["base_config"], best_result_path=str(best_dir),
    )

    assert len(study.trials) == 6
    assert set(outputs.keys()) == {"dataset_a", "dataset_b"}
    for name in pool:
        assert (best_dir / f"{name}.pkl").exists()


# ------------------------------------------------------------------
# optimize_adapt_decomp_pooled_disk -- same shapes as make_optimize_kwargs
# (raw_chs=3, ext_fact=2, M=2, fs=200, n_cal=500, n_full=600), but written to
# tmp_path as a CBSSResult pickle + CBSSConfig YAML + emg .npz per dataset,
# since this function loads via AdaptDecomp.from_calibration() instead of
# taking pre-unpacked tensors.
# ------------------------------------------------------------------

def _make_pooled_disk_dataset(
    tmp_path, name, raw_chs=3, ext_fact=2, M=2, fs=200,
    n_cal=500, n_full=600, spike_stride=20, seed=0, gt_dense=None, spikes=None,
):
    """Write one dataset's CBSSResult pickle/CBSSConfig YAML/emg .npz (and,
    if gt_dense is given, a ground-truth .npz) to tmp_path and return the
    matching PooledDatasetDisk -- shapes/fs/ext_fact match
    make_optimize_kwargs's base_config so from_calibration() reconciles
    nothing here (that's test_from_calibration.py's job, not this module's).

    Args:
        spikes (Optional[np.ndarray]): Calibration spike train override,
            with shape (n_cal, M). Defaults to None, which uses
            spikes[::spike_stride] = 1 for every unit.
        gt_dense (Optional[np.ndarray]): Full-recording ground-truth binary
            spike train, with shape (n_full, n_gt) -- written to a
            "<name>_gt.npz" file and set as path_gt. Defaults to None (no
            ground truth).
    """
    rng = np.random.default_rng(seed)
    D = raw_chs * ext_fact
    if spikes is None:
        spikes = np.zeros((n_cal, M), dtype=np.int32)
        spikes[::spike_stride] = 1
    sv = orthonormalize_rows_qr(
        torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32))
    ).numpy()

    result = CBSSResult(
        sources=rng.standard_normal((n_cal, M)).astype(np.float32),
        spikes=spikes,
        spikes_dict={i: np.where(spikes[:, i])[0] for i in range(M)},
        sep_vectors=sv.T,  # CBSSResult stores [dim, n_mu]; to_adapt_tensors() transposes back
        whitening=np.eye(D, dtype=np.float32),
        extension_mean=np.zeros((1, D), dtype=np.float32),
        spikes_centr=(rng.random(M) + 2.0).astype(np.float32),
        base_centr=(rng.random(M) * 0.5).astype(np.float32),
        sil=np.full(M, 0.9, dtype=np.float32),
        cov_isi=np.full(M, 0.1, dtype=np.float32),
        ext_fact=ext_fact,
        emg=rng.standard_normal((n_cal, raw_chs)).astype(np.float32),
    )
    calib_path = tmp_path / f"{name}_cbss.pkl"
    result.save(calib_path)

    cbss_config_path = tmp_path / f"{name}_cbss_config.yaml"
    CBSSConfig(ext_fact=ext_fact, fs=fs, save_emg=True).to_yaml(cbss_config_path)

    emg_path = tmp_path / f"{name}_emg.npz"
    np.savez(emg_path, emg=rng.standard_normal((n_full, raw_chs)).astype(np.float32))

    path_gt = None
    if gt_dense is not None:
        path_gt = tmp_path / f"{name}_gt.npz"
        np.savez(path_gt, spikes=gt_dense)

    return PooledDatasetDisk(
        path_calib=calib_path,
        path_calib_config=cbss_config_path,
        path_emg=emg_path,
        preprocess=False,  # matches make_optimize_kwargs -- fs=200 is below AdaptConfig's default highcut
        path_gt=path_gt,
        fs=fs,  # calibration.timestamps is never set here, so calibration.fs would raise
    )


def _make_pooled_disk_base_config(fs=200):
    """AdaptConfig matching _make_pooled_disk_dataset's shapes -- same
    values as make_optimize_kwargs's base_config."""
    cfg = AdaptConfig()
    cfg.device = "cpu"
    cfg.fs = fs
    cfg.ext_fact = 2
    cfg.batch_ms = 100
    cfg.__post_init__()
    return cfg


class TestOptimizeAdaptDecompPooledDisk:
    def test_on_trial_log_vars_match_optimize_adapt_decomp_pooled_memory_shape(self, tmp_path):
        """Same canonical on_trial log dict shape as optimize_adapt_decomp_pooled_memory
        (loss = pooled SUM, one per_dataset entry per pool key)."""
        base_config = _make_pooled_disk_base_config()
        pool = {
            "dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a", seed=0),
            "dataset_b": _make_pooled_disk_dataset(tmp_path, "dataset_b", seed=1),
        }
        seen = []

        optimize_adapt_decomp_pooled_disk(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
            base_config=base_config, on_trial=seen.append,
        )

        assert len(seen) == 2
        for v in seen:
            assert set(v["per_dataset"]) == {"dataset_a", "dataset_b"}
            assert v["loss"] == pytest.approx(
                sum(d["loss"] for d in v["per_dataset"].values())
            )

    def test_best_result_path_optional_and_writes_reloadable_results(self, tmp_path):
        """best_result_path stays fully optional, as in optimize_adapt_decomp_pooled_memory;
        when set, best_result_path holds one reloadable AdaptationResult per
        dataset plus config.yaml/study.pkl, and the scratch "_temp" directory
        used to build them is removed once the search finishes."""
        base_config = _make_pooled_disk_base_config()
        pool = {
            "dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a", seed=0),
            "dataset_b": _make_pooled_disk_dataset(tmp_path, "dataset_b", seed=1),
        }

        result_no_path = optimize_adapt_decomp_pooled_disk(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2, base_config=base_config,
        )
        assert len(result_no_path) == 2
        best_config, study = result_no_path
        assert isinstance(best_config, AdaptConfig)
        assert isinstance(study.best_value, float)

        best_dir = tmp_path / "best_trial"
        result_with_path = optimize_adapt_decomp_pooled_disk(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
            base_config=base_config, best_result_path=str(best_dir),
        )
        # Unlike optimize_adapt_decomp_pooled_memory, the return shape never changes --
        # results are on disk, not in memory (see docstring's Returns).
        assert len(result_with_path) == 2

        for name in pool:
            assert (best_dir / f"{name}.pkl").exists()
            reloaded = AdaptationResult.load(best_dir / f"{name}.pkl")
            assert reloaded.wh_loss is not None
        assert (best_dir / "config.yaml").exists()
        assert (best_dir / "study.pkl").exists()
        assert not best_dir.with_name(best_dir.name + "_temp").exists()

    def test_n_jobs_promotes_correct_files_without_scratch_collisions(self, tmp_path):
        """Under n_jobs>1, concurrent trials write trial-scoped scratch
        filenames ("<trial.number>_<dataset>.pkl") instead of colliding on a
        shared "<dataset>.pkl" -- best_result_path still ends up with exactly
        one reloadable AdaptationResult per dataset, and the scratch "_temp"
        directory is fully cleaned up (both each trial's own files, unlinked
        right after its promote check, and the directory itself at the end)."""
        base_config = _make_pooled_disk_base_config()
        pool = {
            "dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a", seed=0),
            "dataset_b": _make_pooled_disk_dataset(tmp_path, "dataset_b", seed=1),
        }
        best_dir = tmp_path / "best_trial"

        best_config, study = optimize_adapt_decomp_pooled_disk(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=8, n_jobs=4,
            base_config=base_config, best_result_path=str(best_dir),
        )

        assert len(study.trials) == 8
        for name in pool:
            assert (best_dir / f"{name}.pkl").exists()
            reloaded = AdaptationResult.load(best_dir / f"{name}.pkl")
            assert reloaded.wh_loss is not None
        assert (best_dir / "config.yaml").exists()
        assert (best_dir / "study.pkl").exists()
        assert not best_dir.with_name(best_dir.name + "_temp").exists()

    def test_compute_roa(self, tmp_path):
        """compute_roa=True logs roa_mean/roa_per_unit as user_attrs on every
        trial AND on on_trial's log_vars, matching optimize_adapt_decomp_pooled_memory."""
        M, n_full = 2, 600
        # select_supervised()'s own default tol_spike_ms=0.5 needs fs high enough
        # that round(0.5 * fs / 1000) doesn't round to 0 samples (an empty
        # convolution kernel) -- 200 (this module's usual fixture fs) is too low.
        fs = 2000
        gt_dense = np.zeros((n_full, M), dtype=np.float32)
        # Matches _make_pooled_disk_dataset's own default spike_stride=20, so
        # select_supervised finds a real correlated match for every unit.
        gt_dense[::20] = 1
        base_config = _make_pooled_disk_base_config(fs=fs)
        pool = {
            "dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a", fs=fs, seed=0, gt_dense=gt_dense),
        }
        # Widened for consistency with the optimize_adapt_decomp_pooled_memory
        # tests' roa_kwargs -- rate_of_agreement_paired's own default (2ms) would
        # work fine at this fs, but this keeps the convention uniform.
        roa_kwargs = {"tol_spike_ms": 25}
        seen = []

        optimize_adapt_decomp_pooled_disk(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
            base_config=base_config, compute_roa=True, roa_kwargs=roa_kwargs,
            on_trial=seen.append,
        )

        assert len(seen) == 2
        for v in seen:
            assert "roa_mean" in v
            assert "roa_mean" in v["per_dataset"]["dataset_a"]
            assert len(v["per_dataset"]["dataset_a"]["roa_per_unit"]) == M

    def test_compute_roa_with_partial_gt_match(self, tmp_path):
        """A calibration unit with no correlated ground truth is dropped by
        select_supervised (called fresh every trial), not left to crash
        rate_of_agreement_paired's shape check -- the bug this session's
        GT-pairing fix targets."""
        M, n_cal, n_full = 3, 500, 600
        # See test_compute_roa's comment on why fs=200 (this module's usual
        # fixture fs) is too low for select_supervised's default tol_spike_ms.
        fs = 2000
        spikes = np.zeros((n_cal, M), dtype=np.int32)
        spikes[::20, :2] = 1  # units 0/1 fire; unit 2 never fires -- no GT will match it
        gt_dense = np.zeros((n_full, 2), dtype=np.float32)  # only 2 GT units, matching units 0/1
        gt_dense[::20] = 1

        base_config = _make_pooled_disk_base_config(fs=fs)
        pool = {"dataset_a": _make_pooled_disk_dataset(
            tmp_path, "dataset_a", M=M, fs=fs, seed=0, spikes=spikes, gt_dense=gt_dense,
        )}

        best_config, study = optimize_adapt_decomp_pooled_disk(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1, base_config=base_config,
            compute_roa=True, roa_kwargs={"tol_spike_ms": 25},
        )
        assert isinstance(study.best_value, float)

    def test_compute_roa_requires_path_gt(self, tmp_path):
        """compute_roa=True without every dataset's path_gt raises
        ValueError up front, before any trial runs."""
        base_config = _make_pooled_disk_base_config()
        pool = {"dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a")}
        with pytest.raises(ValueError):
            optimize_adapt_decomp_pooled_disk(
                pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1,
                base_config=base_config, compute_roa=True,
            )

    def test_invalid_objective_raises(self, tmp_path):
        base_config = _make_pooled_disk_base_config()
        pool = {"dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a")}
        with pytest.raises(ValueError):
            optimize_adapt_decomp_pooled_disk(
                pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1,
                base_config=base_config, objective="bogus",
            )

    def test_uses_emg_loader_not_hardcoded_np_load(self, tmp_path):
        """The trial loop calls loaders.load_emg(spec.path_emg, spec.emg_loader)
        rather than a hardcoded np.load(...)["emg"] -- proven by pointing
        path_emg at a non-.npz (neuromotion HDF5) file via emg_loader="neuromotion"
        and confirming the search still completes."""
        import h5py

        base_config = _make_pooled_disk_base_config()
        dataset = _make_pooled_disk_dataset(tmp_path, "dataset_a", seed=0)

        raw_chs, n_full = 3, 600
        emg_h5 = np.random.default_rng(0).standard_normal((n_full, raw_chs)).astype(np.float32)
        path_emg_h5 = tmp_path / "dataset_a_emg.h5"
        with h5py.File(path_emg_h5, "w") as h5:
            h5.create_dataset("emg", data=emg_h5)
            h5.create_dataset("spikes", data=np.zeros((n_full, 2), dtype=np.float32))
            h5.create_dataset("fs", data=200.0)
            for key in ("rms", "ch_map", "ch_cols", "bad_ch", "angle_profile",
                        "force_profile", "muaps", "muap_muscle_labels",
                        "muap_angle_labels", "roa_0deg", "lags_0deg"):
                h5.create_dataset(key, data=np.zeros(1, dtype=np.float32))
            h5.create_dataset("timestamps", data=np.zeros(1, dtype=np.float64))
            h5.create_group("staircase_phases")
            h5.create_group("paired_units")

        dataset.path_emg = path_emg_h5
        dataset.emg_loader = "neuromotion"
        pool = {"dataset_a": dataset}

        best_config, study = optimize_adapt_decomp_pooled_disk(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1, base_config=base_config,
        )
        assert isinstance(study.best_value, float)
