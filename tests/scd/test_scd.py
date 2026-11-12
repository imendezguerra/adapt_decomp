"""Tests for scd/core.py: SCD calibration on the synthetic recording (needs the scd extra)."""

import numpy as np
import pytest
import torch

from adapt_decomp import SCD, AdaptDecomp, SCDConfig
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.ops import orthonormalize_rows_qr
from adapt_decomp.spikes import compare_spikes

pytest.importorskip("scd")

# Small search: finds the 4 synthetic units in a few seconds
FAST_SCD = {
    "max_iterations": 8,
    "iteration_patience": 3,
    "max_swarm_steps": 4,
    "swarm_patience": 2,
    "max_ica_steps": 150,
    "ica_patience": 20,
}


def _config(rec, **overrides) -> SCDConfig:
    return SCDConfig(fs=rec.fs, ext_fact=8, random_seed=0, scd_kwargs=FAST_SCD, **overrides)


@pytest.fixture(scope="module")
def scd_calibration(synthetic_recording):
    rec = synthetic_recording
    return SCD(_config(rec)).decompose(rec.emg[: rec.n_cal])


def test_scd_finds_the_synthetic_units(synthetic_recording, scd_calibration):
    """Every ground-truth unit is found; peel-off does not change the reconstructed spikes."""
    rec, result = synthetic_recording, scd_calibration
    agreement = compare_spikes(rec.spikes[: rec.n_cal], result.spikes, fs=rec.fs, tol_spike_ms=1)

    assert len(agreement.pairs) == rec.spikes.shape[1]
    assert (agreement.roa > 0.9).all()
    assert result.sources_native.shape == result.sources.shape
    assert set(result.spikes_dict_native) == set(result.spikes_dict)
    assert (result.native_roa > 0.9).all()
    assert result.contrast_exp.shape == (result.sources.shape[1],)
    assert np.isfinite(result.contrast_exp).all()


def test_sep_vectors_are_orthonormal_so_adaptation_keeps_them(scd_calibration):
    """Adaptation QR-orthonormalises the separation vectors every update; SCD's are
    orthonormalised at calibration, so that leaves them unchanged (up to sign)."""
    sv = torch.as_tensor(scd_calibration.sep_vectors).T
    torch.testing.assert_close(sv @ sv.T, torch.eye(sv.shape[0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(orthonormalize_rows_qr(sv).abs(), sv.abs(), atol=1e-5, rtol=0)


def test_sep_vectors_are_the_orthonormalised_spike_triggered_averages(scd_calibration):
    """sep_vectors = S (S^T S)^(-1/2) for S the whitened EMG averaged at the native spikes, so
    the mean of source k at unit j's native spikes, (sep_vectors^T S)[k, j] = (S^T S)^(1/2),
    is symmetric positive definite; SCD's own filters are kept in sep_vectors_native."""
    result = scd_calibration
    n_mu = result.sources.shape[1]
    proj = np.stack(
        [result.sources[result.spikes_dict_native[j]].mean(axis=0) for j in range(n_mu)], axis=1
    ).astype(np.float64)

    np.testing.assert_allclose(proj, proj.T, atol=1e-4 * np.abs(proj).max())
    assert np.linalg.eigvalsh((proj + proj.T) / 2).min() > 0
    assert result.sep_vectors_native.shape == result.sep_vectors.shape
    assert not np.allclose(np.abs(result.sep_vectors_native), np.abs(result.sep_vectors), atol=1e-3)


def test_adaptation_runs_from_an_scd_calibration(synthetic_recording, scd_calibration):
    rec = synthetic_recording
    adapter = AdaptDecomp.from_calibration(
        scd_calibration, adapt_config=AdaptConfig(device="cpu", fs=rec.fs, ext_fact=8)
    )
    n_adapt = 2 * rec.fs
    outputs = adapter.process_data(rec.emg[rec.n_cal : rec.n_cal + n_adapt])
    assert outputs.spikes.shape == (n_adapt, scd_calibration.spikes.shape[1])


def test_seeded_cpu_runs_are_bit_identical(synthetic_recording, scd_calibration):
    rec = synthetic_recording
    again = SCD(_config(rec)).decompose(rec.emg[: rec.n_cal])
    np.testing.assert_array_equal(again.sep_vectors, scd_calibration.sep_vectors)
    np.testing.assert_array_equal(again.sources, scd_calibration.sources)


def test_unknown_scd_kwargs_raise(synthetic_recording):
    rec = synthetic_recording
    config = SCDConfig(fs=rec.fs, ext_fact=8, scd_kwargs={"max_iteration": 5})
    with pytest.raises(ValueError, match=r"unknown scd\.Config options"):
        SCD(config).decompose(rec.emg[: rec.fs])


@pytest.mark.slow
def test_finds_the_same_units_as_vanilla_scd(synthetic_recording):
    """SCD on the shared whitening finds the units vanilla SCD (own extension and
    whitening) finds on the same EMG."""
    from scd.config.structures import Config, set_random_seed
    from scd.models.scd import SwarmContrastiveDecomposition

    rec = synthetic_recording
    emg = rec.emg[: rec.n_cal]
    ours = SCD(_config(rec, preprocess_emg=False)).decompose(emg)

    vanilla_config = Config(
        sampling_frequency=rec.fs, extension_factor=8, verbose_mode=False, device="cpu"
    )
    for key, value in FAST_SCD.items():
        setattr(vanilla_config, key, value)
    set_random_seed(0)  # the same seed as ours, so the result does not depend on test order
    _, vanilla = SwarmContrastiveDecomposition().run(
        torch.as_tensor(emg, dtype=torch.float32), vanilla_config
    )
    vanilla_spikes = np.zeros((emg.shape[0], len(vanilla["timestamps"])), dtype=np.int32)
    for unit, ts in enumerate(vanilla["timestamps"]):
        vanilla_spikes[ts.cpu().numpy(), unit] = 1

    agreement = compare_spikes(vanilla_spikes, ours.spikes, fs=rec.fs, tol_spike_ms=1)
    assert len(agreement.pairs) == vanilla_spikes.shape[1]
    assert (agreement.roa > 0.9).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_seeded_cuda_runs_are_bit_identical(synthetic_recording):
    """Deterministic kernels make a seeded GPU calibration repeat on the same GPU."""
    rec = synthetic_recording
    first, second = (
        SCD(_config(rec, device="cuda")).decompose(rec.emg[: rec.n_cal]) for _ in range(2)
    )
    np.testing.assert_array_equal(first.sep_vectors, second.sep_vectors)
    np.testing.assert_array_equal(first.spikes, second.spikes)
