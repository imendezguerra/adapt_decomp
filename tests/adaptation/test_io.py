"""Tests for adaptation/io.py's HDF5 save/load round-trip."""

import numpy as np
import pytest

from adapt_decomp.adaptation.io import H5ParamsBatchWriter, load_output


class TestH5ParamsBatchWriterAppend:
    def test_append_batch_by_batch_round_trips_through_load_output(self, tmp_path):
        path = tmp_path / "params.h5"
        writer = H5ParamsBatchWriter(
            path=path, wh_shape=(2, 2), sv_shape=(3, 2), sd_shape=(3,), batches=2,
        )
        writer._append({
            "whitening": np.eye(2, dtype=np.float32),
            "sep_vectors": np.ones((3, 2), dtype=np.float32),
            "base_centr": np.zeros(3, dtype=np.float32),
            "spikes_centr": np.ones(3, dtype=np.float32),
        })
        writer._append({
            "whitening": np.eye(2, dtype=np.float32) * 2,
            "sep_vectors": np.ones((3, 2), dtype=np.float32) * 2,
            "base_centr": np.zeros(3, dtype=np.float32),
            "spikes_centr": np.ones(3, dtype=np.float32) * 2,
        })

        out = load_output(path)

        assert out["whitening"].shape == (2, 2, 2)
        np.testing.assert_allclose(out["whitening"][0], np.eye(2))
        np.testing.assert_allclose(out["whitening"][1], np.eye(2) * 2)


class TestH5ParamsBatchWriterSave:
    def test_save_then_load_output_handles_scalar_and_array_values_together(self, tmp_path):
        """Regression test: AdaptationResult.to_dict() mixes array fields (wh_loss,
        shape (batches,)) with scalar summary fields (wh_loss_total, a bare float) --
        load_output() must read both back, not just the array ones."""
        path = tmp_path / "outputs.h5"
        writer = H5ParamsBatchWriter(path=path, wh_shape=(2, 2), sv_shape=(2, 2), sd_shape=(2,))
        writer._save({
            "wh_loss": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "wh_loss_total": np.float32(0.2),  # scalar, as AdaptationResult._compute_losses() produces
            "diagnostics": {"should": "be skipped"},
        })

        out = load_output(path)

        np.testing.assert_allclose(out["wh_loss"], [0.1, 0.2, 0.3])
        assert float(out["wh_loss_total"]) == pytest.approx(0.2)
        assert "diagnostics" not in out
