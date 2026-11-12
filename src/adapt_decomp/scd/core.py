"""SCD — Swarm-Contrastive Decomposition (the scd package) as a calibration method.

The shared Decomposition pipeline preprocesses, extends and whitens the EMG; SCD's own
search (swarm-optimised ICA with peel-off) then runs unmodified on the whitened EMG.
Each unit's separation vector is the whitened EMG averaged at its SCD spikes,
orthonormalised since adaptation keeps separation vectors orthonormal, and the sources
are reconstructed on the unpeeled whitened EMG, which is what adaptation sees online.
SCD's own filters, sources and spikes are kept as the native fields.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from loguru import logger

from adapt_decomp.adaptation.ops import orthonormalize_rows_symmetric
from adapt_decomp.decomposition.base import Decomposition
from adapt_decomp.scd.config import SCDConfig
from adapt_decomp.spikes import compare_paired_spikes, detect_spikes, spikes_dict_to_binary


def _import_scd() -> Tuple[type, Any, Any, Any]:
    """Import the optional scd package, which seeds torch and configures logging on import.

    Returns:
        Tuple[type, Any, Any, Any]: The SwarmContrastiveDecomposition class, scd.Config,
        scd.load_config and scd's set_random_seed.

    Raises:
        ImportError: If scd is not installed.
    """
    try:
        from scd.config.structures import Config, set_random_seed
        from scd.models.scd import SwarmContrastiveDecomposition
        from scd.train import load_config
    except ImportError as exc:
        raise ImportError('SCD needs the scd extra: pip install "adapt_decomp[scd]".') from exc
    return SwarmContrastiveDecomposition, Config, load_config, set_random_seed


class SCD(Decomposition):
    """Swarm-Contrastive Decomposition on the shared, already whitened EMG."""

    config_class = SCDConfig
    config: SCDConfig

    def _scd_config(self, config_cls: type, load_config: Any) -> Any:
        """Build scd's Config: preset, then scd_kwargs, then the wrapper-owned keys.

        Args:
            config_cls (type): scd.Config.
            load_config (Any): scd.load_config.

        Returns:
            Any: The scd.Config to run with.

        Raises:
            ValueError: If scd_kwargs names an option scd.Config does not have.
        """
        cfg = load_config(self.config.scd_preset) if self.config.scd_preset else config_cls()
        kwargs = self.config.scd_kwargs or {}
        unknown = sorted(set(kwargs) - {f.name for f in fields(cfg)})
        if unknown:
            raise ValueError(f"SCDConfig.scd_kwargs has unknown scd.Config options: {unknown}.")
        owned = {
            "sampling_frequency": round(self.config.fs),
            "start_time": 0,
            "end_time": -1,
            "time_differentiate": None,
            "notch_params": None,
            "low_pass_cutoff": None,
            "high_pass_cutoff": None,
            "extension_factor": 1,  # unused: the input is already extended and whitened
            "autocorrelation_whiten": False,
            "bad_channels": None,
            "square_sources_spike_det": True,
            "save_data": False,
            "output_source_plot": False,
            "output_final_source_plot": False,
            "verbose_mode": self.config.verbose,
            "device": str(self._device),
        }
        for key, value in {**kwargs, **owned}.items():
            setattr(cfg, key, value)
        return cfg

    def _extract(self, emg_wh: torch.Tensor, timestamps: torch.Tensor) -> Dict:
        """Run SCD's search, average the whitened EMG at its spikes, orthonormalise, and reconstruct.

        Args:
            emg_wh (torch.Tensor): Whitened, extended EMG with shape (samples, dim).
            timestamps (torch.Tensor): Sample times, in s, with shape (samples,).

        Returns:
            Dict: Per-unit fields plus the native ones; see Decomposition._extract().
        """
        model_cls, config_cls, load_config, set_random_seed = _import_scd()

        class _PrewhitenedSCD(model_cls):
            """SCD with identity preprocessing: Decomposition already whitened the EMG."""

            def preprocess_emg(self, emg: torch.Tensor) -> torch.Tensor:
                self.w_mat = torch.eye(emg.shape[1], dtype=emg.dtype)
                return emg

        samples, dim = emg_wh.shape
        scd_config = self._scd_config(config_cls, load_config)
        if self.config.random_seed is not None:
            set_random_seed(self.config.random_seed)  # after scd's import-time seeding
        # SCD peels units off its input in place: give it a copy
        _, out = _PrewhitenedSCD().run(emg_wh.clone(), scd_config)
        if not out["filters"]:
            logger.warning("SCD found no motor units.")
            return self._empty_result(dim)

        # Separation vectors: the whitened EMG averaged at each unit's native spikes
        filters = torch.from_numpy(np.concatenate(out["filters"], axis=1)).to(emg_wh)
        sta = torch.stack(
            [
                emg_wh[torch.as_tensor(idx, device=emg_wh.device, dtype=torch.long)].mean(dim=0)
                if len(idx)
                else filters[:, i]
                for i, idx in enumerate(out["timestamps"])
            ],
            dim=1,
        )

        # Orthonormal vectors and sources on the unpeeled EMG, as adaptation uses them
        sep_vectors = orthonormalize_rows_symmetric(sta.T).T
        sources = emg_wh @ sep_vectors
        units: List[Dict[str, Any]] = []
        for i in range(sources.shape[1]):
            spike_idx, spike_centr, base_centr, sil = detect_spikes(
                sources[:, i], self.config.spike_min_dist, peak_power=self.config.spike_det_exp
            )
            units.append(
                {
                    "spike_idx": spike_idx.cpu().numpy(),
                    "spike_centr": spike_centr,
                    "base_centr": base_centr,
                    "sil": sil,
                    "cov_isi": self._cov_isi_from_idx(spike_idx, timestamps),
                    "native_idx": out["timestamps"][i].cpu().numpy(),
                }
            )
        keep = [i for i, u in enumerate(units) if len(u["spike_idx"]) >= self.config.min_spikes]
        if not keep:
            logger.warning(f"No SCD unit has {self.config.min_spikes} spikes after reconstruction.")
            return self._empty_result(dim)

        spikes_dict = {new: units[i]["spike_idx"] for new, i in enumerate(keep)}
        native_dict = {new: units[i]["native_idx"] for new, i in enumerate(keep)}
        native_roa = compare_paired_spikes(
            spikes_dict_to_binary(native_dict, samples, device="cpu").numpy(),
            spikes_dict_to_binary(spikes_dict, samples, device="cpu").numpy(),
            fs=self.config.fs,
        ).roa
        idx = torch.as_tensor(keep, dtype=torch.long)
        return {
            "sources": sources[:, idx].cpu(),
            "spikes_dict": spikes_dict,
            "sil": torch.tensor([units[i]["sil"] for i in keep]),
            "cov_isi": torch.tensor([units[i]["cov_isi"] for i in keep]),
            "sep_vectors": sep_vectors[:, idx].cpu(),
            "spikes_centr": torch.tensor(
                [units[i]["spike_centr"] for i in keep], dtype=self._dtype
            ),
            "base_centr": torch.tensor([units[i]["base_centr"] for i in keep], dtype=self._dtype),
            "sep_vectors_native": filters[:, idx].cpu(),
            "sources_native": torch.from_numpy(
                np.concatenate(out["source"], axis=1)[:, keep]
            ).float(),
            "spikes_dict_native": native_dict,
            "native_roa": torch.from_numpy(native_roa).float(),
            "contrast_exp": torch.tensor([float(out["best_exp"][i]) for i in keep]),
        }
