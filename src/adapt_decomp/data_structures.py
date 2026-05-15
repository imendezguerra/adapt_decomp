"""Data structures for input EMG and the precalibrated decomposition model."""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from adapt_decomp.preprocessing import bandpass_filter, remove_powerline
from adapt_decomp.config import Config


def _extend_data_v(data: torch.Tensor, ext_fact: int, device=None) -> torch.Tensor:
    if device is None:
        device = data.device
    samples, chs = data.shape
    data_ext = torch.zeros((samples, int(chs * ext_fact)), device=device)
    for i in range(ext_fact):
        data_ext[i:samples, chs * i: chs * (i + 1)] = data[0:(samples - i), :]
    return data_ext


class Data(Dataset):
    """Dataset wrapper for extended, preprocessed EMG."""

    def __init__(
        self,
        emg: torch.Tensor,
        preprocess: Optional[bool] = True,
        config: Optional[Config] = None,
    ) -> None:
        if config is None:
            config = Config()

        if preprocess:
            emg, offset = self.preprocess_emg(emg, config)
        else:
            offset = emg.mean(axis=0)
        self.extend_data(emg, config.ext_fact)

        self.emg_ext = self.emg_ext.to(device=config.device, dtype=torch.float32)
        self.labels = torch.arange(emg.shape[0]).to(device=config.device)
        self.offset = offset.repeat(config.ext_fact).to(
            device=config.device, dtype=torch.float32
        )

    def __len__(self) -> int:
        return self.emg_ext.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.emg_ext[idx, :], self.labels[idx]

    def preprocess_emg(
        self, emg: torch.Tensor, config: Config
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emg = bandpass_filter(
            emg.cpu().numpy(),
            config.fs,
            cutoff=[config.lowcut, config.highcut],
            filtfilt=False,
        )
        if config.powerline:
            emg = remove_powerline(
                emg, config.fs, cutoff=config.powerline_freq, filtfilt=False
            )
        offset = np.mean(emg, axis=0)
        emg -= offset
        return torch.from_numpy(emg), torch.from_numpy(offset)

    def extend_data(self, emg: torch.Tensor, ext_fact: int) -> None:
        self.emg_ext = _extend_data_v(emg, ext_fact)


class Decomposition:
    """Precalibrated decomposition model with adaptive online state.

    Immutable calibration references (K_cal, kappa_cal, spike_centroid_cal,
    base_centroid_cal) are set once during init and never modified.

    Adaptive state (V, B, spike_centroid, base_centroid, fifo_cov, source_fifo)
    is updated per batch during online adaptation.
    """

    def __init__(
        self,
        V: torch.Tensor,          # whitening matrix [D, D]
        B: torch.Tensor,          # separation matrix [M, D]
        base_centr: torch.Tensor, # baseline centroids [M] — kept as arg name for compat
        spikes_centr: torch.Tensor,  # spike centroids [M]
        emg_calib: torch.Tensor,
        ipts_calib: torch.Tensor,
        spikes_calib: torch.Tensor,
        config: Optional[Config] = None,
    ) -> None:
        if config is None:
            config = Config()

        self.device = config.device
        self.ext_fact = config.ext_fact
        self.batch_size = config.batch_size
        self.shrinkage = config.shrinkage
        self.contrast_scope = config.contrast_scope
        self.fifo_length_cfg = config.fifo_length
        self.source_fifo_batches = config.source_fifo_batches
        self.wh_mode = config.wh_mode

        # --- Adaptive matrices ---
        self.V = V.to(dtype=torch.float32, device=self.device)
        self.B = B.to(dtype=torch.float32, device=self.device)

        # --- Immutable calibration centroid references ---
        self.spike_centroid_cal = spikes_centr.to(
            dtype=torch.float32, device=self.device
        )
        self.base_centroid_cal = base_centr.to(
            dtype=torch.float32, device=self.device
        )

        # D: extended channel count (whitening space dimension)
        self.D = self.V.shape[0]
        self.I = torch.eye(self.D, dtype=torch.float32, device=self.device)

        # Calibration data (kept for init_* recomputation and reset)
        self.emg_calib = emg_calib.to(dtype=torch.float32)
        self.ipts_calib = ipts_calib.to(dtype=torch.float32)
        self.spikes_calib = spikes_calib.to(dtype=torch.int32)

        self.init_wh_update()
        self.init_sv_update()
        self.init_sd_update()

    # ------------------------------------------------------------------
    # Whitening state initialisation
    # ------------------------------------------------------------------

    def init_wh_update(self) -> None:
        """Compute K_cal and initialise the extended-EMG FIFO buffer.

        K_cal is the KL-divergence-like contrast of the whitened calibration data
        against the identity covariance. It is immutable — computed once from the
        full calibration dataset (full rank) and used as the reference for online
        whitening updates.

        The FIFO stores raw extended EMG. At each online step the current V is
        applied to the FIFO to produce Rz, keeping Rz full-rank even when the
        batch is smaller than D.
        """
        X_cal = _extend_data_v(self.emg_calib, self.ext_fact).to(device=self.device)
        X_cal = X_cal - X_cal.mean(0, keepdim=True)

        # FIFO length: at least D to keep Rz full-rank; default = 2×D
        auto_fifo = 2 * self.D
        self.fifo_samples = max(self.D, self.fifo_length_cfg if self.fifo_length_cfg > 0 else auto_fifo)
        self.fifo_cov = X_cal[-self.fifo_samples:].clone()

        # Calibration whitened covariance (full rank — computed from full dataset)
        Z_cal = X_cal @ self.V.T           # [N, D]
        N = Z_cal.shape[0]
        Rz_cal = (Z_cal.T @ Z_cal) / N
        Rz_cal = 0.5 * (Rz_cal + Rz_cal.T)
        Rz_cal = (1 - self.shrinkage) * Rz_cal + self.shrinkage * self.I
        sign, logdet = torch.linalg.slogdet(Rz_cal)

        if sign > 0:
            # kl_to_identity: K_cal = KL(Rz_cal ‖ I) — immutable scalar reference
            self.K_cal = 0.5 * (Rz_cal.trace() - logdet - self.D)
            if self.wh_mode == "kl_to_cal":
                # Precompute Rz_cal⁻¹ and logdet_cal once; used every batch
                self.Rz_cal_inv = torch.linalg.inv(Rz_cal)
                self.logdet_cal  = logdet
        else:
            self.K_cal = torch.zeros(1, device=self.device).squeeze()
            if self.wh_mode == "kl_to_cal":
                self.Rz_cal_inv = self.I.clone()
                self.logdet_cal  = torch.zeros(1, device=self.device).squeeze()

    def _update_fifo_cov(self, emg_batch: torch.Tensor) -> None:
        """Push current batch into the extended-EMG FIFO, trimming to fifo_samples."""
        self.fifo_cov = torch.cat([self.fifo_cov, emg_batch], dim=0)[-self.fifo_samples:]

    def _compute_Rz_from_fifo(self) -> torch.Tensor:
        """Apply current V to the FIFO and return the regularised whitened covariance.

        Reapplying V each call ensures Rz reflects the latest whitening matrix
        rather than an outdated one stored in the FIFO.
        """
        X_fifo = self.fifo_cov - self.fifo_cov.mean(0, keepdim=True)
        Z_fifo = X_fifo @ self.V.T          # [fifo_samples, D]
        N = Z_fifo.shape[0]
        Rz = (Z_fifo.T @ Z_fifo) / N
        Rz = 0.5 * (Rz + Rz.T)
        Rz = (1 - self.shrinkage) * Rz + self.shrinkage * self.I
        return Rz

    # ------------------------------------------------------------------
    # Source contrast state initialisation
    # ------------------------------------------------------------------

    def init_sv_update(self) -> None:
        """Compute kappa_cal — the immutable calibration source contrast reference.

        Mirrors contrast_scope so calibration and online kappa are comparable:
          batch_based — log_cosh over all calibration IPT samples
          spike_based — log_cosh averaged only at spike times per source
        """
        from adapt_decomp.ops import log_cosh
        ipts = self.ipts_calib.to(self.device)

        if self.contrast_scope == "batch_based":
            self.kappa_cal = log_cosh(ipts).mean(dim=0)
        else:
            M = ipts.shape[1]
            kappa_cal = torch.zeros(M, device=self.device)
            spikes = self.spikes_calib.to(self.device)
            for j in range(M):
                ipts_j = ipts[spikes[:, j] == 1, j]
                kappa_cal[j] = log_cosh(ipts_j).mean() if ipts_j.numel() > 0 else 0.0
            self.kappa_cal = kappa_cal

    # ------------------------------------------------------------------
    # Spike detection state initialisation / reset
    # ------------------------------------------------------------------

    def init_sd_update(self) -> None:
        """Initialise adaptive centroids from calibration references and reset source FIFO."""
        self.spike_centroid = self.spike_centroid_cal.clone()
        self.base_centroid = self.base_centroid_cal.clone()
        # Source FIFO is initialised lazily on the first batch
        self.source_fifo: Optional[torch.Tensor] = None
