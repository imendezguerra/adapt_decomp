"""Data structures for input EMG and the precalibrated decomposition model."""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from adapt_decomp.config import Config
from adapt_decomp.preprocessing import filter_kwargs
from adapt_decomp.preprocessing import preprocess_emg as preprocess_emg_fn


def _extend_data_v(data: torch.Tensor, ext_fact: int, device=None) -> torch.Tensor:
    """Build a time-delay-embedded matrix from raw EMG.

    Each of the ext_fact column blocks is the raw channel matrix shifted by one
    additional sample, giving shape [samples, channels * ext_fact].  The resulting
    D = channels * ext_fact dimensional space lets the whitening matrix V capture
    temporal correlations across ext_fact consecutive samples in a single linear
    projection.
    """
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
            emg = emg - offset
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
        """Shares preprocess_emg() with CBSS._preprocess_emg (calibration) so online
        adaptation sees EMG at the same scale/spectral content the whitening
        reference (Rz_cal) was computed from.
        """
        emg = preprocess_emg_fn(emg.cpu().numpy(), config.fs, **filter_kwargs(config))
        offset = np.mean(emg, axis=0)
        emg -= offset
        return torch.from_numpy(emg), torch.from_numpy(offset)

    def extend_data(self, emg: torch.Tensor, ext_fact: int) -> None:
        self.emg_ext = _extend_data_v(emg, ext_fact)


class Decomposition:
    """Precalibrated decomposition model with adaptive online state.

    Immutable calibration references (K_cal, kappa_cal, spike_centroids_cal,
    base_centroids_cal) are set once during init and never modified.

    Adaptive state (V, B, spike_centroids, base_centroids, fifo_cov, source_fifo)
    is updated per batch during online adaptation.
    """

    def __init__(
        self,
        V: torch.Tensor,          # whitening matrix [D, D]
        B: torch.Tensor,          # separation matrix [M, D]
        base_centroids: torch.Tensor,   # baseline centroids [M]
        spike_centroids: torch.Tensor,  # spike centroids [M]
        emg_calib: torch.Tensor,
        ipts_calib: torch.Tensor,
        spikes_calib: torch.Tensor,
        config: Optional[Config] = None,
    ) -> None:
        """Initialise the decomposition model from precalibrated matrices.

        Stores immutable calibration references (V, B, centroids, emg_calib,
        ipts_calib, spikes_calib) and calls init_wh_update, init_sv_update, and
        init_sd_update to compute the calibration statistics needed for online
        adaptation.  After construction, adaptive state (V, B, spike_centroids,
        base_centroids, fifo_cov, source_fifo) is ready for per-batch updates.
        """
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
        self.max_sigma_batches = config.max_sigma_batches
        self.peak_power = config.peak_power
        self.use_abs_for_detection = config.use_abs_for_detection
        self.eps = config.eps

        # --- Adaptive matrices ---
        self.V = V.to(dtype=torch.float32, device=self.device)
        self.B = B.to(dtype=torch.float32, device=self.device)

        # --- Immutable calibration centroid references ---
        self.spike_centroids_cal = spike_centroids.to(
            dtype=torch.float32, device=self.device
        )
        self.base_centroids_cal = base_centroids.to(
            dtype=torch.float32, device=self.device
        )

        # D: extended channel count (whitening space dimension)
        self.D = self.V.shape[0]
        self.I = torch.eye(self.D, dtype=torch.float32, device=self.device)

        # Calibration data (kept for init_* recomputation and reset)
        self.emg_calib = emg_calib.to(dtype=torch.float32)
        self.ipts_calib = ipts_calib.to(dtype=torch.float32)
        self.spikes_calib = spikes_calib.to(dtype=torch.int32)

        expected_D = self.emg_calib.shape[-1] * self.ext_fact
        if expected_D != self.D:
            raise ValueError(
                f"ext_fact mismatch: Config.ext_fact={self.ext_fact} on "
                f"{self.emg_calib.shape[-1]} calibration channels gives an extended "
                f"dimension of {expected_D}, but the whitening matrix V has dimension "
                f"{self.D}. Check that Config.ext_fact matches the CBSSConfig.ext_fact "
                "used to produce this calibration."
            )

        self.init_wh_update()
        self.init_sv_update()
        self.init_sd_update()

    # ------------------------------------------------------------------
    # Whitening state initialisation
    # ------------------------------------------------------------------

    def init_wh_update(self) -> None:
        """Compute K_cal and initialise the extended-EMG FIFO buffer.

        K_cal is estimated as the mean KL divergence over FIFO-sized sliding windows
        of calibration data (stride = batch_size), matching the finite-sample estimation
        regime of online adaptation. This removes the systematic bias that arises when
        comparing a full-dataset K_cal against a FIFO-estimated K_online.

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

        # Full-dataset Rz_cal is needed only when:
        #   kl_to_cal  — requires Rz_cal_inv and logdet_cal every batch
        #   wh_trace_renorm — requires trace_cal for scale correction
        # For the default config (kl_to_identity + wh_trace_renorm=False) both outputs
        # are unused: K_cal is overridden by the sigma loop mean below, and trace_cal is
        # never read. Skipping saves the two dominant [N_cal, D]@[D, D] matmuls (~40% of init).
        needs_full_rz = self.wh_mode == "kl_to_cal"
        if needs_full_rz:
            Z_cal  = X_cal @ self.V.T
            N      = Z_cal.shape[0]
            Rz_cal = (Z_cal.T @ Z_cal) / N
            Rz_cal = 0.5 * (Rz_cal + Rz_cal.T)
            Rz_cal = (1 - self.shrinkage) * Rz_cal + self.shrinkage * self.I
            sign, logdet = torch.linalg.slogdet(Rz_cal)
            self.trace_cal = Rz_cal.trace()
            if sign > 0:
                self.K_cal = 0.5 * (Rz_cal.trace() - logdet - self.D)
                if self.wh_mode == "kl_to_cal":
                    self.Rz_cal_inv = torch.linalg.inv(Rz_cal)
                    self.logdet_cal = logdet
            else:
                self.K_cal = torch.zeros(1, device=self.device).squeeze()
                if self.wh_mode == "kl_to_cal":
                    self.Rz_cal_inv = self.I.clone()
                    self.logdet_cal = torch.zeros(1, device=self.device).squeeze()
        else:
            self.K_cal = torch.zeros(1, device=self.device).squeeze()   # overridden below
            self.trace_cal = torch.tensor(float(self.D), device=self.device)  # target trace(I) = D

        # Compute sigma_K_cal: std of batch-wise K_online over the calibration recording.
        # Strategy: enumerate all valid window starts (stride = batch_size, width = fifo_samples),
        # subsample uniformly to max_sigma_batches, then process in chunks via batched matmul
        # and batched slogdet. Eliminates the serial FIFO loop and all torch.cat allocations.
        # Stays on self.device — MPS/CUDA promotion was tested and found counterproductive
        # for typical window counts (~300) due to transfer overhead dominating slogdet savings.
        all_starts = torch.arange(
            0, X_cal.shape[0] - self.fifo_samples + 1, self.batch_size, device=self.device
        )
        if self.max_sigma_batches > 0 and len(all_starts) > self.max_sigma_batches:
            sel = torch.linspace(0, len(all_starts) - 1, self.max_sigma_batches,
                                 dtype=torch.long, device=self.device)
            starts = all_starts[sel]
        else:
            starts = all_starts

        _chunk = 32
        _arange = torch.arange(self.fifo_samples, device=self.device)
        _K_chunks: list[torch.Tensor] = []

        for c in range(0, len(starts), _chunk):
            s = starts[c : c + _chunk]                              # [cs]
            X_w = X_cal[s[:, None] + _arange[None, :]]             # [cs, fifo_samples, D]
            X_w = X_w - X_w.mean(1, keepdim=True)
            Z_w = X_w @ self.V.T                                    # [cs, fifo_samples, D]
            Rz = torch.bmm(Z_w.permute(0, 2, 1), Z_w) / self.fifo_samples  # [cs, D, D]
            Rz = 0.5 * (Rz + Rz.permute(0, 2, 1))
            Rz = (1 - self.shrinkage) * Rz + self.shrinkage * self.I
            signs, logdets = torch.linalg.slogdet(Rz)              # [cs]
            valid = signs > 0
            if not valid.any():
                continue
            Rz_v, ld_v = Rz[valid], logdets[valid]
            if self.wh_mode == "kl_to_identity":
                tr = Rz_v.diagonal(dim1=-2, dim2=-1).sum(-1)
                _K_chunks.append(0.5 * (tr - ld_v - self.D))
            else:  # kl_to_cal
                A = self.Rz_cal_inv @ Rz_v                          # [D,D] @ [n,D,D]
                tr_A = A.diagonal(dim1=-2, dim2=-1).sum(-1)
                _K_chunks.append(0.5 * (tr_A - (ld_v - self.logdet_cal) - self.D))

        if _K_chunks:
            _K_t = torch.cat(_K_chunks)
            # K_cal set to batch-wise mean so K_online and K_cal share the same
            # finite-sample estimator — ensures e_v_raw is centred at 0 at calibration.
            self.K_cal = _K_t.mean().to(self.device)
            self.sigma_K_cal = _K_t.std().clamp_min(1e-7).to(self.device)
        else:
            self.sigma_K_cal = torch.tensor(1e-7, device=self.device)

        # EMA of ||direction @ V|| used to normalize the whitening natural-gradient
        # direction to unit scale (see _update_V). None = not yet seeded; the first
        # online batch seeds it directly from its own value rather than from a
        # calibration-time sweep. Reset here (called by __init__ and _reset_params)
        # so each fresh Optuna trial starts cold, not carrying over EMA state.
        self.ema_dirnorm_v: Optional[torch.Tensor] = None

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
        M = ipts.shape[1]

        # kappa_cal and sigma_kappa_cal use the same batch-wise estimator as the online
        # step so that e_b_raw = kappa_online - kappa_cal is centred at 0 under calibration.
        # Compute sigma_kappa_cal: std of batch-wise kappa over the calibration recording.
        # batch_based: vectorised reshape; spike_based: loop (irregular masks, cheap).
        if self.contrast_scope == "batch_based":
            n_full = (ipts.shape[0] // self.batch_size) * self.batch_size
            if n_full >= 2 * self.batch_size:
                ipts_b = ipts[:n_full].reshape(-1, self.batch_size, M)
                batch_kappas = log_cosh(ipts_b).mean(dim=1)   # [n_batches, M]
                self.kappa_cal = batch_kappas.mean(dim=0)
                self.sigma_kappa_cal = batch_kappas.std(dim=0).clamp_min(1e-7)
            else:
                self.kappa_cal = log_cosh(ipts).mean(dim=0)   # fallback: full-dataset
                self.sigma_kappa_cal = torch.full((M,), 1e-7, device=self.device)
        else:
            batch_kappas = []
            # Seed kappa_b fallback with full-dataset spike-based estimate
            kappa_seed = torch.zeros(M, device=self.device)
            spikes = self.spikes_calib.to(self.device)
            for j in range(M):
                ipts_j = ipts[spikes[:, j] == 1, j]
                kappa_seed[j] = log_cosh(ipts_j).mean() if ipts_j.numel() > 0 else 0.0

            spikes_cal = self.spikes_calib.to(self.device)
            for i in range(0, ipts.shape[0], self.batch_size):
                batch = ipts[i : i + self.batch_size]
                spikes_b = spikes_cal[i : i + self.batch_size]
                kappa_b = kappa_seed.clone()
                for j in range(M):
                    ipts_j = batch[spikes_b[:, j] == 1, j]
                    if ipts_j.numel() > 0:
                        kappa_b[j] = log_cosh(ipts_j).mean()
                batch_kappas.append(kappa_b)

            if len(batch_kappas) >= 2:
                _kappa_t = torch.stack(batch_kappas)
                self.kappa_cal = _kappa_t.mean(dim=0)
                self.sigma_kappa_cal = _kappa_t.std(dim=0).clamp_min(1e-7)
            else:
                self.kappa_cal = kappa_seed   # fallback: full-dataset estimate
                self.sigma_kappa_cal = torch.full((M,), 1e-7, device=self.device)

        # EMA of ||grad_B_row|| (per unit) used to normalize the B natural-gradient
        # direction to unit scale (see update_B_spike_gated). None = not yet seeded;
        # the first online batch seeds each unit directly from its own value. Reset
        # here (called by __init__ and _reset_params) so each fresh Optuna trial
        # starts cold.
        self.ema_gradnorm_b: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Spike detection state initialisation / reset
    # ------------------------------------------------------------------

    def init_sd_update(self) -> None:
        """Initialise adaptive centroids from calibration references, reset source FIFO,
        and compute IQR-gate statistics (Q75_cal, IQR_cal) in detection domain."""
        self.spike_centroids = self.spike_centroids_cal.clone()
        self.base_centroids = self.base_centroids_cal.clone()
        self.source_fifo: Optional[torch.Tensor] = None

        ipts   = self.ipts_calib.to(self.device)    # [N_cal, M]
        spikes = self.spikes_calib.to(self.device)  # [N_cal, M] int32

        Y_det = ipts.abs().pow(self.peak_power) if self.use_abs_for_detection else ipts.pow(self.peak_power)

        M = ipts.shape[1]
        Q75_cal = torch.zeros(M, dtype=torch.float32, device=self.device)
        IQR_cal = torch.zeros(M, dtype=torch.float32, device=self.device)

        _min_spikes_for_iqr = 4

        for j in range(M):
            spike_amps = Y_det[spikes[:, j] == 1, j]
            if spike_amps.numel() >= _min_spikes_for_iqr:
                q75 = torch.quantile(spike_amps, 0.75)
                q25 = torch.quantile(spike_amps, 0.25)
                Q75_cal[j] = q75
                IQR_cal[j] = q75 - q25
            else:
                # Fallback: centroid-gap heuristic for units with sparse calibration spikes
                warnings.warn(
                    f"Unit {j}: fewer than {_min_spikes_for_iqr} calibration spikes; "
                    "IQR gate using centroid-gap heuristic.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                Q75_cal[j] = self.spike_centroids_cal[j]
                IQR_cal[j] = (self.spike_centroids_cal[j] - self.base_centroids_cal[j]).clamp_min(self.eps)

        self.Q75_cal = Q75_cal  # [M]
        self.IQR_cal = IQR_cal  # [M]
