"""Data structures for input EMG and the precalibrated decomposition model."""

import warnings
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from adapt_decomp.config import Config
from adapt_decomp.preprocessing import filter_kwargs
from adapt_decomp.preprocessing import preprocess_emg as preprocess_emg_fn


def _extend_data_wh(
    data: torch.Tensor,
    ext_fact: int,
    device=None,
    ext_mode: Literal["block", "toeplitz"] = "block",
) -> torch.Tensor:
    """Build a time-delay-embedded matrix from raw EMG.

    Each of the ext_fact column blocks is the raw channel matrix shifted by one
    additional sample, giving shape [samples, channels * ext_fact].  The resulting
    D = channels * ext_fact dimensional space lets the whitening matrix wh capture
    temporal correlations across ext_fact consecutive samples in a single linear
    projection.

    ext_mode:
        "block"    (default) — column block i holds ALL channels shifted by i.
        "toeplitz" — each channel's own ext_fact delayed copies are kept
                   together, so each channel's block of columns is itself a
                   Toeplitz (constant-diagonal) matrix.
    """
    if device is None:
        device = data.device
    samples, chs = data.shape
    data_ext = torch.zeros((samples, int(chs * ext_fact)), device=device)
    for i in range(ext_fact):
        data_ext[i:samples, chs * i: chs * (i + 1)] = data[0:(samples - i), :]
    if ext_mode == "toeplitz":
        data_ext = (
            data_ext.view(samples, ext_fact, chs)
            .permute(0, 2, 1)
            .reshape(samples, chs * ext_fact)
        )
    elif ext_mode != "block":
        raise ValueError(
            f"Unknown ext_mode: {ext_mode!r}. Expected 'block' or 'toeplitz'."
        )
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
        self.extend_data(emg, config.ext_fact, config.ext_mode)

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

    def extend_data(
        self,
        emg: torch.Tensor,
        ext_fact: int,
        ext_mode: Literal["block", "toeplitz"] = "block",
    ) -> None:
        self.emg_ext = _extend_data_wh(emg, ext_fact, ext_mode=ext_mode)


class Decomposition:
    """Precalibrated decomposition model with adaptive online state.

    Immutable calibration references (kl_div_calib_mean, contrast_calib_mean,
    spikes_centr_cal, base_centr_cal) are set once during init and never modified.

    Adaptive state (whitening, sep_vectors, spikes_centr, base_centr, fifo_cov,
    source_fifo) is updated per batch during online adaptation.
    """

    def __init__(
        self,
        whitening: torch.Tensor,   # whitening matrix [D, D] (or [n, D] when pca_components is set)
        sep_vectors: torch.Tensor, # separation matrix [M, D] (or [M, n] when pca_components is set)
        base_centr: torch.Tensor,  # baseline centroids [M]
        spikes_centr: torch.Tensor,  # spike centroids [M]
        emg_calib: torch.Tensor,
        ipts_calib: torch.Tensor,
        spikes_calib: torch.Tensor,
        config: Optional[Config] = None,
        pca_components: Optional[torch.Tensor] = None,    # [n, D] sklearn PCA.components_ convention
        pca_mean: Optional[torch.Tensor] = None,          # [D]
    ) -> None:
        """Initialise the decomposition model from precalibrated matrices.

        Stores immutable calibration references (whitening, sep_vectors, centroids,
        emg_calib, ipts_calib, spikes_calib) and calls init_wh_update, init_sv_update,
        and init_sd_update to compute the calibration statistics needed for online
        adaptation.  After construction, adaptive state (whitening, sep_vectors,
        spikes_centr, base_centr, fifo_cov, source_fifo) is ready for per-batch updates.

        pca_components/pca_mean (from CBSSResult, when CBSSConfig.n_components was
        set): if given, every extended-EMG batch is projected through this fitted
        PCA transform -- (X_ext - pca_mean) @ pca_components.T -- immediately after
        per-batch centering and before whitening is applied (see _apply_pca and its
        call sites in _update_wh/init_wh_update), matching the CBSS calibration
        pipeline's own center -> extend -> PCA -> whiten order. whitening/sep_vectors
        must then be dimensioned for the PCA-reduced space (n), not the raw extended
        space (D = channels*ext_fact).
        """
        if config is None:
            config = Config()

        self.device = config.device
        self.ext_fact = config.ext_fact
        self.ext_mode = config.ext_mode
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
        self.whitening = whitening.to(dtype=torch.float32, device=self.device)
        self.sep_vectors = sep_vectors.to(dtype=torch.float32, device=self.device)

        # --- Immutable PCA projection (None = identity, i.e. no reduction) ---
        self.pca_components = (
            pca_components.to(dtype=torch.float32, device=self.device)
            if pca_components is not None else None
        )
        self.pca_mean = (
            pca_mean.to(dtype=torch.float32, device=self.device)
            if pca_mean is not None else None
        )

        # --- Immutable calibration centroid references ---
        self.spikes_centr_cal = spikes_centr.to(
            dtype=torch.float32, device=self.device
        )
        self.base_centr_cal = base_centr.to(
            dtype=torch.float32, device=self.device
        )

        # n: whitening-space dimension (extended channel count, or PCA-reduced
        # count when pca_components is set)
        self.n = self.whitening.shape[0]
        self.I = torch.eye(self.n, dtype=torch.float32, device=self.device)

        # Calibration data (kept for init_* recomputation and reset)
        self.emg_calib = emg_calib.to(dtype=torch.float32)
        self.ipts_calib = ipts_calib.to(dtype=torch.float32)
        self.spikes_calib = spikes_calib.to(dtype=torch.int32)

        expected_ext_D = self.emg_calib.shape[-1] * self.ext_fact
        if self.pca_components is not None:
            if self.pca_components.shape[1] != expected_ext_D:
                raise ValueError(
                    f"pca_components mismatch: Config.ext_fact={self.ext_fact} on "
                    f"{self.emg_calib.shape[-1]} calibration channels gives an extended "
                    f"dimension of {expected_ext_D}, but pca_components' input dimension "
                    f"is {self.pca_components.shape[1]}. Check that Config.ext_fact matches "
                    "the CBSSConfig.ext_fact used to produce this calibration."
                )
            if self.pca_components.shape[0] != self.n:
                raise ValueError(
                    f"pca_components output dimension ({self.pca_components.shape[0]}) "
                    f"doesn't match the whitening matrix dimension ({self.n})."
                )
        elif expected_ext_D != self.n:
            raise ValueError(
                f"ext_fact mismatch: Config.ext_fact={self.ext_fact} on "
                f"{self.emg_calib.shape[-1]} calibration channels gives an extended "
                f"dimension of {expected_ext_D}, but the whitening matrix has dimension "
                f"{self.n}. Check that Config.ext_fact matches the CBSSConfig.ext_fact "
                "used to produce this calibration."
            )
        # NOTE: ext_mode ("block" vs "toeplitz") has no shape signature to
        # validate against -- both produce the same D = channels*ext_fact width,
        # just with columns in a different order. Config.ext_mode must be
        # set to match the CBSSConfig.ext_mode used to produce this
        # calibration's whitening/sep_vectors, or online adaptation will silently
        # apply a mis-ordered whitening matrix.

        self.init_wh_update()
        self.init_sv_update()
        self.init_sd_update()

    def _apply_pca(self, X_ext: torch.Tensor) -> torch.Tensor:
        """Project a (centered) extended-EMG batch through the fitted PCA transform.

        Identity when pca_components is None. Must be called after per-batch/
        per-window centering and before whitening (self.whitening is dimensioned
        for the PCA-reduced space whenever pca_components is set).
        """
        if self.pca_components is None:
            return X_ext
        return (X_ext - self.pca_mean) @ self.pca_components.T

    # ------------------------------------------------------------------
    # Whitening state initialisation
    # ------------------------------------------------------------------

    def init_wh_update(self) -> None:
        """Compute K_cal and initialise the extended-EMG FIFO buffer.

        K_cal is estimated as the mean KL divergence over FIFO-sized sliding windows
        of calibration data (stride = batch_size), matching the finite-sample estimation
        regime of online adaptation. This removes the systematic bias that arises when
        comparing a full-dataset K_cal against a FIFO-estimated K_online.

        The FIFO stores raw extended EMG. At each online step the current wh is
        applied to the FIFO to produce Rz, keeping Rz full-rank even when the
        batch is smaller than D.
        """
        X_cal = _extend_data_wh(
            self.emg_calib, self.ext_fact, ext_mode=self.ext_mode
        ).to(device=self.device)
        X_cal = X_cal - X_cal.mean(0, keepdim=True)
        X_cal = self._apply_pca(X_cal)

        # FIFO length: at least D to keep Rz full-rank; default = 2×D
        auto_fifo = 2 * self.n
        self.fifo_samples = max(self.n, self.fifo_length_cfg if self.fifo_length_cfg > 0 else auto_fifo)
        self.fifo_cov = X_cal[-self.fifo_samples:].clone()

        # Full-dataset Rz_cal is needed only for kl_to_cal mode (requires Rz_cal_inv
        # and logdet_cal every batch). For the default kl_to_identity mode it's
        # skipped: K_cal is overridden by the sigma loop mean below, and trace_cal
        # is set to the identity-target fallback (float(n)) instead of the
        # Rz_cal-derived value below -- both are still real, consumed values
        # (K_cal by _update_wh, trace_cal by run_optimisation's trace_check),
        # just computed more cheaply. Skipping saves the two dominant
        # [N_cal, D]@[D, D] matmuls (~40% of init).
        needs_full_rz = self.wh_mode == "kl_to_cal"
        if needs_full_rz:
            Z_cal  = X_cal @ self.whitening.T
            N      = Z_cal.shape[0]
            Rz_cal = (Z_cal.T @ Z_cal) / N
            Rz_cal = 0.5 * (Rz_cal + Rz_cal.T)
            Rz_cal = (1 - self.shrinkage) * Rz_cal + self.shrinkage * self.I
            sign, logdet = torch.linalg.slogdet(Rz_cal)
            self.trace_cal = Rz_cal.trace()
            if sign > 0:
                self.kl_div_calib_mean = 0.5 * (Rz_cal.trace() - logdet - self.n)
                if self.wh_mode == "kl_to_cal":
                    self.Rz_cal_inv = torch.linalg.inv(Rz_cal)
                    self.logdet_cal = logdet
            else:
                self.kl_div_calib_mean = torch.zeros(1, device=self.device).squeeze()
                if self.wh_mode == "kl_to_cal":
                    self.Rz_cal_inv = self.I.clone()
                    self.logdet_cal = torch.zeros(1, device=self.device).squeeze()
        else:
            self.kl_div_calib_mean = torch.zeros(1, device=self.device).squeeze()   # overridden below
            self.trace_cal = torch.tensor(float(self.n), device=self.device)  # target trace(I) = D

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
            Z_w = X_w @ self.whitening.T                                    # [cs, fifo_samples, D]
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
                _K_chunks.append(0.5 * (tr - ld_v - self.n))
            else:  # kl_to_cal
                A = self.Rz_cal_inv @ Rz_v                          # [D,D] @ [n,D,D]
                tr_A = A.diagonal(dim1=-2, dim2=-1).sum(-1)
                _K_chunks.append(0.5 * (tr_A - (ld_v - self.logdet_cal) - self.n))

        if _K_chunks:
            _K_t = torch.cat(_K_chunks)
            # K_cal set to batch-wise mean so K_online and K_cal share the same
            # finite-sample estimator — ensures e_wh_raw is centred at 0 at calibration.
            self.kl_div_calib_mean = _K_t.mean().to(self.device)
            self.kl_div_calib_std = _K_t.std().clamp_min(1e-7).to(self.device)
        else:
            self.kl_div_calib_std = torch.tensor(1e-7, device=self.device)

        # EMA of ||direction @ wh|| used to normalize the whitening natural-gradient
        # direction to unit scale (see _update_wh). None = not yet seeded; the first
        # online batch seeds it directly from its own value rather than from a
        # calibration-time sweep. Reset here (called by __init__ and _reset_params)
        # so each fresh Optuna trial starts cold, not carrying over EMA state.
        self.ema_dirnorm_wh: Optional[torch.Tensor] = None

    def _update_fifo_cov(self, emg_batch: torch.Tensor) -> None:
        """Push current batch into the extended-EMG FIFO, trimming to fifo_samples."""
        self.fifo_cov = torch.cat([self.fifo_cov, emg_batch], dim=0)[-self.fifo_samples:]

    def _compute_Rz_from_fifo(self) -> torch.Tensor:
        """Apply current wh to the FIFO and return the regularised whitened covariance.

        Reapplying wh each call ensures Rz reflects the latest whitening matrix
        rather than an outdated one stored in the FIFO.
        """
        X_fifo = self.fifo_cov - self.fifo_cov.mean(0, keepdim=True)
        Z_fifo = X_fifo @ self.whitening.T          # [fifo_samples, D]
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
        # step so that e_sv_raw = kappa_online - kappa_cal is centred at 0 under calibration.
        # Compute sigma_kappa_cal: std of batch-wise kappa over the calibration recording.
        # batch_based: vectorised reshape; spike_based: loop (irregular masks, cheap).
        if self.contrast_scope == "batch_based":
            n_full = (ipts.shape[0] // self.batch_size) * self.batch_size
            if n_full >= 2 * self.batch_size:
                ipts_b = ipts[:n_full].reshape(-1, self.batch_size, M)
                batch_kappas = log_cosh(ipts_b).mean(dim=1)   # [n_batches, M]
                self.contrast_calib_mean = batch_kappas.mean(dim=0)
                self.contrast_calib_std = batch_kappas.std(dim=0).clamp_min(1e-7)
            else:
                self.contrast_calib_mean = log_cosh(ipts).mean(dim=0)   # fallback: full-dataset
                self.contrast_calib_std = torch.full((M,), 1e-7, device=self.device)
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
                self.contrast_calib_mean = _kappa_t.mean(dim=0)
                self.contrast_calib_std = _kappa_t.std(dim=0).clamp_min(1e-7)
            else:
                self.contrast_calib_mean = kappa_seed   # fallback: full-dataset estimate
                self.contrast_calib_std = torch.full((M,), 1e-7, device=self.device)

        # EMA of ||grad_sv_row|| (per unit) used to normalize the sv natural-gradient
        # direction to unit scale (see update_sv_spike_gated). None = not yet seeded;
        # the first online batch seeds each unit directly from its own value. Reset
        # here (called by __init__ and _reset_params) so each fresh Optuna trial
        # starts cold.
        self.ema_gradnorm_sv: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Spike detection state initialisation / reset
    # ------------------------------------------------------------------

    def init_sd_update(self) -> None:
        """Initialise adaptive centroids from calibration references, reset source FIFO,
        and compute IQR-gate statistics (Q75_cal, IQR_cal) in detection domain."""
        self.spikes_centr = self.spikes_centr_cal.clone()
        self.base_centr = self.base_centr_cal.clone()
        self.source_fifo: Optional[torch.Tensor] = None

        ipts   = self.ipts_calib.to(self.device)    # [N_cal, M]
        spikes = self.spikes_calib.to(self.device)  # [N_cal, M] int32

        sources_det = ipts.abs().pow(self.peak_power) if self.use_abs_for_detection else ipts.pow(self.peak_power)

        M = ipts.shape[1]
        Q75_cal = torch.zeros(M, dtype=torch.float32, device=self.device)
        IQR_cal = torch.zeros(M, dtype=torch.float32, device=self.device)

        _min_spikes_for_iqr = 4

        for j in range(M):
            spike_amps = sources_det[spikes[:, j] == 1, j]
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
                Q75_cal[j] = self.spikes_centr_cal[j]
                IQR_cal[j] = (self.spikes_centr_cal[j] - self.base_centr_cal[j]).clamp_min(self.eps)

        self.Q75_cal = Q75_cal  # [M]
        self.IQR_cal = IQR_cal  # [M]
