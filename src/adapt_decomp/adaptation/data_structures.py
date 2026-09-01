"""Data structures for input EMG, the precalibrated decomposition model, and
AdaptDecomp's typed output.
"""

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.ops import stable_cov
from adapt_decomp.preprocessing import extend_data as _extend_data
from adapt_decomp.preprocessing import filter_kwargs
from adapt_decomp.preprocessing import preprocess_emg as preprocess_emg_fn
from adapt_decomp.preprocessing import select_channels as select_channels_fn
from adapt_decomp.preprocessing import validate_channel_selection


class Data(Dataset):
    """Dataset wrapper for extended, preprocessed EMG."""

    def __init__(
        self,
        emg: torch.Tensor,
        preprocess: Optional[bool] = True,
        config: Optional[AdaptConfig] = None,
    ) -> None:
        """Extended and preprocessed EMG data with sample labels.

        Args:
            emg (torch.Tensor): Raw EMG data with shape (samples, channels).
            preprocess (Optional[bool], optional): Whether to filter emg and 
                apply channel selection beforev entering. When False, emg is
                only mean-centered instead. Defaults to True.
            config (Optional[AdaptConfig], optional): Online adaptation
                configuration. Defaults to None, which builds AdaptConfig().

        Raises:
            ValueError: If config.ext_mode is not "block" or "toeplitz",
                config.replace_bad_channels is True with config.ch_map
                unset, or config.ch_mask's length disagrees with emg's raw
                channel count -- see _select_channels.
        """
        if config is None:
            config = AdaptConfig()

        if preprocess:
            emg, offset = self.preprocess_emg(emg, config)
            emg, offset = self._select_channels(emg, offset, config)
        else:
            offset = emg.mean(axis=0)
            emg = emg - offset

        self.extend_data(emg, config.ext_fact, config.ext_mode)

        self.emg_ext = self.emg_ext.to(device=config.device, dtype=torch.float32)
        self.labels = torch.arange(emg.shape[0]).to(device=config.device)
    
        if config.ext_mode == "toeplitz":
            ext_offset = offset.repeat_interleave(config.ext_fact)
        elif config.ext_mode == "block":
            ext_offset = offset.repeat(config.ext_fact)
        else:
            raise ValueError(
                f"Unknown ext_mode: {config.ext_mode!r}. Expected 'block' or 'toeplitz'."
            )
        self.offset = ext_offset.to(device=config.device, dtype=torch.float32)

    def __len__(self) -> int:
        return self.emg_ext.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.emg_ext[idx, :], self.labels[idx]

    def preprocess_emg(
        self, emg: torch.Tensor, config: AdaptConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter and mean-centre emg via the shared preprocessing.preprocess_emg.

        Args:
            emg (torch.Tensor): Raw EMG data with shape (samples, channels).
            config (AdaptConfig): Online adaptation configuration; only the
                filtering fields (consumed via filter_kwargs) and config.fs
                are used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: emg, the filtered, mean-centered
            EMG with shape (samples, channels); and offset, the per-channel
            mean subtracted from it, with shape (channels,).
        """
        emg = preprocess_emg_fn(emg.cpu().numpy(), config.fs, **filter_kwargs(config))
        offset = np.mean(emg, axis=0)
        emg -= offset
        return torch.from_numpy(emg), torch.from_numpy(offset)

    def _select_channels(
        self, emg: torch.Tensor, offset: torch.Tensor, config: AdaptConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply bad-channel handling (drop or interpolate) via the shared helper.

        Args:
            emg (torch.Tensor): Filtered, mean-centred EMG with shape
                (samples, raw_channels).
            offset (torch.Tensor): Per-channel mean subtracted from emg,
                with shape (raw_channels,).
            config (AdaptConfig): Only config.ch_mask/.ch_map/
                .replace_bad_channels are used.

        Raises:
            ValueError: If config.replace_bad_channels is True with
                config.ch_map unset, or if config.ch_mask's length
                disagrees with emg's raw channel count.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: emg, unchanged (no-op or
            interpolated) with shape (samples, raw_channels), or dropped to
            (samples, good_channels); and offset, sliced to match emg's
            channel count if channels were dropped, else unchanged.
        """
        validate_channel_selection(
            config.ch_mask, config.ch_map, config.replace_bad_channels, emg.shape[1]
        )
        emg_np = select_channels_fn(
            emg.cpu().numpy(), config.ch_mask, config.ch_map, config.replace_bad_channels
        )
        emg = torch.from_numpy(emg_np)
        if emg.shape[1] != offset.shape[0]:
            offset = offset[torch.as_tensor(config.ch_mask)]
        return emg, offset

    def extend_data(
        self,
        emg: torch.Tensor,
        ext_fact: int,
        ext_mode: Literal["block", "toeplitz"] = "block",
    ) -> None:
        """Time-delay-embed emg via the shared preprocessing.extend_data.

        Args:
            emg (torch.Tensor): EMG data with shape (samples, channels).
            ext_fact (int): Extension factor.
            ext_mode (Literal["block", "toeplitz"], optional): Extension
                mode, must match the CBSSConfig.ext_mode used to produce the
                calibration this instance is paired with. Defaults to "block".

        Returns:
            None
        """
        self.emg_ext = _extend_data(emg, ext_fact, ext_mode=ext_mode)


class RawData(Dataset):
    """Dataset wrapper for raw, unpreprocessed EMG.

    Used for the streaming mode (data_preprocessed=False): process_batch
    filters/selects/centres/extends each batch itself, so this class does
    none of that -- it only serves raw rows. Kept separate from Data
    rather than added as a mode flag on it, since none of Data's
    preprocessing/extension methods apply here.
    """

    def __init__(self, emg: torch.Tensor, config: Optional[AdaptConfig] = None) -> None:
        """Wrap raw EMG for batch serving.

        Args:
            emg (torch.Tensor): Raw EMG data with shape (samples, channels).
            config (Optional[AdaptConfig], optional): Online adaptation
                configuration. Defaults to None, which builds AdaptConfig().
        """
        if config is None:
            config = AdaptConfig()
        self.emg = emg.to(device=config.device, dtype=torch.float32)
        self.labels = torch.arange(emg.shape[0]).to(device=config.device)

    def __len__(self) -> int:
        return self.emg.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.emg[idx, :], self.labels[idx]


class Decomposition:
    """Precalibrated decomposition model with adaptive online state."""

    def __init__(
        self,
        whitening: torch.Tensor,   
        sep_vectors: torch.Tensor, 
        base_centr: torch.Tensor,  
        spikes_centr: torch.Tensor,  
        emg_calib: torch.Tensor,
        ipts_calib: torch.Tensor,
        spikes_calib: torch.Tensor,
        config: Optional[AdaptConfig] = None,
        pca_components: Optional[torch.Tensor] = None,    
        pca_mean: Optional[torch.Tensor] = None,         
    ) -> None:
        """Initialise the decomposition model from precalibrated matrices.

        Args:
            whitening (torch.Tensor): Initial whitening matrix with shape
                (n, n), where n is the whitening-space dimension: the raw
                extended dimension D = channels * ext_fact when
                pca_components is None, or the PCA-reduced dimension when
                it is set.
            sep_vectors (torch.Tensor): Initial separation vectors with
                shape (n, M), where M is the number of motor units.
            base_centr (torch.Tensor): Calibration baseline centroids with
                shape (M,).
            spikes_centr (torch.Tensor): Calibration spike centroids with
                shape (M,).
            emg_calib (torch.Tensor): Raw, unextended calibration EMG with
                shape (N_cal, channels).
            ipts_calib (torch.Tensor): Calibration source signal with shape
                (N_cal, M).
            spikes_calib (torch.Tensor): Calibration binary spike train
                with shape (N_cal, M).
            config (Optional[AdaptConfig], optional): Online adaptation
                configuration. Defaults to None, which builds AdaptConfig().
            pca_components (Optional[torch.Tensor], optional): Fitted PCA
                components with shape (n, D). Defaults to None (no PCA reduction).
            pca_mean (Optional[torch.Tensor], optional): Fitted PCA mean with 
                shape (D,). Defaults to None.

        Raises:
            ValueError: If pca_components is set and its input/output
                dimensions do not match ext_fact/whitening, or if
                pca_components is None and channels * ext_fact does not
                match the whitening matrix dimension.

        Notes:
            pca_components/pca_mean (from CBSSResult, when CBSSConfig.n_components was
            set): if given, every extended-EMG batch is projected through this fitted
            PCA transform: (X_ext - pca_mean) @ pca_components.T following CBSS 
            pipeline: center -> extend -> PCA -> whiten order.
        """
        if config is None:
            config = AdaptConfig()

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
        self.eps = config.eps
        self.spike_det_exp = config.spike_det_exp

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

        # Check for correct dimensions if PCA model is passed
        expected_ext_D = self.emg_calib.shape[-1] * self.ext_fact
        if self.pca_components is not None:
            if self.pca_components.shape[1] != expected_ext_D:
                raise ValueError(
                    f"pca_components mismatch: AdaptConfig.ext_fact={self.ext_fact} on "
                    f"{self.emg_calib.shape[-1]} calibration channels gives an extended "
                    f"dimension of {expected_ext_D}, but pca_components' input dimension "
                    f"is {self.pca_components.shape[1]}. Check that AdaptConfig.ext_fact matches "
                    "the CBSSConfig.ext_fact used to produce this calibration."
                )
            if self.pca_components.shape[0] != self.n:
                raise ValueError(
                    f"pca_components output dimension ({self.pca_components.shape[0]}) "
                    f"doesn't match the whitening matrix dimension ({self.n})."
                )
        elif expected_ext_D != self.n:
            raise ValueError(
                f"ext_fact mismatch: AdaptConfig.ext_fact={self.ext_fact} on "
                f"{self.emg_calib.shape[-1]} calibration channels gives an extended "
                f"dimension of {expected_ext_D}, but the whitening matrix has dimension "
                f"{self.n}. Check that AdaptConfig.ext_fact matches the CBSSConfig.ext_fact "
                "used to produce this calibration."
            )

        self.init_wh_update()
        self.init_sv_update()
        self.init_sd_update()

        # Streaming (data_preprocessed=False) state, seeded lazily by
        # AdaptDecomp._preprocess_batch_raw on its first call.
        self.zi: Optional[list] = None
        self.ema_mean_online: Optional[torch.Tensor] = None
        self.ext_fifo: Optional[torch.Tensor] = None

    def _apply_pca(self, X_ext: torch.Tensor) -> torch.Tensor:
        """Apply PCA transform to data.

        Args:
            X_ext (torch.Tensor): Centered extended EMG with shape
                (samples, D).

        Returns:
            torch.Tensor: X_ext unchanged when pca_components is None,
            otherwise the PCA-reduced batch with shape (samples, n).
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

        Returns:
            None
        """
        # Extend, centre, and PCA-project (optional) the calibration EMG
        X_cal = self._build_calib_ext()

        # Build FIFO buffer with length at least D to keep Rz full-rank (default = 2×D)
        auto_fifo = 2 * self.n
        self.fifo_samples = max(self.n, self.fifo_length_cfg if (self.fifo_length_cfg is not None and self.fifo_length_cfg > 0) else auto_fifo)
        self.fifo_cov = X_cal[-self.fifo_samples:].clone()

        # Precompute inmutable variables for whitening loss and update
        self._compute_calib_kl_stats(X_cal)
        # Compute mean and std of kl divergence during calibration for loss normalisation
        self._compute_mean_sigma_kl_cal(X_cal)

        # EMA of ||direction @ wh|| used to normalize the whitening natural-gradient
        # direction to unit scale (see _update_wh). None = not yet seeded; the first
        # online batch seeds it directly from its own value.
        self.ema_dirnorm_wh: Optional[torch.Tensor] = None

    def _build_calib_ext(self) -> torch.Tensor:
        """Extend, centre, and PCA-project the calibration EMG.

        Returns:
            torch.Tensor: X_cal, the calibration data in whitening space,
            with shape (N_cal_ext, n).
        """
        # Extend, center, and optionally apply PCA
        X_cal = _extend_data(
            self.emg_calib, self.ext_fact, ext_mode=self.ext_mode
        ).to(device=self.device)
        X_cal = X_cal - X_cal.mean(0, keepdim=True)
        return self._apply_pca(X_cal)

    def _compute_calib_kl_stats(self, X_cal: torch.Tensor) -> None:
        """Set trace_cal/kl_div_calib_mean, and Rz_cal_inv/logdet_cal in kl_to_cal
        mode, from the full-dataset calibration whitened covariance.

        Args:
            X_cal (torch.Tensor): Calibration data with shape (N_cal_ext, n),
                as returned by _build_calib_ext.

        Returns:
            None
        """

        needs_full_rz = self.wh_mode == "kl_to_cal"
        if needs_full_rz:
            Z_cal  = X_cal @ self.whitening.T
            Rz_cal = stable_cov(Z_cal, rowvar=False, rho=self.shrinkage, I=self.I, ddof=0)
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

    def _compute_mean_sigma_kl_cal(self, X_cal: torch.Tensor) -> None:
        """Estimate kl_div_calib_mean/kl_div_calib_std from batch-wise K_online over
        sliding calibration windows (stride = batch_size, width = fifo_samples).

        Args:
            X_cal (torch.Tensor): Calibration data with shape (N_cal_ext, n),
                as returned by _build_calib_ext.

        Returns:
            None
        """
        # Use a max number of batches to compute statistics 
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
            # stable_cov batches over the leading [cs] dim (replaces the old torch.bmm).
            Rz = stable_cov(Z_w, rowvar=False, rho=self.shrinkage, I=self.I, ddof=0)  # [cs, D, D]
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
            self.kl_div_calib_mean = _K_t.mean().to(self.device)
        if _K_chunks and _K_t.numel() >= 2:
            self.kl_div_calib_std = _K_t.std().clamp_min(1e-7).to(self.device)
        else:
            # Fewer than 2 valid sliding windows (small calibration set relative
            # to fifo_samples/batch_size, or no window had a valid slogdet) --
            # std() needs >= 2 samples for its default (ddof=1) correction;
            # calling it on 0 or 1 elements raises a UserWarning and returns NaN.
            # Fall back to the eps floor used everywhere else in this file as a
            # "no meaningful spread" default.
            self.kl_div_calib_std = torch.tensor(1e-7, device=self.device)

    def _update_fifo_cov(self, emg_batch: torch.Tensor) -> None:
        """Push current batch into the extended-EMG FIFO, trimming to fifo_samples.

        Args:
            emg_batch (torch.Tensor): Extended EMG batch to append, with
                shape (batch, n).

        Returns:
            None
        """
        self.fifo_cov = torch.cat([self.fifo_cov, emg_batch], dim=0)[-self.fifo_samples:]

    def _compute_Rz_from_fifo(self) -> torch.Tensor:
        """Apply current wh to the FIFO and return the regularised whitened covariance.

        Reapplying wh each call ensures Rz reflects the latest whitening matrix
        rather than an outdated one stored in the FIFO.

        Returns:
            torch.Tensor: Regularised whitened covariance Rz with shape
            (n, n).
        """
        X_fifo = self.fifo_cov - self.fifo_cov.mean(0, keepdim=True)
        Z_fifo = X_fifo @ self.whitening.T          # [fifo_samples, D]
        return stable_cov(Z_fifo, rowvar=False, rho=self.shrinkage, I=self.I, ddof=0)

    # ------------------------------------------------------------------
    # Source contrast state initialisation
    # ------------------------------------------------------------------

    def init_sv_update(self) -> None:
        """Compute contrast values mean and std per source to ground the
        separation vector update and loss normalisation.

        Mirrors contrast_scope so calibration and online kappa are comparable:
          batch_based — log_cosh over all calibration IPT samples
          spike_based — log_cosh averaged only at spike times per source

        Returns:
            None
        """
        ipts = self.ipts_calib.to(self.device)

        # Compute contrast values during calibration per batch or spikes during calibration 
        if self.contrast_scope == "batch_based":
            self.contrast_calib_mean, self.contrast_calib_std = self._compute_kappa_batch_based(ipts)
        else:
            self.contrast_calib_mean, self.contrast_calib_std = self._compute_kappa_spike_based(ipts)

        # EMA of ||grad_sv_row|| (per unit) used to normalize the sv natural-gradient
        # direction to unit scale (see update_sv_spike_gated). None = not yet seeded;
        # the first online batch seeds each unit directly from its own value. 
        self.ema_gradnorm_sv: Optional[torch.Tensor] = None

    def _compute_kappa_batch_based(self, ipts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorised batch-wise kappa mean/std over reshaped fixed-size batches.

        Args:
            ipts (torch.Tensor): Calibration source signal with shape
                (N_cal, M).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: contrast_calib_mean and
            contrast_calib_std, each with shape (M,).
        """
        from adapt_decomp.cbss.ica import log_cosh
        M = ipts.shape[1]
        n_full = (ipts.shape[0] // self.batch_size) * self.batch_size
        if n_full >= 2 * self.batch_size:
            ipts_b = ipts[:n_full].reshape(-1, self.batch_size, M)
            batch_kappas = log_cosh(ipts_b).mean(dim=1)   # [n_batches, M]
            contrast_calib_mean = batch_kappas.mean(dim=0)
            contrast_calib_std = batch_kappas.std(dim=0).clamp_min(1e-7)
        else:
            contrast_calib_mean = log_cosh(ipts).mean(dim=0)   # fallback: full-dataset
            contrast_calib_std = torch.full((M,), 1e-7, device=self.device)
        return contrast_calib_mean, contrast_calib_std

    def _compute_kappa_spike_based(self, ipts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-unit, spike-masked kappa mean/std via a per-batch loop (irregular masks).

        Args:
            ipts (torch.Tensor): Calibration source signal with shape
                (N_cal, M).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: contrast_calib_mean and
            contrast_calib_std, each with shape (M,).
        """
        from adapt_decomp.cbss.ica import log_cosh
        M = ipts.shape[1]
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
            contrast_calib_mean = _kappa_t.mean(dim=0)
            contrast_calib_std = _kappa_t.std(dim=0).clamp_min(1e-7)
        else:
            contrast_calib_mean = kappa_seed   # fallback: full-dataset estimate
            contrast_calib_std = torch.full((M,), 1e-7, device=self.device)
        return contrast_calib_mean, contrast_calib_std

    # ------------------------------------------------------------------
    # Spike detection state initialisation / reset
    # ------------------------------------------------------------------

    def init_sd_update(self) -> None:
        """Initialise adaptive centroids from calibration references, reset source FIFO,
        and compute IQR-gate statistics (Q75_cal, IQR_cal) in detection domain.

        Returns:
            None
        """
        self.spikes_centr = self.spikes_centr_cal.clone()
        self.base_centr = self.base_centr_cal.clone()
        self.source_fifo: Optional[torch.Tensor] = None

        ipts   = self.ipts_calib.to(self.device)    # [N_cal, M]
        spikes = self.spikes_calib.to(self.device)  # [N_cal, M] int32

        sources_det = ipts.abs().pow(self.spike_det_exp)

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


@dataclass
class AdaptationResult:
    """Per-batch outputs of a full AdaptDecomp.process_data() over an online recording.

    Structural sibling of adapt_decomp.cbss.data_structure.CBSSResult (same
    to_dict()-for-IO convention), but not the same class -- adaptation results
    and calibration results have different shapes and don't need to interoperate
    beyond AdaptDecomp.from_calibration() reading a CBSSResult's fields.

    Attributes:
        spikes (torch.Tensor): Binary spike train with shape (samples, M),
            int32.
        ipts (torch.Tensor): Source signal (before the sv update, so outputs
            are consistent across batches) with shape (samples, M), float32.
        wh_time_ms (torch.Tensor): Per-batch whitening step time in ms, with
            shape (batches,).
        sv_time_ms (torch.Tensor): Per-batch separation-vector step time in
            ms, with shape (batches,).
        sd_time_ms (torch.Tensor): Per-batch spike-detection step time in ms,
            with shape (batches,).
        preprocess_time_ms (torch.Tensor): Per-batch preprocessing step time
            in ms, with shape (batches,). Zero when data_preprocessed is
            True.
        total_time_ms (torch.Tensor): Per-batch total step time in ms, with
            shape (batches,).
        wh_loss (Optional[torch.Tensor]): Whitening loss with shape
            (batches,). Set only when config.compute_loss is True.
        sv_loss (Optional[torch.Tensor]): Separation-vector contrast loss
            with shape (batches, M). Set only when config.compute_loss is True.
        centroid_loss (Optional[torch.Tensor]): Centroid-separation loss with
            shape (batches, M). Set only when config.compute_loss is True.
        wh_trace (Optional[torch.Tensor]): Trace of the whitened covariance
            with shape (batches,). Set only when config.compute_loss is True.
        wh_loss_median (Optional[torch.Tensor]): Guarded scalar median(wh_loss)
            for the whole run -- see AdaptDecomp._compute_losses(). Set only
            when config.compute_loss is True.
        sv_loss_median (Optional[torch.Tensor]): Guarded scalar nanmedian(sv_loss)
            for the whole run -- see AdaptDecomp._compute_losses(). Set only
            when config.compute_loss is True.
        total_loss (Optional[torch.Tensor]): Guarded scalar score for the
            whole run -- wh_loss_median + sv_loss_median, see
            AdaptDecomp._compute_losses(). Set only when config.compute_loss
            is True.
        diagnostics (Optional[Dict[Any, Any]]): Per-batch diagnostic tensors,
            keyed by batch index. Set only when config.debug is True.
        gt_matched_indices (Optional[np.ndarray]): Index into a ground-truth
            unit set for each unit, with shape (M,). Set only when the
            instance was built via AdaptDecomp.from_calibration() with a
            supervised calibration.
        roa (Optional[np.ndarray]): Rate of agreement against a ground-truth
            spike train, with shape (M,). Mirrors
            adapt_decomp.cbss.data_structure.CBSSResult.roa's convention
            (per-unit, not built here) -- callers set it after computing
            their own comparison (e.g.
            adapt_decomp.spikes.comparison.rate_of_agreement_paired), such
            as adaptation/optimize.py's optimize_adapt_decomp_pooled_memory(compute_roa=True).
    """

    spikes: torch.Tensor
    ipts: torch.Tensor
    preprocess_time_ms: torch.Tensor
    wh_time_ms: torch.Tensor
    sv_time_ms: torch.Tensor
    sd_time_ms: torch.Tensor
    total_time_ms: torch.Tensor
    wh_loss: Optional[torch.Tensor] = None
    sv_loss: Optional[torch.Tensor] = None
    centroid_loss: Optional[torch.Tensor] = None
    wh_trace: Optional[torch.Tensor] = None
    wh_loss_median: Optional[torch.Tensor] = None
    sv_loss_median: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    diagnostics: Optional[Dict[Any, Any]] = None
    gt_matched_indices: Optional[np.ndarray] = None
    roa: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict, omitting fields that are still None.

        Preserves the exact dict shape AdaptDecomp.format_outputs() returned
        before AdaptationResult existed, so io.H5ParamsBatchWriter/
        load_output need no changes.

        Returns:
            Dict[str, Any]: Mapping of field name to value, for every field
            that is not None.
        """
        out: Dict[str, Any] = {
            "spikes": self.spikes,
            "ipts": self.ipts,
            "preprocess_time_ms": self.preprocess_time_ms,
            "wh_time_ms": self.wh_time_ms,
            "sv_time_ms": self.sv_time_ms,
            "sd_time_ms": self.sd_time_ms,
            "total_time_ms": self.total_time_ms,
        }
        for key in (
            "wh_loss", "sv_loss", "centroid_loss", "wh_trace",
            "wh_loss_median", "sv_loss_median", "total_loss",
            "diagnostics", "gt_matched_indices", "roa",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Pickle this result to disk.

        Args:
            path (Union[str, Path]): Destination file path.

        Returns:
            None
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AdaptationResult":
        """Load an AdaptationResult previously written by save().

        Args:
            path (Union[str, Path]): Path to a pickle file written by save().

        Returns:
            AdaptationResult: The unpickled result.

        Raises:
            ValueError: If the unpickled object is not an AdaptationResult.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(
                f"{path} does not contain an AdaptationResult (got {type(obj).__name__})."
            )
        return obj

    def __getitem__(self, key: str) -> Any:
        """Dict-style subscript access, delegating to to_dict().

        Args:
            key (str): Field name.

        Returns:
            Any: The field's value.

        Raises:
            KeyError: If key is not a set field.
        """
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        """Check whether key is a set (non-None) field.

        Args:
            key (str): Field name.

        Returns:
            bool: True if key is present in to_dict().
        """
        return key in self.to_dict()

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Dict-style .get(), delegating to to_dict().

        Args:
            key (str): Field name.
            default (Optional[Any], optional): Value to return if key is not
                set. Defaults to None.

        Returns:
            Any: The field's value, or default.
        """
        return self.to_dict().get(key, default)
