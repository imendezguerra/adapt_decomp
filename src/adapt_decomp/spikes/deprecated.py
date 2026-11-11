"""Deprecated 1.0.0 names of the spike comparison functions."""

from __future__ import annotations

import warnings

import numpy as np

from adapt_decomp.spikes.comparison import (
    AgreementMetric,
    SpikeAgreement,
    compare_paired_spikes,
    compare_spikes,
)


def _warn(old_name: str, new_name: str) -> None:
    """Warn that old_name is deprecated in favour of new_name."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version; use {new_name} "
        "instead (tol_train_ms is now max_lag_ms).",
        FutureWarning,
        stacklevel=3,
    )


def rate_of_agreement_paired(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: float = 2048,
    tol_spike_ms: float = 1.0,
    tol_train_ms: float = 40.0,
) -> SpikeAgreement:
    """Deprecated: use compare_paired_spikes.

    Args:
        spike_trains_ref (np.ndarray): See compare_paired_spikes.
        spike_trains_test (np.ndarray): See compare_paired_spikes.
        fs (float): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (float): See compare_paired_spikes.
        tol_train_ms (float): compare_paired_spikes's max_lag_ms.

    Returns:
        SpikeAgreement: See compare_paired_spikes.
    """
    _warn("rate_of_agreement_paired", "compare_paired_spikes")
    return compare_paired_spikes(
        spike_trains_ref,
        spike_trains_test,
        fs=fs,
        tol_spike_ms=tol_spike_ms,
        max_lag_ms=tol_train_ms,
    )


def rate_of_agreement(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: float = 2048,
    tol_spike_ms: float = 1.0,
    tol_train_ms: float = 40.0,
    metric: AgreementMetric = "roa",
) -> SpikeAgreement:
    """Deprecated: use compare_spikes.

    Args:
        spike_trains_ref (np.ndarray): See compare_spikes.
        spike_trains_test (np.ndarray): See compare_spikes.
        fs (float): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (float): See compare_spikes.
        tol_train_ms (float): compare_spikes's max_lag_ms.
        metric (AgreementMetric): See compare_spikes.

    Returns:
        SpikeAgreement: See compare_spikes.
    """
    _warn("rate_of_agreement", "compare_spikes")
    return compare_spikes(
        spike_trains_ref,
        spike_trains_test,
        fs=fs,
        metric=metric,
        tol_spike_ms=tol_spike_ms,
        max_lag_ms=tol_train_ms,
    )
