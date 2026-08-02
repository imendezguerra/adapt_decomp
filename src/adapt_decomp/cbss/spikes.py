"""Spike detection and clustering for CBSS."""

from __future__ import annotations

import torch

from adapt_decomp.ops import find_peaks_multisource, kmeans2_1d


def _spike_detection(
    source: torch.Tensor,
    fs: float,
    min_dist_ms: float,
    a: float = 2.0,
    compute_sil: bool = True,
) -> tuple:
    """Detect spike times on-device: peak-find + optimal 1-D 2-means.

    Args:
        source:      [T] source signal.
        fs:          Sampling frequency.
        min_dist_ms: Minimum inter-spike distance in ms.
        a:           Detection exponent (source is raised to power a).
        compute_sil: Whether to compute and return the silhouette score.

    Returns:
        ``(spike_idx, spike_centr, base_centr)`` or
        ``(spike_idx, spike_centr, base_centr, sil)`` when compute_sil=True.
    """
    source_det = source ** a
    min_dist = max(1, int(min_dist_ms / 1000.0 * fs))

    # Use the canonical multi-source peak finder on a single [T, 1] view
    peak_mask, _ = find_peaks_multisource(source_det.unsqueeze(1), min_dist, peak_power=1.0)
    peaks = peak_mask[:, 0].nonzero(as_tuple=True)[0]

    if peaks.shape[0] < 2:
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        return (empty, 0.0, 0.0, 0.0) if compute_sil else (empty, 0.0, 0.0)

    labels, centers = kmeans2_1d(source_det[peaks])
    spike_cluster = int(centers.argmax().item())
    spike_idx = peaks[labels == spike_cluster]

    if not compute_sil:
        return (
            spike_idx,
            float(centers[spike_cluster].item()),
            float(centers[1 - spike_cluster].item()),
        )

    sil = _sil_peaks(source, peaks, labels, centers, a)
    return (
        spike_idx,
        float(centers[spike_cluster].item()),
        float(centers[1 - spike_cluster].item()),
        float(sil),
    )


def _apply_spike_detection(
    source: torch.Tensor,
    spike_centroid: torch.Tensor,
    base_centroid: torch.Tensor,
    fs: float,
    min_dist_ms: float,
    a: float = 2.0,
    compute_sil: bool = True,
) -> torch.Tensor | tuple:
    """Apply spike detection using previously identified centroids.

    Returns:
        ``spike_idx`` (no sil) or ``(spike_idx, sil)`` when compute_sil=True.
    """
    source_det = source ** a
    min_dist = max(1, int(min_dist_ms / 1000.0 * fs))

    peak_mask, _ = find_peaks_multisource(source_det.unsqueeze(1), min_dist, peak_power=1.0)
    peaks = peak_mask[:, 0].nonzero(as_tuple=True)[0]

    if peaks.shape[0] < 2:
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        return (empty, 0.0) if compute_sil else empty

    spike_thr = base_centroid + (spike_centroid - base_centroid) / 2
    labels = (source_det[peaks] > spike_thr).long()
    spike_idx = peaks[labels == 1]

    if not compute_sil:
        return spike_idx

    centers = torch.stack([base_centroid, spike_centroid])
    sil = _sil_peaks(source, peaks, labels, centers, a)
    return spike_idx, float(sil)


def _sil_peaks(
    source: torch.Tensor,
    peaks: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    a: float = 2.0,
) -> float:
    """Silhouette from peak clusters: (between − within) / max(between, within)."""
    source_sil = source ** a
    peak_vals = source_sil[peaks]
    spike_cluster = int(centers.argmax().item())
    spike_mask = labels == spike_cluster
    if not spike_mask.any():
        return 0.0
    D = (peak_vals.unsqueeze(1) - centers.unsqueeze(0)) ** 2
    within = float(D[spike_mask, spike_cluster].sum().item())
    between = float(D[spike_mask, 1 - spike_cluster].sum().item())
    denom = max(within, between)
    return float((between - within) / denom) if denom > 0 else 0.0
