"""Spike detection and clustering."""

import torch
from torch.nn import functional as F

@torch.no_grad()
def detect_spikes(
    source: torch.Tensor,
    min_dist: int,
    spike_centroid: torch.Tensor = None,
    base_centroid: torch.Tensor = None,
    peak_power: float = 2.0,
    use_abs: bool = False,
    compute_sil: bool = True,
) -> tuple:
    """Detect spikes based on centroids if provided, otherwise calculate them.

    Args:
        source (torch.Tensor): Source signal of shape [N] where N is the number of time points.
        min_dist (int): Minimum distance between peaks.
        spike_centroid (torch.Tensor, optional): Spike centroid. Defaults to None.
        base_centroid (torch.Tensor, optional): Base centroid. Defaults to None.
        peak_power (float, optional): Power to raise the source signal to before peak detection. Defaults to 2.0.
        use_abs (bool, optional): Whether to use the absolute value of the source signal. Defaults to False.
        compute_sil (bool, optional): Whether to compute the silhouette score. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - spike_idx (torch.Tensor): Indices of detected spikes.
            - spike_centroid (float): Centroid of the spike cluster.
            - base_centroid (float): Centroid of the base cluster.
            - sil (float, optional): Silhouette score if compute_sil is True.
    """
    # Apply peak detection to the source signal using torch
    peak_mask, peak_values = find_peaks_multisource(source.unsqueeze(1), min_dist, peak_power, use_abs)
    peak_values = peak_values[:, 0]                      
    peaks = peak_mask[:, 0].nonzero(as_tuple=True)[0]

    # If centroids are provided, use them to classify peaks; otherwise, compute centroids using k-means clustering
    if spike_centroid is not None and base_centroid is not None:
        spike_thr = base_centroid + (spike_centroid - base_centroid) / 2
        labels = (peak_values[peaks] > spike_thr).long()
        centers = torch.tensor([base_centroid, spike_centroid], dtype=peak_values.dtype, device=source.device)
    else:
        if peaks.shape[0] < 2:
            empty = torch.empty(0, dtype=torch.long, device=source.device)
            # float(...) rather than `x or 0.0`: the latter returns the raw tensor
            # unchanged whenever it's truthy (nonzero), silently breaking the float
            # return type this function documents for every other code path.
            sc = float(spike_centroid) if spike_centroid is not None else 0.0
            bc = float(base_centroid) if base_centroid is not None else 0.0
            vals = (empty, sc, bc)
            return vals + (0.0,) if compute_sil else vals
        labels, centers = _kmeans2_1d(peak_values[peaks])

    # Determine the spike cluster and extract the indices of detected spikes
    spike_cluster = int(centers.argmax().item())
    spike_idx = peaks[labels == spike_cluster]

    # If compute_sil is False, return the spike indices and centroids; otherwise, compute the silhouette score
    if not compute_sil:
        return (spike_idx, float(centers[spike_cluster].item()), float(centers[1 - spike_cluster].item()))
    sil = _sil_peaks(source, peaks, labels, centers, peak_power, use_abs)
    return (spike_idx, float(centers[spike_cluster].item()), float(centers[1 - spike_cluster].item()), float(sil))


@torch.no_grad()
def find_peaks_multisource(
    sources: torch.Tensor,
    min_dist: int,
    peak_power: float = 2.0,
    use_abs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized multi-source peak detector using 1-D max-pool NMS.

    Canonical peak-finding primitive for the package -- used by detect_spikes
    below (single source), spikes/metrics.py (pulse-to-noise ratio), and
    adaptation/ops.py (per-batch online multi-source detection). Suppresses
    flat plateaus (requires a strict local max on both neighbours) so tied
    runs cannot emit multiple peaks.

    Args:
        sources (torch.Tensor): sources of shape [N, M] where N is the number of time points and M is the number of sources.
        min_dist (int): Minimum distance between peaks.
        peak_power (float, optional): Power to raise the source signal to before peak detection. Defaults to 2.0.
        use_abs (bool, optional): Whether to use the absolute value of the source signal. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the peak mask and the detected peak values.
    """

    N, M = sources.shape
    sources_det = sources.abs().pow(peak_power) if use_abs else sources.pow(peak_power)

    win = 2 * min_dist + 1
    # Apply max-pool across time for all sources simultaneously: input [1, M, N]
    pooled = (
        F.max_pool1d(
            sources_det.T.unsqueeze(0).float(),
            kernel_size=win,
            stride=1,
            padding=min_dist,
        )
        .squeeze(0)
        .T.to(sources_det.dtype)
    )

    strict_mask = torch.zeros_like(sources_det, dtype=torch.bool)
    if N >= 3:
        strict_mask[1:-1] = (sources_det[1:-1] > sources_det[:-2]) & (sources_det[1:-1] > sources_det[2:])
    peak_mask = strict_mask & (sources_det == pooled) & (sources_det > 0)

    return peak_mask, sources_det


@torch.no_grad()
def _kmeans2_1d(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 1D k-means clustering with k=2 to a 1D tensor of values.

    Args:
        vals (torch.Tensor): Values to cluster (1D).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Labels and cluster centroids (2).
    """
    sorted_vals, sort_idx = vals.sort()
    N = sorted_vals.shape[0]
    cs  = sorted_vals.cumsum(0)
    cs2 = (sorted_vals ** 2).cumsum(0)
    ks  = torch.arange(1, N, device=vals.device, dtype=vals.dtype)
    n0, n1  = ks, N - ks
    s0, s2_0 = cs[:-1], cs2[:-1]
    s1, s2_1 = cs[-1] - s0, cs2[-1] - s2_0
    cost = (s2_0 - s0 ** 2 / n0) + (s2_1 - s1 ** 2 / n1)
    k = int(cost.argmin().item()) + 1
    labels_sorted = torch.zeros(N, dtype=torch.long, device=vals.device)
    labels_sorted[k:] = 1
    labels = torch.empty_like(labels_sorted)
    labels[sort_idx] = labels_sorted
    c0 = sorted_vals[:k].median()
    c1 = sorted_vals[k:].median()
    return labels, torch.stack([c0, c1])

@torch.no_grad()
def _sil_peaks(
    source: torch.Tensor,
    peaks: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    peak_power: float = 2.0,
    use_abs: bool = False,
) -> float:
    """Compute the silhouette score for the detected peaks based on their labels and cluster centers.

    Args:
        source (torch.Tensor): Source signal of shape [N] where N is the number of time points.
        peaks (torch.Tensor): Indices of detected peaks.
        labels (torch.Tensor): Labels for each peak.
        centers (torch.Tensor): Cluster centers.
        peak_power (float, optional): Power for peak amplitude normalization. Defaults to 2.0.
        use_abs (bool, optional): Whether to use absolute values for peak amplitude normalization. Defaults to False.

    Returns:
        float: Silhouette score.
    """
    
    source_sil = source.abs().pow(peak_power) if use_abs else source.pow(peak_power)
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


