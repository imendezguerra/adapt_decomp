"""Tests for spikes/metrics.py: get_sil, the full-recording silhouette score
built from an already-detected spike train (no re-detection/re-clustering
for the spike side; a single cheap peak-finding pass for the base side).
"""

import torch

from adapt_decomp.spikes.metrics import get_sil


def test_get_sil_high_for_separated_clusters():
    """Well-separated spike/base peak amplitudes should give a SIL close to 1."""
    N = 1000
    min_dist = 5
    source = torch.zeros(N, 1)
    spike_trains = torch.zeros(N, 1, dtype=torch.int32)

    spike_pos = torch.arange(10, N, 50)
    source[spike_pos, 0] = 10.0
    spike_trains[spike_pos, 0] = 1

    base_pos = spike_pos[:-1] + 25  # clear of spikes and of the NMS window
    source[base_pos, 0] = 1.0

    sil = get_sil(source, spike_trains, min_dist=min_dist, peak_power=2.0)

    assert sil.shape == (1,)
    assert sil.item() > 0.9


def test_get_sil_low_for_overlapping_clusters():
    """Near-identical spike/base amplitudes should give a SIL close to 0."""
    torch.manual_seed(0)
    N = 1000
    min_dist = 5
    source = torch.zeros(N, 1)
    spike_trains = torch.zeros(N, 1, dtype=torch.int32)

    spike_pos = torch.arange(10, N, 50)
    base_pos = spike_pos[:-1] + 25
    source[spike_pos, 0] = 5.0 + 0.1 * torch.randn(len(spike_pos))
    source[base_pos, 0] = 5.0 + 0.1 * torch.randn(len(base_pos))
    spike_trains[spike_pos, 0] = 1

    sil = get_sil(source, spike_trains, min_dist=min_dist, peak_power=2.0)

    assert sil.item() < 0.3


def test_get_sil_excludes_spike_peaks_from_base_population():
    """A peak coinciding with a labelled spike must not also count toward the base cluster."""
    N = 300
    min_dist = 5
    source = torch.zeros(N, 1)
    spike_trains = torch.zeros(N, 1, dtype=torch.int32)

    spike_pos = torch.tensor([20, 60, 100, 140])
    spike_amps = torch.tensor([8.0, 9.0, 10.0, 11.0])
    source[spike_pos, 0] = spike_amps
    spike_trains[spike_pos, 0] = 1

    base_pos = torch.tensor([180, 220])  # genuine base peaks, well clear of any spike
    base_amps = torch.tensor([1.0, 1.5])
    source[base_pos, 0] = base_amps

    sil = get_sil(source, spike_trains, min_dist=min_dist, peak_power=2.0)

    # Expected score using ONLY the genuine base peaks -- if the 4 spike-coincident
    # peaks leaked into the base population instead of being excluded, this would differ.
    spike_vals, base_vals = spike_amps ** 2, base_amps ** 2
    spike_centroid, base_centroid = spike_vals.median(), base_vals.median()
    within = ((spike_vals - spike_centroid) ** 2).sum()
    between = ((spike_vals - base_centroid) ** 2).sum()
    expected = (between - within) / max(within, between)

    assert torch.isclose(sil[0], expected, atol=1e-4)


def test_get_sil_falls_back_to_non_peak_samples_when_no_base_peaks():
    """When every found peak coincides with a spike, fall back to non-peak samples."""
    N = 500
    min_dist = 5
    source = torch.zeros(N, 1)
    spike_trains = torch.zeros(N, 1, dtype=torch.int32)

    spike_pos = torch.arange(10, N, 40)
    source[spike_pos, 0] = 10.0
    spike_trains[spike_pos, 0] = 1
    # No other local maxima anywhere -- every peak found coincides with a spike.

    sil = get_sil(source, spike_trains, min_dist=min_dist, peak_power=2.0)

    assert torch.isfinite(sil).all()
    assert sil.item() > 0.5  # the zero-background fallback population is still well separated


def test_get_sil_zero_when_no_spikes():
    """A unit with no detected spikes should return 0.0, not raise."""
    N = 200
    source = torch.randn(N, 1)
    spike_trains = torch.zeros(N, 1, dtype=torch.int32)

    sil = get_sil(source, spike_trains, min_dist=5, peak_power=2.0)

    assert sil.item() == 0.0


def test_get_sil_shape_multi_unit():
    """Output shape matches the number of units (source columns)."""
    N, M = 200, 3
    torch.manual_seed(1)
    source = torch.randn(N, M)
    spike_trains = (torch.rand(N, M) > 0.95).to(torch.int32)

    sil = get_sil(source, spike_trains, min_dist=5, peak_power=2.0)

    assert sil.shape == (M,)
    assert torch.isfinite(sil).all()
