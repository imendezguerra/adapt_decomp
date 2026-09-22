"""Tests for spikes/detection.py: find_peaks_multisource, the canonical
peak-finding primitive shared by cbss and adaptation.
"""

import torch

from adapt_decomp.spikes.detection import find_peaks_multisource


def test_find_peaks_multisource_shape():
    """Output tensors match input shape [N, M]."""
    N, M = 200, 4
    sources = torch.randn(N, M)
    peak_mask, sources_det = find_peaks_multisource(sources, min_dist=10)
    assert peak_mask.shape == (N, M)
    assert sources_det.shape == (N, M)
    assert peak_mask.dtype == torch.bool


def test_find_peaks_strict_suppresses_plateau():
    """A flat plateau should produce at most one peak per source column."""
    N, M = 50, 2
    sources = torch.zeros(N, M)
    # Insert a flat plateau in rows 10-15 for both sources
    sources[10:16, :] = 5.0
    peak_mask, _ = find_peaks_multisource(sources, min_dist=3)
    # Plateau should yield 0 or 1 peak per source (always strict)
    for m in range(M):
        assert peak_mask[:, m].sum() <= 1


def test_source_fifo_edge_spike_not_missed():
    """A spike placed at the last sample of the previous batch is detected in the combined window."""
    N, M = 60, 1
    min_dist = 5

    # prev_sources has a clear spike at its last sample
    prev_sources = torch.zeros(N, M)
    prev_sources[-1, 0] = 10.0  # spike at right edge

    # curr_sources has no spike
    curr_sources = torch.zeros(N, M)

    # Without FIFO: peak detection on curr_sources alone → no spike
    peak_mask_solo, _ = find_peaks_multisource(curr_sources, min_dist=min_dist)
    assert peak_mask_solo.sum() == 0

    # With FIFO: detection on [prev_sources, curr_sources] → spike in prev region
    sources_full = torch.cat([prev_sources, curr_sources], dim=0)
    peak_mask_full, _ = find_peaks_multisource(sources_full, min_dist=min_dist)
    # The spike is in the prev_sources portion (rows 0..N-1) not in curr_sources portion
    # But it would have been missed if we only ran detection on curr_sources
    assert peak_mask_full[:N].sum() > 0


def test_non_debug_fast_path():
    """find_peaks_multisource returns bool mask and float sources_det on the correct device."""
    N, M = 100, 3
    sources = torch.randn(N, M)
    peak_mask, sources_det = find_peaks_multisource(sources, min_dist=5)
    assert peak_mask.dtype == torch.bool
    assert sources_det.dtype == sources.dtype
    assert peak_mask.device == sources.device
    assert sources_det.device == sources.device
