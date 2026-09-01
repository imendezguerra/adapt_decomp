"""Tests for preprocessing/preprocessing.py's select_channels: the shared
bad-channel drop/interpolate/no-op helper used by both CBSS._preprocess_emg
(calibration) and Data._select_channels (online adaptation); and
preprocess_emg_stateful, the per-batch filtering helper used by
AdaptDecomp._preprocess_batch_raw (streaming mode).
"""

import numpy as np
import pytest

from adapt_decomp.preprocessing import (
    preprocess_emg,
    preprocess_emg_stateful,
    select_channels,
    validate_channel_selection,
)


# ---------------------------------------------------------------------------
# No-op: ch_mask=None
# ---------------------------------------------------------------------------

def test_select_channels_noop_when_ch_mask_none():
    """ch_mask=None must pass emg through unchanged, regardless of ch_map/interpolate."""
    emg = np.random.randn(50, 4).astype(np.float32)
    out = select_channels(emg, ch_mask=None, ch_map=np.arange(4).reshape(2, 2), interpolate=True)
    assert out is emg or np.array_equal(out, emg)
    assert out.shape == emg.shape


# ---------------------------------------------------------------------------
# Drop mode
# ---------------------------------------------------------------------------

def test_select_channels_drop_matches_boolean_indexing():
    """ch_mask set, interpolate=False: drop must equal emg[:, ch_mask] exactly."""
    emg = np.random.randn(50, 5).astype(np.float32)
    ch_mask = np.array([True, False, True, True, False])
    out = select_channels(emg, ch_mask=ch_mask, ch_map=None, interpolate=False)
    np.testing.assert_array_equal(out, emg[:, ch_mask])
    assert out.shape == (50, 3)


def test_select_channels_drop_used_even_with_ch_map_when_not_interpolating():
    """interpolate=False must drop (not interpolate) even if ch_map is also set --
    drop no longer conflicts with ch_map (see CHANGELOG/plan)."""
    emg = np.random.randn(20, 4).astype(np.float32)
    ch_mask = np.array([True, True, False, True])
    ch_map = np.array([[0, 1], [2, 3]])
    out = select_channels(emg, ch_mask=ch_mask, ch_map=ch_map, interpolate=False)
    np.testing.assert_array_equal(out, emg[:, ch_mask])


# ---------------------------------------------------------------------------
# Interpolate mode
# ---------------------------------------------------------------------------

def test_select_channels_interpolate_preserves_channel_count():
    """interpolate=True with ch_mask+ch_map set: channel count is unchanged (in-place fill)."""
    emg = np.random.randn(30, 4).astype(np.float32)
    ch_mask = np.array([True, True, True, False])   # channel 3 is bad
    ch_map = np.array([[0, 1], [2, 3]])
    out = select_channels(emg, ch_mask=ch_mask, ch_map=ch_map, interpolate=True)
    assert out.shape == emg.shape
    # Bad channel's column must no longer equal the original (it was replaced).
    assert not np.array_equal(out[:, 3], emg[:, 3])
    # Good channels are untouched.
    np.testing.assert_array_equal(out[:, :3], emg[:, :3])


def test_select_channels_interpolate_requires_ch_map_else_drops():
    """interpolate=True but ch_map=None: falls back to drop (matches
    CBSS._preprocess_emg's precedence -- interpolate only fires when BOTH
    ch_mask and ch_map are set)."""
    emg = np.random.randn(20, 4).astype(np.float32)
    ch_mask = np.array([True, False, True, True])
    out = select_channels(emg, ch_mask=ch_mask, ch_map=None, interpolate=True)
    np.testing.assert_array_equal(out, emg[:, ch_mask])


# ---------------------------------------------------------------------------
# preprocess_emg_stateful: per-batch filtering with zi threaded across calls
# ---------------------------------------------------------------------------

def test_preprocess_emg_stateful_matches_whole_array_call():
    """Filtering in two zi-threaded chunks must match filtering the whole
    array in one preprocess_emg call, within float tolerance."""
    np.random.seed(0)
    fs = 2000.0
    data = np.random.randn(200, 3).astype(np.float32)
    whole = preprocess_emg(data, fs)

    half = 100
    chunk1, zi1 = preprocess_emg_stateful(data[:half], fs)
    chunk2, _ = preprocess_emg_stateful(data[half:], fs, zi=zi1)
    chunked = np.concatenate([chunk1, chunk2], axis=0)

    np.testing.assert_allclose(chunked, whole, atol=1e-4)


def test_preprocess_emg_stateful_single_chunk_matches_preprocess_emg():
    """zi=None on the first call must reproduce preprocess_emg's zero
    initial-state behaviour for a single-chunk input."""
    np.random.seed(1)
    fs = 2000.0
    data = np.random.randn(150, 2).astype(np.float32)
    whole = preprocess_emg(data, fs)
    chunk, _ = preprocess_emg_stateful(data, fs, zi=None)
    np.testing.assert_allclose(chunk, whole, atol=1e-5)


# ---------------------------------------------------------------------------
# validate_channel_selection: shared guard for ch_mask/ch_map/replace_bad_channels
# ---------------------------------------------------------------------------

def test_validate_channel_selection_replace_without_ch_map_raises():
    """replace_bad_channels=True with ch_map=None must raise ValueError."""
    with pytest.raises(ValueError, match="ch_map"):
        validate_channel_selection(
            ch_mask=None, ch_map=None, replace_bad_channels=True, n_raw_channels=4
        )


def test_validate_channel_selection_length_mismatch_raises():
    """ch_mask whose length disagrees with n_raw_channels must raise ValueError."""
    with pytest.raises(ValueError, match="ch_mask"):
        validate_channel_selection(
            ch_mask=np.array([True, False, True]),
            ch_map=None, replace_bad_channels=False, n_raw_channels=4,
        )


def test_validate_channel_selection_noop_when_ch_mask_none():
    """ch_mask=None, replace_bad_channels=False: no error, regardless of n_raw_channels."""
    validate_channel_selection(
        ch_mask=None, ch_map=None, replace_bad_channels=False, n_raw_channels=4
    )
