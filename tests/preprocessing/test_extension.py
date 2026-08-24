"""Tests for preprocessing/extension.py: extend_data's block/toeplitz modes."""

import torch
import pytest
from torch.testing import assert_close

from adapt_decomp.preprocessing import extend_data


def test_extend_data_toeplitz_matches_manual_construction():
    """toeplitz mode must place each channel's own delays in contiguous columns,
    forming a per-channel Toeplitz (constant-diagonal) block -- i.e. column
    c*ext_fact+i holds channel c delayed by i samples."""
    samples, chs, ext_fact = 20, 3, 4
    data = torch.randn(samples, chs)

    out = extend_data(data, ext_fact, ext_mode="toeplitz")
    assert out.shape == (samples, chs * ext_fact)

    expected = torch.zeros_like(out)
    for c in range(chs):
        for i in range(ext_fact):
            col = c * ext_fact + i
            expected[i:, col] = data[: samples - i, c]
    assert_close(out, expected)


def test_extend_data_toeplitz_is_permutation_of_block():
    """block and toeplitz modes must carry the same information, just reordered
    (delay-major vs channel-major) -- neither mode invents or drops content."""
    samples, chs, ext_fact = 15, 2, 3
    data = torch.randn(samples, chs)

    block = extend_data(data, ext_fact, ext_mode="block")
    toeplitz = extend_data(data, ext_fact, ext_mode="toeplitz")

    reordered = (
        block.view(samples, ext_fact, chs).permute(0, 2, 1).reshape(samples, chs * ext_fact)
    )
    assert_close(toeplitz, reordered)


def test_extend_data_unknown_mode_raises():
    with pytest.raises(ValueError):
        extend_data(torch.randn(10, 2), 2, ext_mode="bogus")


def test_extend_data_preserves_dtype():
    """extend_data must not silently upcast/downcast -- output dtype must match input."""
    data = torch.randn(10, 2, dtype=torch.float64)
    for mode in ("block", "toeplitz"):
        out = extend_data(data, 3, ext_mode=mode)
        assert out.dtype == torch.float64
