"""Tests for decomposition/base.py: deterministic kernels for seeded CUDA runs."""

from types import SimpleNamespace

import pytest
import torch

from adapt_decomp.decomposition.base import _deterministic_when_seeded_on_cuda


@_deterministic_when_seeded_on_cuda
def _flag(self):
    return torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize(
    ("seed", "device", "expected"),
    [(0, "cuda", True), (None, "cuda", False), (0, "cpu", False)],
)
def test_only_seeded_cuda_runs_use_deterministic_kernels(seed, device, expected):
    # torch.device("cuda") needs no GPU: only the method's own device check runs here
    model = SimpleNamespace(config=SimpleNamespace(random_seed=seed), _device=torch.device(device))
    assert _flag(model) is expected
    assert torch.are_deterministic_algorithms_enabled() is False
