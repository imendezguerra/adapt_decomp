"""Tests for preprocessing/whitening.py: float64 covariance and eigendecomposition, float32 output."""

import pytest
import torch

from adapt_decomp.preprocessing import whiten


def _ill_conditioned(samples: int = 20000, dim: int = 64) -> torch.Tensor:
    """float32 data whose covariance has a condition number of about 1e8."""
    g = torch.Generator().manual_seed(0)
    scales = torch.logspace(0, -4, dim, dtype=torch.float64)
    return (torch.randn(samples, dim, generator=g, dtype=torch.float64) * scales).float()


def _whiteness_error(z: torch.Tensor) -> float:
    """Largest deviation of z's covariance from the identity, computed in float64."""
    z = z.double()
    cov = z.T @ z / (z.shape[0] - 1)
    return float((cov - torch.eye(z.shape[1], dtype=torch.float64)).abs().max())


@pytest.mark.parametrize("method", ["ZCA", "PCA"])
def test_whiten_returns_input_dtype(method):
    """Whitened data and W come back in the input's dtype, whatever compute_dtype."""
    x = _ill_conditioned(samples=2000, dim=16)
    z, W = whiten(x, method=method)
    assert z.dtype == W.dtype == torch.float32
    assert z.shape == x.shape and W.shape == (16, 16)


def test_whiten_float64_is_whiter_than_float32_on_ill_conditioned_data():
    """float64 resolves the small eigenvalues that float32 rounds away."""
    x = _ill_conditioned()
    err32 = _whiteness_error(whiten(x, regularization=None, compute_dtype=torch.float32)[0])
    err64 = _whiteness_error(whiten(x, regularization=None)[0])
    assert err64 < 0.05
    assert err32 > 10 * err64
