"""Tests for cbss/ica.py's shared contrast math (log_cosh is also imported
directly by adaptation/ops.py).
"""

import math

import pytest
import torch
from torch.testing import assert_close

from adapt_decomp.cbss.ica import contrast_exponent, contrast_fn, log_cosh


def test_log_cosh_stable():
    """log_cosh should be numerically stable at large |x| and match scipy reference."""
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    out = log_cosh(x)
    # At x=0: log(cosh(0)) = log(1) = 0
    assert_close(out[2], torch.tensor(0.0), atol=1e-6, rtol=0)
    # At large |x|: log(cosh(x)) ≈ |x| - log(2)
    assert_close(out[0], torch.tensor(100.0 - math.log(2.0)), atol=1e-4, rtol=0)
    assert_close(out[4], torch.tensor(100.0 - math.log(2.0)), atol=1e-4, rtol=0)
    # No inf or nan
    assert torch.all(torch.isfinite(out))


@pytest.mark.parametrize("fn", ["square", "cube", "smooth_abs"])
def test_contrast_exponent_is_the_power_of_the_contrast_whose_derivative_is_g(fn):
    """g ∝ u^(e-1) for large u, so G = ∫g ∝ u^e: the scale of SCD's exponents."""
    e = contrast_exponent(fn, contrast_exp=2.5)
    u = torch.tensor([10.0, 100.0], dtype=torch.float64)
    g, _ = contrast_fn(u, fn=fn, contrast_exp=2.5)
    ratio = g / u ** (e - 1)
    assert_close(ratio[0], ratio[1], rtol=1e-3, atol=0)


def test_logcosh_has_no_contrast_exponent():
    assert contrast_exponent("logcosh") is None
