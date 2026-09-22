"""Tests for cbss/ica.py's shared contrast math (log_cosh is also imported
directly by adaptation/ops.py -- see CLAUDE.md's cbss/ica.py note).
"""

import math
import torch
from torch.testing import assert_close

from adapt_decomp.cbss.ica import log_cosh


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
