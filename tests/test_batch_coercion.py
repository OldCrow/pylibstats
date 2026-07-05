"""Tests for P2 fix: pdf / log_pdf / cdf batch input coercion.

The C++ batch overload requires a C-contiguous float64 NDArray.  Before this
fix, int arrays, lists, and strided views raised TypeError.  The Python layer
now coerces all array-like inputs to C-contiguous float64 before dispatch.

Uses Gaussian as the representative distribution; the coercion path is shared
across all 19 classes via the _install_batch_coercion loop in __init__.py.
"""

import numpy as np
import pytest

import pylibstats


@pytest.fixture
def g():
    return pylibstats.Gaussian(mu=0.0, sigma=1.0)


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------

class TestScalarCoercion:
    def test_int_scalar_pdf(self, g):
        assert g.pdf(0) == pytest.approx(g.pdf(0.0))

    def test_int_scalar_log_pdf(self, g):
        assert g.log_pdf(0) == pytest.approx(g.log_pdf(0.0))

    def test_int_scalar_cdf(self, g):
        assert g.cdf(0) == pytest.approx(g.cdf(0.0))

    def test_numpy_int_scalar_pdf(self, g):
        assert g.pdf(np.int64(1)) == pytest.approx(g.pdf(1.0))

    def test_zero_dim_array_pdf(self, g):
        assert g.pdf(np.array(1.0)) == pytest.approx(g.pdf(1.0))


# ---------------------------------------------------------------------------
# Batch array coercion
# ---------------------------------------------------------------------------

class TestBatchIntArray:
    def test_int32_array_pdf(self, g):
        x_int = np.array([-1, 0, 1], dtype=np.int32)
        x_flt = x_int.astype(np.float64)
        np.testing.assert_allclose(g.pdf(x_int), g.pdf(x_flt))

    def test_int64_array_log_pdf(self, g):
        x_int = np.array([-2, 0, 2], dtype=np.int64)
        x_flt = x_int.astype(np.float64)
        np.testing.assert_allclose(g.log_pdf(x_int), g.log_pdf(x_flt))

    def test_int32_array_cdf(self, g):
        x_int = np.array([-1, 0, 1], dtype=np.int32)
        x_flt = x_int.astype(np.float64)
        np.testing.assert_allclose(g.cdf(x_int), g.cdf(x_flt))


class TestBatchListInput:
    def test_list_pdf(self, g):
        x_list = [-1.0, 0.0, 1.0]
        x_arr = np.array(x_list, dtype=np.float64)
        np.testing.assert_allclose(g.pdf(x_list), g.pdf(x_arr))

    def test_list_log_pdf(self, g):
        x_list = [-1.0, 0.0, 1.0]
        x_arr = np.array(x_list, dtype=np.float64)
        np.testing.assert_allclose(g.log_pdf(x_list), g.log_pdf(x_arr))

    def test_list_cdf(self, g):
        x_list = [-1.0, 0.0, 1.0]
        x_arr = np.array(x_list, dtype=np.float64)
        np.testing.assert_allclose(g.cdf(x_list), g.cdf(x_arr))

    def test_int_list_pdf(self, g):
        x_list = [-1, 0, 1]
        x_arr = np.array(x_list, dtype=np.float64)
        np.testing.assert_allclose(g.pdf(x_list), g.pdf(x_arr))


class TestBatchStridedArray:
    def test_strided_pdf(self, g):
        # Every other element: not C-contiguous
        x_full = np.array([-2.0, 99.0, -1.0, 99.0, 0.0, 99.0, 1.0, 99.0, 2.0, 99.0])
        x_strided = x_full[::2]  # stride-2 view, not contiguous
        assert not x_strided.flags['C_CONTIGUOUS']
        x_contig = np.ascontiguousarray(x_strided, dtype=np.float64)
        np.testing.assert_allclose(g.pdf(x_strided), g.pdf(x_contig))

    def test_strided_cdf(self, g):
        x_full = np.linspace(-3.0, 3.0, 20)
        x_strided = x_full[::3]
        assert not x_strided.flags['C_CONTIGUOUS'] or x_strided.strides[0] != x_strided.itemsize
        x_contig = np.ascontiguousarray(x_strided)
        np.testing.assert_allclose(g.cdf(x_strided), g.cdf(x_contig))


# ---------------------------------------------------------------------------
# Multi-distribution spot check (coercion installed on all 19 classes)
# ---------------------------------------------------------------------------

class TestAllDistributionsIntArray:
    """Spot-check that the coercion loop ran for every distribution."""

    @pytest.mark.parametrize("dist,x", [
        (pylibstats.Exponential(lam=1.0), np.array([0, 1, 2], dtype=np.int32)),
        (pylibstats.Gamma(alpha=2.0, beta=1.0), np.array([1, 2, 3], dtype=np.int32)),
        (pylibstats.Poisson(lam=3.0), np.array([0, 1, 2, 3], dtype=np.int32)),
        (pylibstats.Beta(alpha=2.0, beta=2.0), np.array([0, 1], dtype=np.int32)),
        (pylibstats.Weibull(shape=2.0, scale=1.0), np.array([1, 2, 3], dtype=np.int32)),
        (pylibstats.Laplace(mu=0.0, b=1.0), np.array([-1, 0, 1], dtype=np.int32)),
        (pylibstats.Cauchy(x0=0.0, gamma=1.0), np.array([-1, 0, 1], dtype=np.int32)),
    ])
    def test_int_array_accepted(self, dist, x):
        x_flt = x.astype(np.float64)
        np.testing.assert_allclose(dist.pdf(x), dist.pdf(x_flt))
        np.testing.assert_allclose(dist.cdf(x), dist.cdf(x_flt))
