"""Tests for pylibstats.Geometric (GeometricDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestGeometricConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.Geometric()
        assert dist.p == pytest.approx(0.5)

    def test_custom_params(self):
        dist = pylibstats.Geometric(p=0.3)
        assert dist.p == pytest.approx(0.3)

    def test_zero_p_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Geometric(0.0)

    def test_negative_p_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Geometric(-0.1)

    def test_p_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Geometric(1.1)

    def test_p_equals_one_valid(self):
        dist = pylibstats.Geometric(1.0)
        assert dist.p == pytest.approx(1.0)


class TestGeometricScalar:
    """Scalar PMF, CDF, log_pdf, ppf."""

    def test_pmf_at_zero(self, geometric):
        # P(X=0) = p*(1-p)^0 = p
        assert geometric.pdf(0.0) == pytest.approx(geometric.p, rel=1e-10)

    def test_pmf_at_one(self, geometric):
        p = geometric.p
        assert geometric.pdf(1.0) == pytest.approx(p * (1 - p), rel=1e-10)

    def test_cdf_at_zero(self, geometric):
        # CDF(0) = 1 - (1-p)^1 = p
        assert geometric.cdf(0.0) == pytest.approx(geometric.p, rel=1e-10)

    def test_log_pdf_consistency(self, geometric):
        pmf = geometric.pdf(2.0)
        assert pmf > 0.0
        assert geometric.log_pdf(2.0) == pytest.approx(math.log(pmf), rel=1e-10)

    def test_ppf_roundtrip(self, geometric):
        # ppf(CDF(k)) should return k for integer k
        for k in [0.0, 1.0, 2.0, 5.0]:
            cdf_val = geometric.cdf(k)
            assert geometric.ppf(cdf_val) == pytest.approx(k, abs=1.0)


class TestGeometricBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, geometric):
        x = np.arange(10, dtype=np.float64)
        result = geometric.pdf(x)
        assert result.shape == (10,)

    def test_batch_pdf_matches_scalar(self, geometric):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        batch_result = geometric.pdf(x)
        scalar_results = np.array([geometric.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, geometric):
        x = np.array([0.0, 1.0, 2.0, 5.0])
        batch_result = geometric.cdf(x)
        scalar_results = np.array([geometric.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestGeometricProperties:
    """Moment properties and metadata."""

    def test_mean_property(self, geometric):
        # E[X] = (1-p)/p
        p = geometric.p
        assert geometric.mean == pytest.approx((1 - p) / p, rel=1e-10)

    def test_variance_property(self, geometric):
        # Var[X] = (1-p)/p^2
        p = geometric.p
        assert geometric.variance == pytest.approx((1 - p) / (p**2), rel=1e-10)

    def test_is_discrete(self, geometric):
        assert geometric.is_discrete

    def test_support_lower_bound(self, geometric):
        lower, _ = geometric.support
        assert lower == pytest.approx(0.0)


class TestGeometricFitAndSample:
    """Fitting and sampling."""

    def test_sample_shape(self, geometric):
        samples = geometric.sample(n=500, seed=42)
        assert samples.shape == (500,)

    def test_sample_reproducible(self, geometric):
        s1 = geometric.sample(n=100, seed=7)
        s2 = geometric.sample(n=100, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_recovers_params(self):
        dist = pylibstats.Geometric(0.4)
        data = dist.sample(n=50_000, seed=99)
        fitted = pylibstats.Geometric()
        fitted.fit(data)
        assert fitted.p == pytest.approx(0.4, abs=0.02)


class TestGeometricRepr:
    """String representation."""

    def test_repr_not_empty(self, geometric):
        assert len(repr(geometric)) > 0
