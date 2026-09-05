"""Tests for pylibstats.HalfNormal (HalfNormalDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestHalfNormalConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.HalfNormal()
        assert dist.sigma == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.HalfNormal(sigma=2.0)
        assert dist.sigma == pytest.approx(2.0)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.HalfNormal(0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.HalfNormal(-1.0)

    def test_inf_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.HalfNormal(math.inf)

    def test_nan_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.HalfNormal(float("nan"))


class TestHalfNormalScalar:
    """Scalar PDF, CDF, log_pdf, ppf, cross-checked against scipy.stats.halfnorm."""

    def test_pdf_matches_scipy(self, half_normal):
        for x in (0.5, 1.0, 2.0):
            expected = sp.halfnorm.pdf(x, scale=2.0)
            assert half_normal.pdf(x) == pytest.approx(expected, rel=1e-10)

    def test_cdf_matches_scipy(self, half_normal):
        for x in (0.5, 1.0, 2.0):
            expected = sp.halfnorm.cdf(x, scale=2.0)
            assert half_normal.cdf(x) == pytest.approx(expected, rel=1e-8)

    def test_log_pdf_consistency(self, half_normal):
        x = 1.5
        assert half_normal.log_pdf(x) == pytest.approx(math.log(half_normal.pdf(x)), rel=1e-10)

    def test_ppf_matches_scipy(self, half_normal):
        p = 0.9
        expected = sp.halfnorm.ppf(p, scale=2.0)
        assert half_normal.ppf(p) == pytest.approx(expected, rel=1e-6)

    def test_ppf_cdf_round_trip(self, half_normal):
        for x in (0.25, 1.0, 3.0):
            p = half_normal.cdf(x)
            assert half_normal.ppf(p) == pytest.approx(x, rel=1e-6)

    def test_pdf_outside_support(self, half_normal):
        assert half_normal.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)


class TestHalfNormalBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, half_normal):
        x = np.linspace(0.0, 10, 1000)
        result = half_normal.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, half_normal):
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 8.0])
        batch_result = half_normal.pdf(x)
        scalar_results = np.array([half_normal.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, half_normal):
        x = np.array([0.5, 1.0, 3.0])
        batch_result = half_normal.cdf(x)
        scalar_results = np.array([half_normal.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestHalfNormalProperties:
    """Moment properties and metadata, cross-checked against scipy."""

    def test_mean_matches_scipy(self, half_normal):
        expected = sp.halfnorm.mean(scale=2.0)
        assert half_normal.mean == pytest.approx(expected, rel=1e-10)

    def test_variance_matches_scipy(self, half_normal):
        expected = sp.halfnorm.var(scale=2.0)
        assert half_normal.variance == pytest.approx(expected, rel=1e-10)

    def test_support(self, half_normal):
        lower, upper = half_normal.support
        assert lower == pytest.approx(0.0)
        assert upper == math.inf

    def test_is_not_discrete(self, half_normal):
        assert not half_normal.is_discrete


class TestHalfNormalFitAndSample:
    """Fitting and sampling."""

    def test_sample_nonnegative(self, half_normal):
        samples = half_normal.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)

    def test_sample_shape(self, half_normal):
        assert half_normal.sample(n=200, seed=1).shape == (200,)

    def test_sample_reproducible(self, half_normal):
        s1 = half_normal.sample(n=100, seed=17)
        s2 = half_normal.sample(n=100, seed=17)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_recovers_params(self):
        dist = pylibstats.HalfNormal(sigma=2.5)
        data = dist.sample(n=50_000, seed=77)
        fitted = pylibstats.HalfNormal()
        fitted.fit(data)
        assert fitted.sigma == pytest.approx(2.5, abs=0.05)


class TestHalfNormalRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.HalfNormal(2.0)
        r = repr(dist)
        assert "2" in r
