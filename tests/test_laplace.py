"""Tests for pylibstats.Laplace (LaplaceDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestLaplaceConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.Laplace()
        assert dist.mu == pytest.approx(0.0)
        assert dist.b == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Laplace(mu=2.0, b=0.5)
        assert dist.mu == pytest.approx(2.0)
        assert dist.b == pytest.approx(0.5)

    def test_zero_b_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Laplace(0.0, 0.0)

    def test_negative_b_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Laplace(0.0, -1.0)

    def test_inf_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Laplace(math.inf, 1.0)

    def test_nan_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Laplace(float("nan"), 1.0)


class TestLaplaceScalar:
    """Scalar PDF, CDF, log_pdf, ppf."""

    def test_pdf_at_mean(self, laplace):
        # f(mu; mu, b) = 1/(2b)
        expected = 1.0 / (2.0 * laplace.b)
        assert laplace.pdf(laplace.mu) == pytest.approx(expected, rel=1e-12)

    def test_cdf_at_mean(self, laplace):
        # CDF(mu) = 0.5
        assert laplace.cdf(laplace.mu) == pytest.approx(0.5, rel=1e-12)

    def test_log_pdf_consistency(self, laplace):
        x = 1.5
        assert laplace.log_pdf(x) == pytest.approx(math.log(laplace.pdf(x)), rel=1e-10)

    def test_ppf_median(self, laplace):
        assert laplace.ppf(0.5) == pytest.approx(laplace.mu, abs=1e-10)

    def test_ppf_symmetry(self, laplace):
        # Laplace is symmetric around mu
        mu = laplace.mu
        q = laplace.ppf(0.25)
        assert laplace.ppf(0.75) == pytest.approx(2 * mu - q, rel=1e-10)


class TestLaplaceBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, laplace):
        x = np.linspace(-5, 5, 1000)
        result = laplace.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, laplace):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        batch_result = laplace.pdf(x)
        scalar_results = np.array([laplace.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, laplace):
        x = np.array([-3.0, 0.0, 3.0])
        batch_result = laplace.cdf(x)
        scalar_results = np.array([laplace.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestLaplaceProperties:
    """Moment properties and metadata."""

    def test_mean_property(self, laplace):
        assert laplace.mean == pytest.approx(laplace.mu)

    def test_variance_property(self, laplace):
        # Var = 2*b^2
        assert laplace.variance == pytest.approx(2.0 * laplace.b ** 2, rel=1e-10)

    def test_skewness_zero(self, laplace):
        assert laplace.skewness == pytest.approx(0.0, abs=1e-10)

    def test_kurtosis(self, laplace):
        # Excess kurtosis = 3
        assert laplace.kurtosis == pytest.approx(3.0, rel=1e-10)

    def test_support(self, laplace):
        lower, upper = laplace.support
        assert lower == -math.inf
        assert upper == math.inf

    def test_is_not_discrete(self, laplace):
        assert not laplace.is_discrete


class TestLaplaceFitAndSample:
    """Fitting and sampling."""

    def test_sample_shape(self, laplace):
        samples = laplace.sample(n=500, seed=42)
        assert samples.shape == (500,)

    def test_sample_reproducible(self, laplace):
        s1 = laplace.sample(n=100, seed=17)
        s2 = laplace.sample(n=100, seed=17)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_recovers_params(self):
        dist = pylibstats.Laplace(mu=3.0, b=0.5)
        data = dist.sample(n=50_000, seed=77)
        fitted = pylibstats.Laplace()
        fitted.fit(data)
        assert fitted.mu == pytest.approx(3.0, abs=0.05)
        assert fitted.b == pytest.approx(0.5, abs=0.05)


class TestLaplaceRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.Laplace(2.0, 0.5)
        r = repr(dist)
        assert "2" in r
        assert "0.5" in r
