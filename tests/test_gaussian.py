"""Tests for pylibstats.Gaussian (GaussianDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestGaussianConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.Gaussian()
        assert dist.mu == pytest.approx(0.0)
        assert dist.sigma == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Gaussian(mu=5.0, sigma=2.0)
        assert dist.mu == pytest.approx(5.0)
        assert dist.sigma == pytest.approx(2.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="[Ss]tandard deviation"):
            pylibstats.Gaussian(0.0, -1.0)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gaussian(0.0, 0.0)

    def test_normal_alias(self):
        dist = pylibstats.Normal(1.0, 2.0)
        assert isinstance(dist, pylibstats.Gaussian)


class TestGaussianScalar:
    """Scalar PDF, CDF, log_pdf, ppf."""

    def test_pdf_at_mean(self, gaussian):
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert gaussian.pdf(0.0) == pytest.approx(expected, rel=1e-12)

    def test_cdf_at_zero(self, gaussian):
        assert gaussian.cdf(0.0) == pytest.approx(0.5, rel=1e-12)

    def test_log_pdf_consistency(self, gaussian):
        x = 1.5
        assert gaussian.log_pdf(x) == pytest.approx(math.log(gaussian.pdf(x)), rel=1e-10)

    def test_ppf_median(self, gaussian):
        assert gaussian.ppf(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_ppf_symmetry(self, gaussian):
        assert gaussian.ppf(0.025) == pytest.approx(-gaussian.ppf(0.975), rel=1e-10)


class TestGaussianBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, gaussian):
        x = np.linspace(-3, 3, 1000)
        result = gaussian.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, gaussian):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        batch_result = gaussian.pdf(x)
        scalar_results = np.array([gaussian.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, gaussian):
        x = np.array([-2.0, 0.0, 2.0])
        batch_result = gaussian.cdf(x)
        scalar_results = np.array([gaussian.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestGaussianProperties:
    """Moment properties and metadata."""

    def test_mean_property(self, gaussian):
        assert gaussian.mean == pytest.approx(0.0)

    def test_variance_property(self, gaussian):
        assert gaussian.variance == pytest.approx(1.0)

    def test_std_property(self, gaussian):
        assert gaussian.std == pytest.approx(1.0)

    def test_skewness(self, gaussian):
        assert gaussian.skewness == pytest.approx(0.0)

    def test_kurtosis(self, gaussian):
        assert gaussian.kurtosis == pytest.approx(0.0)

    def test_support(self, gaussian):
        lower, upper = gaussian.support
        assert lower == -math.inf
        assert upper == math.inf

    def test_is_not_discrete(self, gaussian):
        assert not gaussian.is_discrete


class TestGaussianFitAndSample:
    """Fitting and sampling."""

    def test_sample_shape(self, gaussian):
        samples = gaussian.sample(n=500, seed=42)
        assert samples.shape == (500,)

    def test_sample_reproducible(self, gaussian):
        s1 = gaussian.sample(n=100, seed=123)
        s2 = gaussian.sample(n=100, seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_recovers_params(self):
        dist = pylibstats.Gaussian(3.0, 0.5)
        data = dist.sample(n=50_000, seed=99)
        fitted = pylibstats.Gaussian()
        fitted.fit(data)
        assert fitted.mu == pytest.approx(3.0, abs=0.05)
        assert fitted.sigma == pytest.approx(0.5, abs=0.05)


class TestGaussianRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.Gaussian(2.0, 0.5)
        r = repr(dist)
        assert "2" in r
        assert "0.5" in r
