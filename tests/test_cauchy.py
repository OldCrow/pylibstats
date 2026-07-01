"""Tests for pylibstats.Cauchy (CauchyDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestCauchyConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.Cauchy()
        assert dist.x0 == pytest.approx(0.0)
        assert dist.gamma == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Cauchy(x0=3.0, gamma=2.0)
        assert dist.x0 == pytest.approx(3.0)
        assert dist.gamma == pytest.approx(2.0)

    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Cauchy(0.0, 0.0)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Cauchy(0.0, -1.0)

    def test_inf_x0_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Cauchy(math.inf, 1.0)

    def test_nan_x0_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Cauchy(float("nan"), 1.0)


class TestCauchyScalar:
    """Scalar PDF, CDF, log_pdf, ppf."""

    def test_pdf_at_location(self, cauchy):
        # f(x0; x0, gamma) = 1/(pi*gamma)
        expected = 1.0 / (math.pi * cauchy.gamma)
        assert cauchy.pdf(cauchy.x0) == pytest.approx(expected, rel=1e-12)

    def test_cdf_at_location(self, cauchy):
        # CDF(x0) = 0.5
        assert cauchy.cdf(cauchy.x0) == pytest.approx(0.5, rel=1e-12)

    def test_log_pdf_consistency(self, cauchy):
        x = 1.5
        assert cauchy.log_pdf(x) == pytest.approx(math.log(cauchy.pdf(x)), rel=1e-10)

    def test_ppf_median(self, cauchy):
        assert cauchy.ppf(0.5) == pytest.approx(cauchy.x0, abs=1e-10)

    def test_ppf_symmetry(self, cauchy):
        # Cauchy is symmetric around x0
        x0 = cauchy.x0
        assert cauchy.ppf(0.25) == pytest.approx(2 * x0 - cauchy.ppf(0.75), rel=1e-10)


class TestCauchyBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, cauchy):
        x = np.linspace(-10, 10, 1000)
        result = cauchy.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, cauchy):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        batch_result = cauchy.pdf(x)
        scalar_results = np.array([cauchy.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, cauchy):
        x = np.array([-5.0, 0.0, 5.0])
        batch_result = cauchy.cdf(x)
        scalar_results = np.array([cauchy.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestCauchyProperties:
    """Moment properties — all undefined (NaN) for Cauchy."""

    def test_mean_is_nan(self, cauchy):
        assert math.isnan(cauchy.mean)

    def test_variance_is_nan(self, cauchy):
        assert math.isnan(cauchy.variance)

    def test_skewness_is_nan(self, cauchy):
        assert math.isnan(cauchy.skewness)

    def test_kurtosis_is_nan(self, cauchy):
        assert math.isnan(cauchy.kurtosis)

    def test_support(self, cauchy):
        lower, upper = cauchy.support
        assert lower == -math.inf
        assert upper == math.inf

    def test_is_not_discrete(self, cauchy):
        assert not cauchy.is_discrete


class TestCauchyFitAndSample:
    """Fitting and sampling."""

    def test_sample_shape(self, cauchy):
        samples = cauchy.sample(n=500, seed=42)
        assert samples.shape == (500,)

    def test_sample_reproducible(self, cauchy):
        s1 = cauchy.sample(n=100, seed=55)
        s2 = cauchy.sample(n=100, seed=55)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_recovers_params(self):
        dist = pylibstats.Cauchy(x0=2.0, gamma=0.5)
        data = dist.sample(n=50_000, seed=11)
        fitted = pylibstats.Cauchy()
        fitted.fit(data)
        assert fitted.x0 == pytest.approx(2.0, abs=0.1)
        assert fitted.gamma == pytest.approx(0.5, abs=0.1)


class TestCauchyRepr:
    """String representation."""

    def test_repr_not_empty(self, cauchy):
        assert len(repr(cauchy)) > 0
