"""Tests for pylibstats.Logistic (LogisticDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestLogisticConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.Logistic()
        assert dist.mu == pytest.approx(0.0)
        assert dist.s == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Logistic(mu=2.0, s=1.5)
        assert dist.mu == pytest.approx(2.0)
        assert dist.s == pytest.approx(1.5)

    def test_zero_s_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Logistic(0.0, 0.0)

    def test_negative_s_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Logistic(0.0, -1.0)

    def test_inf_s_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Logistic(0.0, math.inf)

    def test_nan_s_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Logistic(0.0, float("nan"))

    def test_inf_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Logistic(math.inf, 1.0)

    def test_nan_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Logistic(float("nan"), 1.0)


class TestLogisticScalar:
    """Scalar PDF, CDF, log_pdf, ppf, cross-checked against scipy.stats.logistic."""

    def test_pdf_matches_scipy(self, logistic):
        for x in (0.0, 2.0, 4.5):
            expected = sp.logistic.pdf(x, loc=2.0, scale=1.5)
            assert logistic.pdf(x) == pytest.approx(expected, rel=1e-10)

    def test_cdf_matches_scipy(self, logistic):
        for x in (0.0, 2.0, 4.5):
            expected = sp.logistic.cdf(x, loc=2.0, scale=1.5)
            assert logistic.cdf(x) == pytest.approx(expected, rel=1e-10)

    def test_pdf_at_mean(self, logistic):
        # f(mu; mu, s) = 1/(4s)
        expected = 1.0 / (4.0 * logistic.s)
        assert logistic.pdf(logistic.mu) == pytest.approx(expected, rel=1e-12)

    def test_cdf_at_mean(self, logistic):
        assert logistic.cdf(logistic.mu) == pytest.approx(0.5, rel=1e-12)

    def test_log_pdf_consistency(self, logistic):
        x = 1.5
        assert logistic.log_pdf(x) == pytest.approx(math.log(logistic.pdf(x)), rel=1e-10)

    def test_ppf_median(self, logistic):
        assert logistic.ppf(0.5) == pytest.approx(logistic.mu, abs=1e-10)

    def test_ppf_cdf_round_trip(self, logistic):
        for x in (-1.0, 0.5, 3.0):
            p = logistic.cdf(x)
            assert logistic.ppf(p) == pytest.approx(x, rel=1e-8)


class TestLogisticBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, logistic):
        x = np.linspace(-5, 5, 1000)
        result = logistic.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, logistic):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.5])
        batch_result = logistic.pdf(x)
        scalar_results = np.array([logistic.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, logistic):
        x = np.array([-3.0, 0.0, 3.0])
        batch_result = logistic.cdf(x)
        scalar_results = np.array([logistic.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestLogisticProperties:
    """Moment properties and metadata."""

    def test_mean_property(self, logistic):
        assert logistic.mean == pytest.approx(logistic.mu)

    def test_variance_property(self, logistic):
        # Var = s^2 * pi^2 / 3
        assert logistic.variance == pytest.approx(logistic.s**2 * math.pi**2 / 3.0, rel=1e-10)

    def test_skewness_zero(self, logistic):
        assert logistic.skewness == pytest.approx(0.0, abs=1e-10)

    def test_support(self, logistic):
        lower, upper = logistic.support
        assert lower == -math.inf
        assert upper == math.inf

    def test_is_not_discrete(self, logistic):
        assert not logistic.is_discrete


class TestLogisticFitAndSample:
    """Fitting and sampling."""

    def test_sample_shape(self, logistic):
        samples = logistic.sample(n=500, seed=42)
        assert samples.shape == (500,)

    def test_sample_reproducible(self, logistic):
        s1 = logistic.sample(n=100, seed=17)
        s2 = logistic.sample(n=100, seed=17)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_recovers_params(self):
        dist = pylibstats.Logistic(mu=3.0, s=0.5)
        data = dist.sample(n=50_000, seed=77)
        fitted = pylibstats.Logistic()
        fitted.fit(data)
        assert fitted.mu == pytest.approx(3.0, abs=0.05)
        assert fitted.s == pytest.approx(0.5, abs=0.05)


class TestLogisticRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.Logistic(2.0, 1.5)
        r = repr(dist)
        assert "2" in r
        assert "1.5" in r
