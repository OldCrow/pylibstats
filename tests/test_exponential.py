"""Tests for pylibstats.Exponential (ExponentialDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestExponentialConstruction:
    def test_default_params(self):
        dist = pylibstats.Exponential()
        assert dist.lam == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Exponential(lam=2.5)
        assert dist.lam == pytest.approx(2.5)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Exponential(-1.0)

    def test_zero_lambda_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Exponential(0.0)


class TestExponentialScalar:
    def test_pdf_at_zero(self, exponential):
        # Exp(1) PDF at 0 = lambda * exp(0) = 1
        assert exponential.pdf(0.0) == pytest.approx(1.0, rel=1e-12)

    def test_cdf_at_zero(self, exponential):
        assert exponential.cdf(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_pdf_matches_scipy(self, exponential):
        x = 1.5
        expected = sp.expon.pdf(x)  # scale=1/lambda=1
        assert exponential.pdf(x) == pytest.approx(expected, rel=1e-10)

    def test_cdf_matches_scipy(self, exponential):
        x = 2.0
        expected = sp.expon.cdf(x)
        assert exponential.cdf(x) == pytest.approx(expected, rel=1e-10)

    def test_ppf_median(self, exponential):
        expected = math.log(2.0)  # median of Exp(1)
        assert exponential.ppf(0.5) == pytest.approx(expected, rel=1e-10)


class TestExponentialBatch:
    def test_batch_pdf_matches_scalar(self, exponential):
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        batch = exponential.pdf(x)
        scalar = np.array([exponential.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestExponentialProperties:
    def test_mean(self, exponential):
        assert exponential.mean == pytest.approx(1.0, rel=1e-10)

    def test_variance(self, exponential):
        assert exponential.variance == pytest.approx(1.0, rel=1e-10)

    def test_skewness(self, exponential):
        assert exponential.skewness == pytest.approx(2.0, rel=1e-10)

    def test_support(self, exponential):
        lower, upper = exponential.support
        assert lower == pytest.approx(0.0)
        assert upper == math.inf

    def test_is_not_discrete(self, exponential):
        assert not exponential.is_discrete


class TestExponentialFitAndSample:
    def test_sample_shape(self, exponential):
        samples = exponential.sample(n=500, seed=42)
        assert samples.shape == (500,)

    def test_fit_recovers_params(self):
        dist = pylibstats.Exponential(lam=3.0)
        data = dist.sample(n=50_000, seed=99)
        fitted = pylibstats.Exponential()
        fitted.fit(data)
        assert fitted.lam == pytest.approx(3.0, rel=0.05)
