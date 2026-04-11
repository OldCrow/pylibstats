"""Tests for pylibstats.ChiSquared (ChiSquaredDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestChiSquaredConstruction:
    def test_default_params(self):
        dist = pylibstats.ChiSquared()
        assert dist.k == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.ChiSquared(k=10.0)
        assert dist.k == pytest.approx(10.0)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            pylibstats.ChiSquared(k=0.0)


class TestChiSquaredScalar:
    def test_pdf_matches_scipy(self, chi_squared):
        x = 3.0
        expected = sp.chi2.pdf(x, df=5)
        assert chi_squared.pdf(x) == pytest.approx(expected, rel=1e-8)

    def test_cdf_matches_scipy(self, chi_squared):
        x = 5.0
        expected = sp.chi2.cdf(x, df=5)
        assert chi_squared.cdf(x) == pytest.approx(expected, rel=1e-6)

    def test_ppf_matches_scipy(self, chi_squared):
        p = 0.95
        expected = sp.chi2.ppf(p, df=5)
        assert chi_squared.ppf(p) == pytest.approx(expected, rel=1e-4)


class TestChiSquaredBatch:
    def test_batch_pdf_matches_scalar(self, chi_squared):
        x = np.array([1.0, 3.0, 5.0, 8.0, 12.0])
        batch = chi_squared.pdf(x)
        scalar = np.array([chi_squared.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestChiSquaredProperties:
    def test_mean(self, chi_squared):
        assert chi_squared.mean == pytest.approx(5.0, rel=1e-10)

    def test_variance(self, chi_squared):
        # variance = 2k = 10
        assert chi_squared.variance == pytest.approx(10.0, rel=1e-10)

    def test_skewness(self, chi_squared):
        # skewness = sqrt(8/k) = sqrt(8/5)
        assert chi_squared.skewness == pytest.approx(math.sqrt(8.0 / 5.0), rel=1e-8)

    def test_support(self, chi_squared):
        lower, upper = chi_squared.support
        assert lower >= 0.0
        assert upper == float("inf")

    def test_is_not_discrete(self, chi_squared):
        assert not chi_squared.is_discrete


class TestChiSquaredFitAndSample:
    def test_sample_nonnegative(self, chi_squared):
        samples = chi_squared.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)

    def test_sample_shape(self, chi_squared):
        assert chi_squared.sample(n=500, seed=1).shape == (500,)
