"""Tests for pylibstats.Gamma (GammaDistribution bindings)."""

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestGammaConstruction:
    def test_default_params(self):
        dist = pylibstats.Gamma()
        assert dist.alpha == pytest.approx(1.0)
        assert dist.beta == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Gamma(alpha=3.0, beta=2.0)
        assert dist.alpha == pytest.approx(3.0)
        assert dist.beta == pytest.approx(2.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gamma(alpha=0.0, beta=1.0)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gamma(alpha=1.0, beta=-1.0)


class TestGammaScalar:
    def test_pdf_matches_scipy(self, gamma_dist):
        # Gamma(alpha=2, beta=1) → scipy: shape=2, scale=1/1=1
        x = 1.5
        expected = sp.gamma.pdf(x, a=2.0, scale=1.0)
        assert gamma_dist.pdf(x) == pytest.approx(expected, rel=1e-8)

    def test_cdf_matches_scipy(self, gamma_dist):
        x = 2.0
        expected = sp.gamma.cdf(x, a=2.0, scale=1.0)
        assert gamma_dist.cdf(x) == pytest.approx(expected, rel=1e-8)

    def test_pdf_at_zero(self, gamma_dist):
        # Gamma(2,1) PDF at 0 = 0 (since alpha > 1)
        assert gamma_dist.pdf(0.0) == pytest.approx(0.0, abs=1e-12)


class TestGammaBatch:
    def test_batch_pdf_matches_scalar(self, gamma_dist):
        x = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        batch = gamma_dist.pdf(x)
        scalar = np.array([gamma_dist.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestGammaProperties:
    def test_mean(self, gamma_dist):
        # mean = alpha/beta = 2/1 = 2
        assert gamma_dist.mean == pytest.approx(2.0, rel=1e-10)

    def test_variance(self, gamma_dist):
        # variance = alpha/beta^2 = 2/1 = 2
        assert gamma_dist.variance == pytest.approx(2.0, rel=1e-10)

    def test_skewness(self, gamma_dist):
        # skewness = 2/sqrt(alpha) = 2/sqrt(2)
        import math

        assert gamma_dist.skewness == pytest.approx(2.0 / math.sqrt(2.0), rel=1e-8)

    def test_support(self, gamma_dist):
        lower, upper = gamma_dist.support
        assert lower >= 0.0
        assert upper == float("inf")

    def test_is_not_discrete(self, gamma_dist):
        assert not gamma_dist.is_discrete


class TestGammaFitAndSample:
    def test_sample_nonnegative(self, gamma_dist):
        samples = gamma_dist.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)

    def test_sample_shape(self, gamma_dist):
        assert gamma_dist.sample(n=300, seed=1).shape == (300,)
