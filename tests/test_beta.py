"""Tests for pylibstats.Beta (BetaDistribution bindings)."""

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestBetaConstruction:
    def test_default_params(self):
        dist = pylibstats.Beta()
        assert dist.alpha == pytest.approx(1.0)
        assert dist.beta == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Beta(alpha=2.0, beta=5.0)
        assert dist.alpha == pytest.approx(2.0)
        assert dist.beta == pytest.approx(5.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Beta(alpha=0.0, beta=1.0)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Beta(alpha=1.0, beta=-1.0)


class TestBetaScalar:
    def test_pdf_matches_scipy(self, beta_dist):
        x = 0.3
        expected = sp.beta.pdf(x, a=2.0, b=5.0)
        assert beta_dist.pdf(x) == pytest.approx(expected, rel=1e-8)

    def test_cdf_matches_scipy(self, beta_dist):
        x = 0.4
        expected = sp.beta.cdf(x, a=2.0, b=5.0)
        assert beta_dist.cdf(x) == pytest.approx(expected, rel=1e-6)

    def test_pdf_outside_support(self, beta_dist):
        assert beta_dist.pdf(-0.1) == pytest.approx(0.0, abs=1e-12)
        assert beta_dist.pdf(1.1) == pytest.approx(0.0, abs=1e-12)


class TestBetaBatch:
    def test_batch_pdf_matches_scalar(self, beta_dist):
        x = np.array([0.1, 0.2, 0.3, 0.5, 0.8])
        batch = beta_dist.pdf(x)
        scalar = np.array([beta_dist.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestBetaProperties:
    def test_mean(self, beta_dist):
        # mean = alpha/(alpha+beta) = 2/7
        assert beta_dist.mean == pytest.approx(2.0 / 7.0, rel=1e-10)

    def test_variance(self, beta_dist):
        # variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        #          = 2*5 / (49*8) = 10/392
        assert beta_dist.variance == pytest.approx(10.0 / 392.0, rel=1e-8)

    def test_support(self, beta_dist):
        lower, upper = beta_dist.support
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(1.0)

    def test_is_not_discrete(self, beta_dist):
        assert not beta_dist.is_discrete


class TestBetaFitAndSample:
    def test_sample_in_unit_interval(self, beta_dist):
        samples = beta_dist.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sample_shape(self, beta_dist):
        assert beta_dist.sample(n=200, seed=1).shape == (200,)
