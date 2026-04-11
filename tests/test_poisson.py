"""Tests for pylibstats.Poisson (PoissonDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestPoissonConstruction:
    def test_default_params(self):
        dist = pylibstats.Poisson()
        assert dist.lam == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Poisson(lam=5.0)
        assert dist.lam == pytest.approx(5.0)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Poisson(-1.0)


class TestPoissonScalar:
    def test_pmf_at_mode(self, poisson):
        # Pois(3): P(X=3) via SciPy
        expected = sp.poisson.pmf(3, mu=3.0)
        assert poisson.pdf(3.0) == pytest.approx(expected, rel=1e-8)

    def test_pmf_at_zero(self, poisson):
        expected = math.exp(-3.0)
        assert poisson.pdf(0.0) == pytest.approx(expected, rel=1e-8)

    def test_cdf_matches_scipy(self, poisson):
        expected = sp.poisson.cdf(4, mu=3.0)
        assert poisson.cdf(4.0) == pytest.approx(expected, rel=1e-6)


class TestPoissonBatch:
    def test_batch_pdf_matches_scalar(self, poisson):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        batch = poisson.pdf(x)
        scalar = np.array([poisson.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestPoissonProperties:
    def test_mean(self, poisson):
        assert poisson.mean == pytest.approx(3.0, rel=1e-10)

    def test_variance(self, poisson):
        # Poisson: mean = variance = lambda
        assert poisson.variance == pytest.approx(3.0, rel=1e-10)

    def test_is_discrete(self, poisson):
        assert poisson.is_discrete

    def test_support(self, poisson):
        lower, _ = poisson.support
        assert lower >= 0.0


class TestPoissonFitAndSample:
    def test_sample_nonnegative(self, poisson):
        samples = poisson.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)

    def test_fit_recovers_params(self):
        dist = pylibstats.Poisson(lam=7.0)
        data = dist.sample(n=50_000, seed=99)
        fitted = pylibstats.Poisson()
        fitted.fit(data)
        assert fitted.lam == pytest.approx(7.0, rel=0.05)
