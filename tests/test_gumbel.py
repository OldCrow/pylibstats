"""Tests for pylibstats.Gumbel (GumbelDistribution bindings).

libstats' Gumbel is the maximum-stable (right-skewed) variant, matching
scipy.stats.gumbel_r(loc=mu, scale=beta) -- NOT gumbel_l.
"""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestGumbelConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.Gumbel()
        assert dist.mu == pytest.approx(0.0)
        assert dist.beta == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Gumbel(mu=1.0, beta=2.0)
        assert dist.mu == pytest.approx(1.0)
        assert dist.beta == pytest.approx(2.0)

    def test_zero_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gumbel(0.0, 0.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gumbel(0.0, -1.0)

    def test_inf_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gumbel(0.0, math.inf)

    def test_nan_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gumbel(0.0, float("nan"))

    def test_inf_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gumbel(math.inf, 1.0)

    def test_nan_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Gumbel(float("nan"), 1.0)


class TestGumbelScalar:
    """Scalar PDF, CDF, log_pdf, ppf, cross-checked against scipy.stats.gumbel_r."""

    def test_pdf_matches_scipy(self, gumbel):
        for x in (-1.0, 1.0, 3.0):
            expected = sp.gumbel_r.pdf(x, loc=1.0, scale=2.0)
            assert gumbel.pdf(x) == pytest.approx(expected, rel=1e-10)

    def test_cdf_matches_scipy(self, gumbel):
        for x in (-1.0, 1.0, 3.0):
            expected = sp.gumbel_r.cdf(x, loc=1.0, scale=2.0)
            assert gumbel.cdf(x) == pytest.approx(expected, rel=1e-10)

    def test_log_pdf_consistency(self, gumbel):
        x = 1.5
        assert gumbel.log_pdf(x) == pytest.approx(math.log(gumbel.pdf(x)), rel=1e-10)

    def test_ppf_matches_scipy(self, gumbel):
        p = 0.75
        expected = sp.gumbel_r.ppf(p, loc=1.0, scale=2.0)
        assert gumbel.ppf(p) == pytest.approx(expected, rel=1e-8)

    def test_ppf_cdf_round_trip(self, gumbel):
        for x in (-1.0, 0.5, 3.0):
            p = gumbel.cdf(x)
            assert gumbel.ppf(p) == pytest.approx(x, rel=1e-8)


class TestGumbelBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, gumbel):
        x = np.linspace(-5, 10, 1000)
        result = gumbel.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, gumbel):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        batch_result = gumbel.pdf(x)
        scalar_results = np.array([gumbel.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, gumbel):
        x = np.array([-3.0, 0.0, 3.0])
        batch_result = gumbel.cdf(x)
        scalar_results = np.array([gumbel.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestGumbelProperties:
    """Moment properties and metadata, cross-checked against scipy."""

    def test_mean_matches_scipy(self, gumbel):
        expected = sp.gumbel_r.mean(loc=1.0, scale=2.0)
        assert gumbel.mean == pytest.approx(expected, rel=1e-10)

    def test_variance_matches_scipy(self, gumbel):
        expected = sp.gumbel_r.var(loc=1.0, scale=2.0)
        assert gumbel.variance == pytest.approx(expected, rel=1e-10)

    def test_skewness_matches_scipy(self, gumbel):
        expected = sp.gumbel_r.stats(loc=1.0, scale=2.0, moments="s")
        assert gumbel.skewness == pytest.approx(float(expected), rel=1e-6)

    def test_support(self, gumbel):
        lower, upper = gumbel.support
        assert lower == -math.inf
        assert upper == math.inf

    def test_is_not_discrete(self, gumbel):
        assert not gumbel.is_discrete


class TestGumbelFitAndSample:
    """Fitting and sampling."""

    def test_sample_shape(self, gumbel):
        samples = gumbel.sample(n=500, seed=42)
        assert samples.shape == (500,)

    def test_sample_reproducible(self, gumbel):
        s1 = gumbel.sample(n=100, seed=17)
        s2 = gumbel.sample(n=100, seed=17)
        np.testing.assert_array_equal(s1, s2)

    def test_fit_smoke(self):
        dist = pylibstats.Gumbel(mu=3.0, beta=1.5)
        data = dist.sample(n=50_000, seed=77)
        fitted = pylibstats.Gumbel()
        fitted.fit(data)
        assert fitted.mu == pytest.approx(3.0, abs=0.1)
        assert fitted.beta == pytest.approx(1.5, abs=0.1)


class TestGumbelRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.Gumbel(1.0, 2.0)
        r = repr(dist)
        assert "1" in r
        assert "2" in r
