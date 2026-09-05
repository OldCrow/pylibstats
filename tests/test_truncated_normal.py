"""Tests for pylibstats.TruncatedNormal (TruncatedNormalDistribution bindings).

Truncation bounds a and b are ABSOLUTE coordinates, unlike scipy.stats.truncnorm,
whose a/b are standardized: truncnorm((a - mu) / sigma, (b - mu) / sigma,
loc=mu, scale=sigma).
"""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


def _scipy_ref(d):
    alpha = (d.a - d.mu) / d.sigma
    beta = (d.b - d.mu) / d.sigma
    return sp.truncnorm(alpha, beta, loc=d.mu, scale=d.sigma)


class TestTruncatedNormalConstruction:
    def test_default_params(self):
        d = pylibstats.TruncatedNormal()
        assert d.mu == pytest.approx(0.0)
        assert d.sigma == pytest.approx(1.0)
        assert d.a == -math.inf
        assert d.b == math.inf

    def test_custom_params(self):
        d = pylibstats.TruncatedNormal(mu=1.0, sigma=2.0, a=-1.0, b=4.0)
        assert d.mu == pytest.approx(1.0)
        assert d.sigma == pytest.approx(2.0)
        assert d.a == pytest.approx(-1.0)
        assert d.b == pytest.approx(4.0)

    def test_one_sided_truncation_valid(self):
        d = pylibstats.TruncatedNormal(a=0.0)
        assert d.a == pytest.approx(0.0)
        assert d.b == math.inf

    def test_inf_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(mu=math.inf)

    def test_nan_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(mu=float("nan"))

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(sigma=-1.0)

    def test_nan_bound_raises(self):
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(a=float("nan"))
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(b=float("nan"))

    def test_reversed_bounds_raise(self):
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(a=1.0, b=-1.0)

    def test_equal_bounds_raise(self):
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(a=1.0, b=1.0)

    def test_underflowing_window_raises(self):
        # Z = Phi(beta) - Phi(alpha) underflows double for a same-tail window
        # this deep (~40 sigma); the constructor must reject it (#57 contract).
        with pytest.raises(ValueError):
            pylibstats.TruncatedNormal(mu=0.0, sigma=1.0, a=40.0, b=41.0)

    def test_setter_validates_window(self):
        d = pylibstats.TruncatedNormal(a=-1.0, b=1.0)
        with pytest.raises(ValueError):
            d.a = 2.0  # would cross above b


class TestTruncatedNormalScalar:
    def test_pdf_zero_outside_window(self, truncated_normal):
        assert truncated_normal.pdf(truncated_normal.a - 0.5) == 0.0
        assert truncated_normal.pdf(truncated_normal.b + 0.5) == 0.0

    def test_pdf_matches_scipy(self, truncated_normal):
        ref = _scipy_ref(truncated_normal)
        for x in (-0.5, 0.0, 0.7, 1.4):
            assert truncated_normal.pdf(x) == pytest.approx(ref.pdf(x), rel=1e-10)

    def test_cdf_matches_scipy(self, truncated_normal):
        ref = _scipy_ref(truncated_normal)
        for x in (-0.5, 0.0, 0.7, 1.4):
            assert truncated_normal.cdf(x) == pytest.approx(ref.cdf(x), rel=1e-10)

    def test_cdf_saturates_at_bounds(self, truncated_normal):
        assert truncated_normal.cdf(truncated_normal.a) == pytest.approx(0.0, abs=1e-15)
        assert truncated_normal.cdf(truncated_normal.b) == pytest.approx(1.0, rel=1e-12)

    def test_log_pdf_consistency(self, truncated_normal):
        x = 0.25
        assert truncated_normal.log_pdf(x) == pytest.approx(
            math.log(truncated_normal.pdf(x)), rel=1e-10
        )

    def test_ppf_round_trip(self, truncated_normal):
        for p in (0.1, 0.5, 0.9):
            x = truncated_normal.ppf(p)
            assert truncated_normal.cdf(x) == pytest.approx(p, rel=1e-9)

    def test_same_tail_window(self):
        # Moderately deep same-tail truncation exercises the erfc-difference
        # survival form (#49-class cancellation is the hazard being guarded).
        d = pylibstats.TruncatedNormal(mu=0.0, sigma=1.0, a=5.0, b=7.0)
        ref = sp.truncnorm(5.0, 7.0)
        for x in (5.1, 5.5, 6.0):
            assert d.pdf(x) == pytest.approx(ref.pdf(x), rel=1e-8)
            assert d.cdf(x) == pytest.approx(ref.cdf(x), rel=1e-8)


class TestTruncatedNormalBatch:
    def test_batch_shape(self, truncated_normal):
        x = np.linspace(-1.5, 1.5, 7)
        assert truncated_normal.pdf(x).shape == x.shape

    def test_batch_matches_scalar(self, truncated_normal):
        x = np.linspace(truncated_normal.a - 0.5, truncated_normal.b + 0.5, 11)
        batch = truncated_normal.pdf(x)
        for i, xi in enumerate(x):
            assert batch[i] == pytest.approx(truncated_normal.pdf(float(xi)), rel=1e-14)

    def test_batch_cdf_matches_scalar(self, truncated_normal):
        x = np.linspace(-2.0, 2.0, 9)
        batch = truncated_normal.cdf(x)
        for i, xi in enumerate(x):
            assert batch[i] == pytest.approx(truncated_normal.cdf(float(xi)), rel=1e-14)

    def test_batch_nan_propagates(self, truncated_normal):
        out = truncated_normal.pdf(np.array([0.0, np.nan, 0.5]))
        assert math.isnan(out[1])
        assert not math.isnan(out[0]) and not math.isnan(out[2])


class TestTruncatedNormalProperties:
    def test_mean_matches_scipy(self, truncated_normal):
        ref = _scipy_ref(truncated_normal)
        assert truncated_normal.mean == pytest.approx(ref.mean(), rel=1e-10)

    def test_variance_matches_scipy(self, truncated_normal):
        ref = _scipy_ref(truncated_normal)
        assert truncated_normal.variance == pytest.approx(ref.var(), rel=1e-10)

    def test_untruncated_matches_gaussian(self):
        d = pylibstats.TruncatedNormal(mu=2.0, sigma=3.0)
        g = pylibstats.Gaussian(mu=2.0, sigma=3.0)
        for x in (-1.0, 2.0, 5.0):
            assert d.pdf(x) == pytest.approx(g.pdf(x), rel=1e-12)
        assert d.mean == pytest.approx(2.0, rel=1e-12)

    def test_is_continuous(self, truncated_normal):
        assert not truncated_normal.is_discrete


class TestTruncatedNormalSampleFit:
    def test_samples_within_window(self, truncated_normal):
        samples = truncated_normal.sample(500)
        assert np.all(samples >= truncated_normal.a)
        assert np.all(samples <= truncated_normal.b)

    def test_fit_smoke(self, truncated_normal):
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, 2000)
        data = data[(data > -1.5) & (data < 1.5)]
        d = pylibstats.TruncatedNormal(a=-1.5, b=1.5)
        d.fit(data)
        assert math.isfinite(d.mu)
        assert d.sigma > 0.0
