"""Tests for pylibstats.LogNormal (LogNormalDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestLogNormalConstruction:
    def test_default_params(self):
        d = pylibstats.LogNormal()
        assert d.mu == pytest.approx(0.0)
        assert d.sigma == pytest.approx(1.0)

    def test_custom_params(self):
        d = pylibstats.LogNormal(mu=1.0, sigma=2.0)
        assert d.mu == pytest.approx(1.0)
        assert d.sigma == pytest.approx(2.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.LogNormal(0.0, -1.0)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.LogNormal(0.0, 0.0)

    def test_inf_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.LogNormal(math.inf, 1.0)


class TestLogNormalScalar:
    def test_cdf_at_median(self, lognormal):
        # LogN(0,1): median = exp(0) = 1, CDF(1) = 0.5
        assert lognormal.cdf(1.0) == pytest.approx(0.5, rel=1e-10)

    def test_pdf_at_one(self, lognormal):
        # LogN(0,1): PDF(1) = 1/sqrt(2pi) ≈ 0.3989
        assert lognormal.pdf(1.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), rel=1e-10)

    def test_log_pdf_consistency(self, lognormal):
        x = 2.0
        assert lognormal.log_pdf(x) == pytest.approx(math.log(lognormal.pdf(x)), rel=1e-10)

    def test_ppf_median(self, lognormal):
        assert lognormal.ppf(0.5) == pytest.approx(1.0, rel=1e-8)

    def test_pdf_zero_or_negative_returns_zero(self, lognormal):
        assert lognormal.pdf(0.0) == pytest.approx(0.0, abs=1e-12)
        assert lognormal.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)


class TestLogNormalBatch:
    def test_batch_pdf_shape(self, lognormal):
        x = np.linspace(0.1, 5.0, 500)
        assert lognormal.pdf(x).shape == (500,)

    def test_batch_matches_scalar(self, lognormal):
        x = np.array([0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(lognormal.pdf(x),
                                   [lognormal.pdf(v) for v in x], rtol=1e-12)


class TestLogNormalProperties:
    def test_mean(self, lognormal):
        # LogN(0,1): mean = exp(0.5) ≈ 1.6487
        assert lognormal.mean == pytest.approx(math.exp(0.5), rel=1e-10)

    def test_support(self, lognormal):
        lower, upper = lognormal.support
        assert lower == pytest.approx(0.0, abs=1e-10)
        assert upper == math.inf

    def test_is_not_discrete(self, lognormal):
        assert not lognormal.is_discrete


class TestLogNormalFitAndSample:
    def test_sample_shape(self, lognormal):
        assert lognormal.sample(n=200, seed=1).shape == (200,)

    def test_sample_positive(self, lognormal):
        assert np.all(lognormal.sample(n=500, seed=2) > 0)

    def test_fit_recovers_params(self):
        d = pylibstats.LogNormal(1.0, 0.5)
        data = d.sample(n=50_000, seed=7)
        fitted = pylibstats.LogNormal()
        fitted.fit(data)
        assert fitted.mu == pytest.approx(1.0, abs=0.05)
        assert fitted.sigma == pytest.approx(0.5, abs=0.05)
