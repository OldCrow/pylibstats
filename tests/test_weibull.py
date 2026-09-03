"""Tests for pylibstats.Weibull (WeibullDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestWeibullConstruction:
    def test_default_params(self):
        d = pylibstats.Weibull()
        assert d.shape == pytest.approx(1.0)
        assert d.scale == pytest.approx(1.0)

    def test_custom_params(self):
        d = pylibstats.Weibull(shape=2.0, scale=3.0)
        assert d.shape == pytest.approx(2.0)
        assert d.scale == pytest.approx(3.0)

    def test_zero_shape_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Weibull(0.0, 1.0)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Weibull(1.0, -1.0)


class TestWeibullScalar:
    def test_cdf_at_zero(self, weibull):
        assert weibull.cdf(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_log_pdf_consistency(self, weibull):
        x = 1.5
        assert weibull.log_pdf(x) == pytest.approx(math.log(weibull.pdf(x)), rel=1e-10)

    def test_ppf_round_trip(self, weibull):
        for p in [0.1, 0.5, 0.9]:
            assert weibull.cdf(weibull.ppf(p)) == pytest.approx(p, rel=1e-8)

    def test_pdf_negative_is_zero(self, weibull):
        assert weibull.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)


class TestWeibullBatch:
    def test_batch_shape(self, weibull):
        x = np.linspace(0.0, 5.0, 500)
        assert weibull.pdf(x).shape == (500,)

    def test_batch_matches_scalar(self, weibull):
        x = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(weibull.pdf(x), [weibull.pdf(v) for v in x], rtol=1e-12)


class TestWeibullProperties:
    def test_mean_shape2_scale1(self, weibull):
        # W(2,1): mean = Gamma(1+1/2) = Gamma(1.5) = sqrt(pi)/2
        assert weibull.mean == pytest.approx(math.sqrt(math.pi) / 2.0, rel=1e-8)

    def test_support(self, weibull):
        lower, upper = weibull.support
        assert lower == pytest.approx(0.0, abs=1e-10)
        assert upper == math.inf

    def test_is_not_discrete(self, weibull):
        assert not weibull.is_discrete


class TestWeibullFitAndSample:
    def test_sample_non_negative(self, weibull):
        assert np.all(weibull.sample(n=500, seed=4) >= 0.0)

    def test_fit_recovers_params(self):
        d = pylibstats.Weibull(2.0, 3.0)
        data = d.sample(n=50_000, seed=9)
        fitted = pylibstats.Weibull()
        fitted.fit(data)
        assert fitted.shape == pytest.approx(2.0, rel=0.1)
        assert fitted.scale == pytest.approx(3.0, rel=0.1)
