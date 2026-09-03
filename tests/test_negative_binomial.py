"""Tests for pylibstats.NegativeBinomial (NegativeBinomialDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestNegativeBinomialConstruction:
    def test_default_params(self):
        d = pylibstats.NegativeBinomial()
        assert d.r == pytest.approx(1.0)
        assert d.p == pytest.approx(0.5)

    def test_custom_params(self):
        d = pylibstats.NegativeBinomial(r=3.0, p=0.6)
        assert d.r == pytest.approx(3.0)
        assert d.p == pytest.approx(0.6)

    def test_real_r_valid(self):
        d = pylibstats.NegativeBinomial(r=1.5, p=0.7)
        assert d.r == pytest.approx(1.5)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError):
            pylibstats.NegativeBinomial(0.0, 0.5)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            pylibstats.NegativeBinomial(-1.0, 0.5)

    def test_zero_p_raises(self):
        with pytest.raises(ValueError):
            pylibstats.NegativeBinomial(1.0, 0.0)

    def test_p_above_one_raises(self):
        with pytest.raises(ValueError):
            pylibstats.NegativeBinomial(1.0, 1.1)

    def test_p_one_valid(self):
        d = pylibstats.NegativeBinomial(2.0, 1.0)
        assert d.p == pytest.approx(1.0)


class TestNegativeBinomialScalar:
    def test_pmf_at_zero(self, negative_binomial):
        # NB(2, 0.5): PMF(0) = p^r = 0.5^2 = 0.25
        assert negative_binomial.pdf(0.0) == pytest.approx(0.25, rel=1e-10)

    def test_pmf_at_one(self, negative_binomial):
        # NB(2, 0.5): PMF(1) = C(2,1)*(0.5)^2*(0.5) = 0.25
        assert negative_binomial.pdf(1.0) == pytest.approx(0.25, rel=1e-10)

    def test_pmf_negative_is_zero(self, negative_binomial):
        assert negative_binomial.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)

    def test_cdf_at_zero(self, negative_binomial):
        # CDF(0) = PMF(0) = 0.25
        assert negative_binomial.cdf(0.0) == pytest.approx(0.25, rel=1e-8)

    def test_cdf_at_one(self, negative_binomial):
        # CDF(1) = PMF(0) + PMF(1) = 0.5
        assert negative_binomial.cdf(1.0) == pytest.approx(0.5, rel=1e-8)

    def test_log_pdf_consistency(self, negative_binomial):
        pmf0 = negative_binomial.pdf(0.0)
        assert negative_binomial.log_pdf(0.0) == pytest.approx(math.log(pmf0), rel=1e-10)


class TestNegativeBinomialBatch:
    def test_batch_shape(self, negative_binomial):
        x = np.arange(0, 20, dtype=float)
        assert negative_binomial.pdf(x).shape == (20,)

    def test_batch_matches_scalar(self, negative_binomial):
        x = np.array([0.0, 1.0, 2.0, 5.0])
        np.testing.assert_allclose(
            negative_binomial.pdf(x), [negative_binomial.pdf(v) for v in x], rtol=1e-12
        )


class TestNegativeBinomialProperties:
    def test_mean(self, negative_binomial):
        # NB(2, 0.5): mean = r(1-p)/p = 2
        assert negative_binomial.mean == pytest.approx(2.0, rel=1e-10)

    def test_variance(self, negative_binomial):
        # NB(2, 0.5): variance = r(1-p)/p^2 = 4
        assert negative_binomial.variance == pytest.approx(4.0, rel=1e-10)

    def test_is_discrete(self, negative_binomial):
        assert negative_binomial.is_discrete

    def test_support_lower(self, negative_binomial):
        lower, _ = negative_binomial.support
        assert lower == pytest.approx(0.0)


class TestNegativeBinomialFitAndSample:
    def test_sample_non_negative(self, negative_binomial):
        assert np.all(negative_binomial.sample(n=500, seed=14) >= 0)

    def test_fit_recovers_params(self):
        d = pylibstats.NegativeBinomial(3.0, 0.6)
        data = d.sample(n=10_000, seed=15)
        fitted = pylibstats.NegativeBinomial()
        fitted.fit(data)
        assert fitted.r == pytest.approx(3.0, rel=0.2)
        assert fitted.p == pytest.approx(0.6, abs=0.1)

    def test_real_r_sampling(self):
        d = pylibstats.NegativeBinomial(r=1.5, p=0.6)
        s = d.sample(n=200, seed=16)
        assert np.all(s >= 0)
