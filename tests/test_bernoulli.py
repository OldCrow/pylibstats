"""Tests for pylibstats.Bernoulli (BernoulliDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestBernoulliConstruction:
    def test_default_params(self):
        d = pylibstats.Bernoulli()
        assert d.p == pytest.approx(0.5)

    def test_custom_params(self):
        d = pylibstats.Bernoulli(p=0.3)
        assert d.p == pytest.approx(0.3)

    def test_p_zero_valid(self):
        # p in [0, 1] inclusive — Binomial's convention, adopted by #55.
        d = pylibstats.Bernoulli(0.0)
        assert d.p == pytest.approx(0.0)

    def test_p_one_valid(self):
        d = pylibstats.Bernoulli(1.0)
        assert d.p == pytest.approx(1.0)

    def test_p_above_one_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Bernoulli(1.5)

    def test_p_negative_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Bernoulli(-0.1)

    def test_p_nan_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Bernoulli(float("nan"))

    def test_p_setter_validates(self, bernoulli):
        with pytest.raises(ValueError):
            bernoulli.p = 2.0


class TestBernoulliScalar:
    def test_pmf_values(self, bernoulli):
        assert bernoulli.pdf(0.0) == pytest.approx(1.0 - bernoulli.p, rel=1e-12)
        assert bernoulli.pdf(1.0) == pytest.approx(bernoulli.p, rel=1e-12)

    def test_pmf_out_of_support_is_zero(self, bernoulli):
        assert bernoulli.pdf(2.0) == 0.0
        assert bernoulli.pdf(-1.0) == 0.0

    def test_pmf_matches_scipy(self, bernoulli):
        ref = sp.bernoulli(bernoulli.p)
        for k in (0, 1):
            assert bernoulli.pdf(float(k)) == pytest.approx(ref.pmf(k), rel=1e-12)

    def test_cdf_steps(self, bernoulli):
        q = 1.0 - bernoulli.p
        assert bernoulli.cdf(-0.5) == pytest.approx(0.0, abs=1e-15)
        assert bernoulli.cdf(0.0) == pytest.approx(q, rel=1e-12)
        assert bernoulli.cdf(0.5) == pytest.approx(q, rel=1e-12)
        assert bernoulli.cdf(1.0) == pytest.approx(1.0, rel=1e-12)

    def test_log_pdf_consistency(self, bernoulli):
        for k in (0.0, 1.0):
            assert bernoulli.log_pdf(k) == pytest.approx(math.log(bernoulli.pdf(k)), rel=1e-10)

    def test_ppf_right_continuous_inverse(self, bernoulli):
        # Q(p*) = min{k : F(k) >= p*} — the #104 discrete quantile contract.
        q = 1.0 - bernoulli.p
        assert bernoulli.ppf(q / 2.0) == pytest.approx(0.0)
        assert bernoulli.ppf(q) == pytest.approx(0.0)
        assert bernoulli.ppf(min(q + 1e-9, 1.0)) == pytest.approx(1.0)


class TestBernoulliBatch:
    def test_batch_shape(self, bernoulli):
        x = np.array([0.0, 1.0, 0.0, 1.0, 2.0])
        assert bernoulli.pdf(x).shape == x.shape

    def test_batch_matches_scalar(self, bernoulli):
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        batch = bernoulli.pdf(x)
        for i, xi in enumerate(x):
            assert batch[i] == pytest.approx(bernoulli.pdf(float(xi)), rel=1e-14)

    def test_batch_cdf_matches_scalar(self, bernoulli):
        x = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        batch = bernoulli.cdf(x)
        for i, xi in enumerate(x):
            assert batch[i] == pytest.approx(bernoulli.cdf(float(xi)), rel=1e-14)

    def test_batch_nan_propagates(self, bernoulli):
        out = bernoulli.pdf(np.array([0.0, np.nan, 1.0]))
        assert math.isnan(out[1])
        assert not math.isnan(out[0]) and not math.isnan(out[2])


class TestBernoulliProperties:
    def test_moments(self, bernoulli):
        p = bernoulli.p
        assert bernoulli.mean == pytest.approx(p, rel=1e-12)
        assert bernoulli.variance == pytest.approx(p * (1.0 - p), rel=1e-12)

    def test_is_discrete(self, bernoulli):
        assert bernoulli.is_discrete

    def test_num_parameters(self, bernoulli):
        assert bernoulli.num_parameters == 1


class TestBernoulliSampleFit:
    def test_sample_values_are_zero_or_one(self, bernoulli):
        samples = bernoulli.sample(200)
        assert set(np.unique(samples)) <= {0.0, 1.0}

    def test_fit_recovers_p(self):
        rng = np.random.default_rng(42)
        data = (rng.random(5000) < 0.3).astype(np.float64)
        d = pylibstats.Bernoulli()
        d.fit(data)
        assert d.p == pytest.approx(data.mean(), abs=1e-12)
