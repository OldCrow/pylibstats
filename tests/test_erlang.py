"""Tests for pylibstats.Erlang (ErlangDistribution bindings).

libstats' Erlang(k, lam) uses lam as a RATE, matching
scipy.stats.erlang(k, scale=1/lam) -- verified empirically, not 1*lam.
"""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestErlangConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.Erlang()
        assert dist.k == 1
        assert dist.lam == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Erlang(k=3, lam=2.0)
        assert dist.k == 3
        assert dist.lam == pytest.approx(2.0)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Erlang(0, 1.0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Erlang(-1, 1.0)

    def test_non_integer_k_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Erlang(2.5, 1.0)

    def test_zero_lam_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Erlang(1, 0.0)

    def test_negative_lam_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Erlang(1, -1.0)

    def test_inf_lam_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Erlang(1, math.inf)

    def test_nan_lam_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Erlang(1, float("nan"))


class TestErlangScalar:
    """Scalar PDF, CDF, log_pdf, ppf, cross-checked against scipy.stats.erlang."""

    def test_pdf_matches_scipy(self, erlang):
        for x in (0.5, 1.0, 2.0):
            expected = sp.erlang.pdf(x, 3, scale=1.0 / 2.0)
            assert erlang.pdf(x) == pytest.approx(expected, rel=1e-8)

    def test_cdf_matches_scipy(self, erlang):
        for x in (0.5, 1.0, 2.0):
            expected = sp.erlang.cdf(x, 3, scale=1.0 / 2.0)
            assert erlang.cdf(x) == pytest.approx(expected, rel=1e-6)

    def test_log_pdf_consistency(self, erlang):
        x = 1.5
        assert erlang.log_pdf(x) == pytest.approx(math.log(erlang.pdf(x)), rel=1e-10)

    def test_ppf_matches_scipy(self, erlang):
        p = 0.9
        expected = sp.erlang.ppf(p, 3, scale=1.0 / 2.0)
        assert erlang.ppf(p) == pytest.approx(expected, rel=1e-6)

    def test_ppf_cdf_round_trip(self, erlang):
        for x in (0.25, 1.0, 3.0):
            p = erlang.cdf(x)
            assert erlang.ppf(p) == pytest.approx(x, rel=1e-6)

    def test_pdf_outside_support(self, erlang):
        assert erlang.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)


class TestErlangBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, erlang):
        x = np.linspace(0.01, 10, 1000)
        result = erlang.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, erlang):
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 8.0])
        batch_result = erlang.pdf(x)
        scalar_results = np.array([erlang.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, erlang):
        x = np.array([0.5, 1.0, 3.0])
        batch_result = erlang.cdf(x)
        scalar_results = np.array([erlang.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestErlangProperties:
    """Moment properties and metadata, cross-checked against scipy."""

    def test_mean_matches_scipy(self, erlang):
        expected = sp.erlang.mean(3, scale=1.0 / 2.0)
        assert erlang.mean == pytest.approx(expected, rel=1e-10)

    def test_variance_matches_scipy(self, erlang):
        expected = sp.erlang.var(3, scale=1.0 / 2.0)
        assert erlang.variance == pytest.approx(expected, rel=1e-10)

    def test_support(self, erlang):
        lower, upper = erlang.support
        assert lower == pytest.approx(0.0)
        assert upper == math.inf

    def test_is_not_discrete(self, erlang):
        assert not erlang.is_discrete


class TestErlangFitAndSample:
    """Fitting and sampling."""

    def test_sample_nonnegative(self, erlang):
        samples = erlang.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)

    def test_sample_shape(self, erlang):
        assert erlang.sample(n=200, seed=1).shape == (200,)

    def test_sample_reproducible(self, erlang):
        s1 = erlang.sample(n=100, seed=17)
        s2 = erlang.sample(n=100, seed=17)
        np.testing.assert_array_equal(s1, s2)


class TestErlangRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.Erlang(3, 2.0)
        r = repr(dist)
        assert "3" in r
        assert "2" in r
