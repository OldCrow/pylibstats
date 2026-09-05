"""Tests for pylibstats.FisherF (FDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestFisherFConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.FisherF()
        assert dist.d1 == pytest.approx(1.0)
        assert dist.d2 == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.FisherF(d1=3.0, d2=5.0)
        assert dist.d1 == pytest.approx(3.0)
        assert dist.d2 == pytest.approx(5.0)

    def test_zero_d1_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(0.0, 5.0)

    def test_negative_d1_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(-1.0, 5.0)

    def test_inf_d1_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(math.inf, 5.0)

    def test_nan_d1_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(float("nan"), 5.0)

    def test_zero_d2_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(3.0, 0.0)

    def test_negative_d2_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(3.0, -1.0)

    def test_inf_d2_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(3.0, math.inf)

    def test_nan_d2_raises(self):
        with pytest.raises(ValueError):
            pylibstats.FisherF(3.0, float("nan"))


class TestFisherFScalar:
    """Scalar PDF, CDF, log_pdf, ppf, cross-checked against scipy.stats.f."""

    def test_pdf_matches_scipy(self, fisher_f):
        for x in (0.5, 1.0, 2.0):
            expected = sp.f.pdf(x, 3, 5)
            assert fisher_f.pdf(x) == pytest.approx(expected, rel=1e-8)

    def test_cdf_matches_scipy(self, fisher_f):
        for x in (0.5, 1.0, 2.0):
            expected = sp.f.cdf(x, 3, 5)
            assert fisher_f.cdf(x) == pytest.approx(expected, rel=1e-6)

    def test_log_pdf_consistency(self, fisher_f):
        x = 1.5
        assert fisher_f.log_pdf(x) == pytest.approx(math.log(fisher_f.pdf(x)), rel=1e-10)

    def test_ppf_matches_scipy(self, fisher_f):
        p = 0.9
        expected = sp.f.ppf(p, 3, 5)
        assert fisher_f.ppf(p) == pytest.approx(expected, rel=1e-5)

    def test_ppf_cdf_round_trip(self, fisher_f):
        for x in (0.25, 1.0, 3.0):
            p = fisher_f.cdf(x)
            assert fisher_f.ppf(p) == pytest.approx(x, rel=1e-6)

    def test_pdf_outside_support(self, fisher_f):
        assert fisher_f.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)


class TestFisherFBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, fisher_f):
        x = np.linspace(0.01, 10, 1000)
        result = fisher_f.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, fisher_f):
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 8.0])
        batch_result = fisher_f.pdf(x)
        scalar_results = np.array([fisher_f.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, fisher_f):
        x = np.array([0.5, 1.0, 3.0])
        batch_result = fisher_f.cdf(x)
        scalar_results = np.array([fisher_f.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestFisherFProperties:
    """Moment properties and metadata, cross-checked against scipy."""

    def test_mean_matches_scipy(self, fisher_f):
        expected = sp.f.mean(3, 5)
        assert fisher_f.mean == pytest.approx(expected, rel=1e-10)

    def test_variance_matches_scipy(self, fisher_f):
        expected = sp.f.var(3, 5)
        assert fisher_f.variance == pytest.approx(expected, rel=1e-10)

    def test_support(self, fisher_f):
        lower, upper = fisher_f.support
        assert lower == pytest.approx(0.0)
        assert upper == math.inf

    def test_is_not_discrete(self, fisher_f):
        assert not fisher_f.is_discrete


class TestFisherFFitAndSample:
    """Fitting and sampling."""

    def test_sample_nonnegative(self, fisher_f):
        samples = fisher_f.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)

    def test_sample_shape(self, fisher_f):
        assert fisher_f.sample(n=200, seed=1).shape == (200,)

    def test_sample_reproducible(self, fisher_f):
        s1 = fisher_f.sample(n=100, seed=17)
        s2 = fisher_f.sample(n=100, seed=17)
        np.testing.assert_array_equal(s1, s2)


class TestFisherFRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.FisherF(3.0, 5.0)
        r = repr(dist)
        assert "3" in r
        assert "5" in r
