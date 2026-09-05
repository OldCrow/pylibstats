"""Tests for pylibstats.InverseGamma (InverseGammaDistribution bindings).

libstats' InverseGamma(alpha, beta) parameterizes beta as a SCALE, matching
scipy.stats.invgamma(alpha, scale=beta) directly -- verified empirically.
The candidate scale=1/beta mapping was ruled out (see the probe values
below): with alpha=3, beta=2, scale=beta gives pdf(0.5)=1.1722008888789872
(matches the binding to 1e-12) while scale=1/beta gives 0.3678794411714423,
off by more than 3x.
"""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestInverseGammaConstruction:
    """Construction and parameter validation."""

    def test_default_params(self):
        dist = pylibstats.InverseGamma()
        assert dist.alpha == pytest.approx(1.0)
        assert dist.beta == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.InverseGamma(alpha=3.0, beta=2.0)
        assert dist.alpha == pytest.approx(3.0)
        assert dist.beta == pytest.approx(2.0)

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(0.0, 1.0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(-1.0, 1.0)

    def test_inf_alpha_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(math.inf, 1.0)

    def test_nan_alpha_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(float("nan"), 1.0)

    def test_zero_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(1.0, 0.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(1.0, -1.0)

    def test_inf_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(1.0, math.inf)

    def test_nan_beta_raises(self):
        with pytest.raises(ValueError):
            pylibstats.InverseGamma(1.0, float("nan"))


class TestInverseGammaScalar:
    """Scalar PDF, CDF, log_pdf, ppf, cross-checked against scipy.stats.invgamma."""

    def test_pdf_matches_scipy_scale_beta(self, inverse_gamma):
        # beta is a SCALE: scipy.stats.invgamma(alpha, scale=beta).
        for x in (0.5, 1.0, 2.0):
            expected = sp.invgamma.pdf(x, 3, scale=2.0)
            assert inverse_gamma.pdf(x) == pytest.approx(expected, rel=1e-8)

    def test_cdf_matches_scipy_scale_beta(self, inverse_gamma):
        for x in (0.5, 1.0, 2.0):
            expected = sp.invgamma.cdf(x, 3, scale=2.0)
            assert inverse_gamma.cdf(x) == pytest.approx(expected, rel=1e-6)

    def test_log_pdf_consistency(self, inverse_gamma):
        x = 1.5
        assert inverse_gamma.log_pdf(x) == pytest.approx(math.log(inverse_gamma.pdf(x)), rel=1e-10)

    def test_ppf_matches_scipy(self, inverse_gamma):
        p = 0.9
        expected = sp.invgamma.ppf(p, 3, scale=2.0)
        assert inverse_gamma.ppf(p) == pytest.approx(expected, rel=1e-6)

    def test_ppf_cdf_round_trip(self, inverse_gamma):
        for x in (0.25, 1.0, 3.0):
            p = inverse_gamma.cdf(x)
            assert inverse_gamma.ppf(p) == pytest.approx(x, rel=1e-6)

    def test_pdf_outside_support(self, inverse_gamma):
        assert inverse_gamma.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)


class TestInverseGammaBatch:
    """Batch (NumPy array) operations."""

    def test_batch_pdf_shape(self, inverse_gamma):
        x = np.linspace(0.01, 10, 1000)
        result = inverse_gamma.pdf(x)
        assert result.shape == (1000,)

    def test_batch_pdf_matches_scalar(self, inverse_gamma):
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 8.0])
        batch_result = inverse_gamma.pdf(x)
        scalar_results = np.array([inverse_gamma.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)

    def test_batch_cdf_matches_scalar(self, inverse_gamma):
        x = np.array([0.5, 1.0, 3.0])
        batch_result = inverse_gamma.cdf(x)
        scalar_results = np.array([inverse_gamma.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch_result, scalar_results, rtol=1e-12)


class TestInverseGammaProperties:
    """Moment properties and metadata, cross-checked against scipy."""

    def test_mean_matches_scipy(self, inverse_gamma):
        expected = sp.invgamma.mean(3, scale=2.0)
        assert inverse_gamma.mean == pytest.approx(expected, rel=1e-10)

    def test_variance_matches_scipy(self, inverse_gamma):
        expected = sp.invgamma.var(3, scale=2.0)
        assert inverse_gamma.variance == pytest.approx(expected, rel=1e-10)

    def test_support(self, inverse_gamma):
        lower, upper = inverse_gamma.support
        assert lower == pytest.approx(0.0)
        assert upper == math.inf

    def test_is_not_discrete(self, inverse_gamma):
        assert not inverse_gamma.is_discrete


class TestInverseGammaFitAndSample:
    """Fitting and sampling."""

    def test_sample_positive(self, inverse_gamma):
        samples = inverse_gamma.sample(n=1000, seed=42)
        assert np.all(samples > 0.0)

    def test_sample_shape(self, inverse_gamma):
        assert inverse_gamma.sample(n=200, seed=1).shape == (200,)

    def test_sample_reproducible(self, inverse_gamma):
        s1 = inverse_gamma.sample(n=100, seed=17)
        s2 = inverse_gamma.sample(n=100, seed=17)
        np.testing.assert_array_equal(s1, s2)


class TestInverseGammaRepr:
    """String representation."""

    def test_repr_contains_params(self):
        dist = pylibstats.InverseGamma(3.0, 2.0)
        r = repr(dist)
        assert "3" in r
        assert "2" in r
