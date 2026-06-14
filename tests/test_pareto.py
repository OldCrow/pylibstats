"""Tests for pylibstats.Pareto (ParetoDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestParetoConstruction:
    def test_default_params(self):
        d = pylibstats.Pareto()
        assert d.scale == pytest.approx(1.0)
        assert d.alpha == pytest.approx(1.0)

    def test_custom_params(self):
        d = pylibstats.Pareto(scale=2.0, alpha=3.0)
        assert d.scale == pytest.approx(2.0)
        assert d.alpha == pytest.approx(3.0)

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Pareto(0.0, 1.0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Pareto(1.0, -1.0)


class TestParetoScalar:
    def test_pdf_at_scale(self, pareto):
        # Pareto(1, 2): PDF(1) = alpha/scale = 2
        assert pareto.pdf(1.0) == pytest.approx(2.0, rel=1e-10)

    def test_cdf_at_scale_is_zero(self, pareto):
        # CDF(scale) = 0
        assert pareto.cdf(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_cdf_below_scale_is_zero(self, pareto):
        assert pareto.cdf(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_log_pdf_consistency(self, pareto):
        x = 2.0
        assert pareto.log_pdf(x) == pytest.approx(math.log(pareto.pdf(x)), rel=1e-10)

    def test_ppf_round_trip(self, pareto):
        for p in [0.1, 0.5, 0.9]:
            assert pareto.cdf(pareto.ppf(p)) == pytest.approx(p, rel=1e-8)


class TestParetoBatch:
    def test_batch_shape(self, pareto):
        x = np.linspace(1.0, 10.0, 500)
        assert pareto.pdf(x).shape == (500,)

    def test_batch_matches_scalar(self, pareto):
        x = np.array([1.0, 2.0, 5.0, 10.0])
        np.testing.assert_allclose(pareto.pdf(x),
                                   [pareto.pdf(v) for v in x], rtol=1e-12)


class TestParetoProperties:
    def test_mean(self, pareto):
        # Pareto(1, 2): mean = alpha*scale/(alpha-1) = 2
        assert pareto.mean == pytest.approx(2.0, rel=1e-10)

    def test_support_lower(self, pareto):
        lower, _ = pareto.support
        assert lower == pytest.approx(1.0)

    def test_is_not_discrete(self, pareto):
        assert not pareto.is_discrete


class TestParetoFitAndSample:
    def test_sample_above_scale(self, pareto):
        samples = pareto.sample(n=500, seed=3)
        assert np.all(samples >= 1.0)

    def test_fit_recovers_params(self):
        d = pylibstats.Pareto(2.0, 3.0)
        data = d.sample(n=50_000, seed=8)
        fitted = pylibstats.Pareto()
        fitted.fit(data)
        assert fitted.scale == pytest.approx(2.0, rel=0.05)
        assert fitted.alpha == pytest.approx(3.0, rel=0.1)
