"""Tests for pylibstats.Rayleigh (RayleighDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestRayleighConstruction:
    def test_default_params(self):
        assert pylibstats.Rayleigh().sigma == pytest.approx(1.0)

    def test_custom_params(self):
        assert pylibstats.Rayleigh(sigma=2.0).sigma == pytest.approx(2.0)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Rayleigh(0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Rayleigh(-1.0)


class TestRayleighScalar:
    def test_cdf_at_zero(self, rayleigh):
        assert rayleigh.cdf(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_pdf_at_mode(self, rayleigh):
        # R(1): PDF(sigma) = PDF(1) = (1/1)*exp(-0.5) ≈ 0.6065
        assert rayleigh.pdf(1.0) == pytest.approx(math.exp(-0.5), rel=1e-10)

    def test_log_pdf_consistency(self, rayleigh):
        x = 1.5
        assert rayleigh.log_pdf(x) == pytest.approx(math.log(rayleigh.pdf(x)), rel=1e-10)

    def test_ppf_round_trip(self, rayleigh):
        for p in [0.1, 0.5, 0.9]:
            assert rayleigh.cdf(rayleigh.ppf(p)) == pytest.approx(p, rel=1e-8)

    def test_pdf_negative_is_zero(self, rayleigh):
        assert rayleigh.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)


class TestRayleighBatch:
    def test_batch_shape(self, rayleigh):
        assert rayleigh.pdf(np.linspace(0.0, 5.0, 300)).shape == (300,)

    def test_batch_matches_scalar(self, rayleigh):
        x = np.array([0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(rayleigh.pdf(x),
                                   [rayleigh.pdf(v) for v in x], rtol=1e-12)


class TestRayleighProperties:
    def test_mean(self, rayleigh):
        # R(1): mean = sigma*sqrt(pi/2) = sqrt(pi/2)
        assert rayleigh.mean == pytest.approx(math.sqrt(math.pi / 2.0), rel=1e-10)

    def test_support(self, rayleigh):
        lower, upper = rayleigh.support
        assert lower == pytest.approx(0.0, abs=1e-10)
        assert upper == math.inf

    def test_is_not_discrete(self, rayleigh):
        assert not rayleigh.is_discrete


class TestRayleighFitAndSample:
    def test_sample_non_negative(self, rayleigh):
        assert np.all(rayleigh.sample(n=500, seed=5) >= 0.0)

    def test_fit_recovers_sigma(self):
        d = pylibstats.Rayleigh(2.0)
        data = d.sample(n=50_000, seed=10)
        fitted = pylibstats.Rayleigh()
        fitted.fit(data)
        assert fitted.sigma == pytest.approx(2.0, rel=0.05)
