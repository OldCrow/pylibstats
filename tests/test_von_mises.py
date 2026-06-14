"""Tests for pylibstats.VonMises (VonMisesDistribution bindings)."""

import math

import numpy as np
import pytest

import pylibstats


class TestVonMisesConstruction:
    def test_default_params(self):
        d = pylibstats.VonMises()
        assert d.mu == pytest.approx(0.0)
        assert d.kappa == pytest.approx(1.0)

    def test_custom_params(self):
        d = pylibstats.VonMises(mu=1.0, kappa=2.0)
        assert d.mu == pytest.approx(1.0)
        assert d.kappa == pytest.approx(2.0)

    def test_kappa_zero_valid(self):
        # kappa=0 is valid (uniform circular)
        d = pylibstats.VonMises(0.0, 0.0)
        assert d.kappa == pytest.approx(0.0)

    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError):
            pylibstats.VonMises(0.0, -1.0)

    def test_inf_mu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.VonMises(math.inf, 1.0)

    def test_mu_wrapping(self):
        # mu is stored wrapped to (-pi, pi]
        d = pylibstats.VonMises(mu=4.0, kappa=1.0)
        assert -math.pi < d.mu <= math.pi


class TestVonMisesScalar:
    def test_pdf_mode_is_maximum(self, von_mises):
        # PDF is maximised at mu=0 for VM(0, 1)
        assert von_mises.pdf(0.0) > von_mises.pdf(math.pi / 2.0)
        assert von_mises.pdf(0.0) > von_mises.pdf(-math.pi / 2.0)

    def test_pdf_kappa_zero_is_uniform(self):
        d = pylibstats.VonMises(0.0, 0.0)
        inv_2pi = 1.0 / (2.0 * math.pi)
        assert d.pdf(0.0) == pytest.approx(inv_2pi, rel=1e-8)
        assert d.pdf(1.5) == pytest.approx(inv_2pi, rel=1e-8)

    def test_log_pdf_consistency(self, von_mises):
        x = 1.0
        assert von_mises.log_pdf(x) == pytest.approx(math.log(von_mises.pdf(x)), rel=1e-10)

    def test_cdf_symmetry(self, von_mises):
        # CDF(mu) ≈ 0.5 by symmetry
        assert von_mises.cdf(0.0) == pytest.approx(0.5, abs=0.01)


class TestVonMisesBatch:
    def test_batch_shape(self, von_mises):
        x = np.linspace(-math.pi, math.pi, 200)
        assert von_mises.pdf(x).shape == (200,)

    def test_batch_matches_scalar(self, von_mises):
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(von_mises.pdf(x),
                                   [von_mises.pdf(v) for v in x], rtol=1e-12)


class TestVonMisesProperties:
    def test_mean(self, von_mises):
        assert von_mises.mean == pytest.approx(0.0, abs=1e-10)

    def test_is_not_discrete(self, von_mises):
        assert not von_mises.is_discrete

    def test_support(self, von_mises):
        lower, upper = von_mises.support
        assert lower == pytest.approx(-math.pi, rel=1e-10)
        assert upper == pytest.approx(math.pi, rel=1e-10)


class TestVonMisesFitAndSample:
    def test_sample_in_range(self, von_mises):
        s = von_mises.sample(n=500, seed=6)
        assert np.all(s > -math.pi) and np.all(s <= math.pi)

    def test_fit_recovers_params(self):
        d = pylibstats.VonMises(1.0, 3.0)
        data = d.sample(n=10_000, seed=11)
        fitted = pylibstats.VonMises()
        fitted.fit(data)
        assert fitted.mu == pytest.approx(1.0, abs=0.1)
        assert fitted.kappa == pytest.approx(3.0, abs=0.5)
