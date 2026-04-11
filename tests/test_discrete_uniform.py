"""Tests for pylibstats.DiscreteUniform (DiscreteDistribution bindings)."""

import numpy as np
import pytest

import pylibstats


class TestDiscreteUniformConstruction:
    def test_default_params(self):
        dist = pylibstats.DiscreteUniform()
        assert dist.a == 0
        assert dist.b == 1

    def test_dice(self, discrete_uniform):
        assert discrete_uniform.a == 1
        assert discrete_uniform.b == 6

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            pylibstats.DiscreteUniform(6, 1)  # a > b


class TestDiscreteUniformScalar:
    def test_pmf_equal(self, discrete_uniform):
        # Each outcome has probability 1/6
        for k in range(1, 7):
            assert discrete_uniform.pdf(float(k)) == pytest.approx(1.0 / 6.0, rel=1e-10)

    def test_cdf_at_bounds(self, discrete_uniform):
        assert discrete_uniform.cdf(0.0) == pytest.approx(0.0, abs=1e-10)
        assert discrete_uniform.cdf(6.0) == pytest.approx(1.0, rel=1e-10)

    def test_cdf_at_three(self, discrete_uniform):
        assert discrete_uniform.cdf(3.0) == pytest.approx(3.0 / 6.0, rel=1e-10)


class TestDiscreteUniformBatch:
    def test_batch_pdf_matches_scalar(self, discrete_uniform):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        batch = discrete_uniform.pdf(x)
        scalar = np.array([discrete_uniform.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestDiscreteUniformProperties:
    def test_mean(self, discrete_uniform):
        assert discrete_uniform.mean == pytest.approx(3.5, rel=1e-10)

    def test_variance(self, discrete_uniform):
        # Var = ((b-a)(b-a+2))/12 = (5*7)/12 = 35/12
        assert discrete_uniform.variance == pytest.approx(35.0 / 12.0, rel=1e-10)

    def test_is_discrete(self, discrete_uniform):
        assert discrete_uniform.is_discrete

    def test_support(self, discrete_uniform):
        lower, upper = discrete_uniform.support
        assert lower == pytest.approx(1.0)
        assert upper == pytest.approx(6.0)


class TestDiscreteUniformSample:
    def test_sample_in_range(self, discrete_uniform):
        samples = discrete_uniform.sample(n=1000, seed=42)
        assert np.all(samples >= 1)
        assert np.all(samples <= 6)

    def test_sample_shape(self, discrete_uniform):
        assert discrete_uniform.sample(n=200, seed=1).shape == (200,)
