"""Tests for pylibstats.Uniform (UniformDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestUniformConstruction:
    def test_default_params(self):
        dist = pylibstats.Uniform()
        assert dist.a == pytest.approx(0.0)
        assert dist.b == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.Uniform(a=-2.0, b=3.0)
        assert dist.a == pytest.approx(-2.0)
        assert dist.b == pytest.approx(3.0)

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Uniform(5.0, 2.0)  # a >= b


class TestUniformScalar:
    def test_pdf_in_support(self, uniform):
        assert uniform.pdf(0.5) == pytest.approx(1.0, rel=1e-12)

    def test_pdf_outside_support(self, uniform):
        assert uniform.pdf(-0.1) == pytest.approx(0.0, abs=1e-12)
        assert uniform.pdf(1.1) == pytest.approx(0.0, abs=1e-12)

    def test_cdf_linear(self, uniform):
        assert uniform.cdf(0.0) == pytest.approx(0.0, abs=1e-12)
        assert uniform.cdf(0.5) == pytest.approx(0.5, rel=1e-12)
        assert uniform.cdf(1.0) == pytest.approx(1.0, rel=1e-12)

    def test_ppf(self, uniform):
        assert uniform.ppf(0.25) == pytest.approx(0.25, rel=1e-10)
        assert uniform.ppf(0.75) == pytest.approx(0.75, rel=1e-10)


class TestUniformBatch:
    def test_batch_cdf_matches_scalar(self, uniform):
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        batch = uniform.cdf(x)
        scalar = np.array([uniform.cdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestUniformProperties:
    def test_mean(self, uniform):
        assert uniform.mean == pytest.approx(0.5, rel=1e-12)

    def test_variance(self, uniform):
        # Var = (b-a)^2 / 12 = 1/12
        assert uniform.variance == pytest.approx(1.0 / 12.0, rel=1e-10)

    def test_skewness(self, uniform):
        assert uniform.skewness == pytest.approx(0.0, abs=1e-12)

    def test_support(self, uniform):
        lower, upper = uniform.support
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(1.0)

    def test_is_not_discrete(self, uniform):
        assert not uniform.is_discrete


class TestUniformFitAndSample:
    def test_sample_in_bounds(self, uniform):
        samples = uniform.sample(n=1000, seed=42)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sample_shape(self, uniform):
        assert uniform.sample(n=100, seed=1).shape == (100,)
