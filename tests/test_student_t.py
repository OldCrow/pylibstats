"""Tests for pylibstats.StudentT (StudentTDistribution bindings)."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import pylibstats


class TestStudentTConstruction:
    def test_default_params(self):
        dist = pylibstats.StudentT()
        assert dist.nu == pytest.approx(1.0)

    def test_custom_params(self):
        dist = pylibstats.StudentT(nu=30.0)
        assert dist.nu == pytest.approx(30.0)

    def test_invalid_nu_raises(self):
        with pytest.raises(ValueError):
            pylibstats.StudentT(nu=0.0)


class TestStudentTScalar:
    def test_pdf_at_zero(self, student_t):
        expected = sp.t.pdf(0.0, df=10)
        assert student_t.pdf(0.0) == pytest.approx(expected, rel=1e-8)

    def test_cdf_at_zero(self, student_t):
        # Symmetric: CDF(0) = 0.5
        assert student_t.cdf(0.0) == pytest.approx(0.5, rel=1e-10)

    def test_pdf_symmetric(self, student_t):
        assert student_t.pdf(2.0) == pytest.approx(student_t.pdf(-2.0), rel=1e-12)

    def test_ppf_matches_scipy(self, student_t):
        p = 0.975
        expected = sp.t.ppf(p, df=10)
        assert student_t.ppf(p) == pytest.approx(expected, rel=1e-4)


class TestStudentTBatch:
    def test_batch_pdf_matches_scalar(self, student_t):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        batch = student_t.pdf(x)
        scalar = np.array([student_t.pdf(float(v)) for v in x])
        np.testing.assert_allclose(batch, scalar, rtol=1e-12)


class TestStudentTProperties:
    def test_mean(self, student_t):
        # Mean = 0 for nu > 1
        assert student_t.mean == pytest.approx(0.0, abs=1e-12)

    def test_variance(self, student_t):
        # Variance = nu/(nu-2) = 10/8 = 1.25
        assert student_t.variance == pytest.approx(10.0 / 8.0, rel=1e-8)

    def test_skewness(self, student_t):
        # Skewness = 0 for nu > 3
        assert student_t.skewness == pytest.approx(0.0, abs=1e-12)

    def test_kurtosis(self, student_t):
        # Excess kurtosis = 6/(nu-4) = 6/6 = 1.0
        assert student_t.kurtosis == pytest.approx(1.0, rel=1e-8)

    def test_support(self, student_t):
        lower, upper = student_t.support
        assert lower == -math.inf
        assert upper == math.inf

    def test_is_not_discrete(self, student_t):
        assert not student_t.is_discrete


class TestStudentTFitAndSample:
    def test_sample_shape(self, student_t):
        assert student_t.sample(n=500, seed=42).shape == (500,)

    def test_heavier_tails_than_gaussian(self):
        """Student's t(10) should have more extreme values than N(0,1)."""
        gauss = pylibstats.Gaussian(0.0, 1.0)
        tdist = pylibstats.StudentT(nu=10.0)
        # Compare P(X > 3): should be larger for t
        gauss_tail = 1.0 - gauss.cdf(3.0)
        t_tail = 1.0 - tdist.cdf(3.0)
        assert t_tail > gauss_tail
