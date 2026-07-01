"""Tests for Python-level parameter validation (setters and fit).

Covers property setter rejection of invalid values and fit() input
guards for all nine distribution classes.
"""

import math

import numpy as np
import pytest

import pylibstats


# ---------------------------------------------------------------------------
# Setter validation
# ---------------------------------------------------------------------------

class TestGaussianSetterValidation:
    def test_sigma_rejects_zero(self):
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError, match="[Ss]tandard deviation"):
            g.sigma = 0.0

    def test_sigma_rejects_negative(self):
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError):
            g.sigma = -1.0

    def test_mu_rejects_inf(self):
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError, match="[Mm]ean"):
            g.mu = math.inf

    def test_mu_rejects_nan(self):
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError):
            g.mu = float("nan")

    def test_valid_setters_apply(self):
        g = pylibstats.Gaussian()
        g.mu = 5.0
        g.sigma = 2.0
        assert g.mu == pytest.approx(5.0)
        assert g.sigma == pytest.approx(2.0)


class TestExponentialSetterValidation:
    def test_lam_rejects_zero(self):
        e = pylibstats.Exponential()
        with pytest.raises(ValueError):
            e.lam = 0.0

    def test_lam_rejects_negative(self):
        e = pylibstats.Exponential()
        with pytest.raises(ValueError):
            e.lam = -1.0

    def test_valid_setter_applies(self):
        e = pylibstats.Exponential()
        e.lam = 5.0
        assert e.lam == pytest.approx(5.0)


class TestUniformSetterValidation:
    def test_a_rejects_ge_b(self):
        u = pylibstats.Uniform(0.0, 1.0)
        with pytest.raises(ValueError):
            u.a = 2.0

    def test_b_rejects_le_a(self):
        u = pylibstats.Uniform(0.0, 1.0)
        with pytest.raises(ValueError):
            u.b = -1.0

    def test_a_rejects_inf(self):
        u = pylibstats.Uniform(0.0, 1.0)
        with pytest.raises(ValueError):
            u.a = math.inf

    def test_valid_setters_apply(self):
        u = pylibstats.Uniform(0.0, 1.0)
        u.a = -5.0
        u.b = 10.0
        assert u.a == pytest.approx(-5.0)
        assert u.b == pytest.approx(10.0)


class TestPoissonSetterValidation:
    def test_lam_rejects_zero(self):
        p = pylibstats.Poisson()
        with pytest.raises(ValueError):
            p.lam = 0.0


class TestDiscreteUniformSetterValidation:
    def test_a_rejects_gt_b(self):
        d = pylibstats.DiscreteUniform(1, 6)
        with pytest.raises(ValueError):
            d.a = 7

    def test_b_rejects_lt_a(self):
        d = pylibstats.DiscreteUniform(1, 6)
        with pytest.raises(ValueError):
            d.b = 0

    def test_equal_bounds_rejected(self):
        # libstats v1.2.0: equal bounds (a == b) are now invalid (a >= b fails).
        d = pylibstats.DiscreteUniform(1, 6)
        d.a = 3  # valid: a=3, b=6
        with pytest.raises(ValueError):
            d.b = 3  # a == b is now invalid


class TestGammaSetterValidation:
    def test_alpha_rejects_zero(self):
        g = pylibstats.Gamma()
        with pytest.raises(ValueError):
            g.alpha = 0.0

    def test_beta_rejects_negative(self):
        g = pylibstats.Gamma()
        with pytest.raises(ValueError):
            g.beta = -1.0


class TestBetaSetterValidation:
    def test_alpha_rejects_zero(self):
        b = pylibstats.Beta()
        with pytest.raises(ValueError):
            b.alpha = 0.0

    def test_beta_rejects_nan(self):
        b = pylibstats.Beta()
        with pytest.raises(ValueError):
            b.beta = float("nan")


class TestChiSquaredSetterValidation:
    def test_k_rejects_zero(self):
        c = pylibstats.ChiSquared()
        with pytest.raises(ValueError):
            c.k = 0.0


class TestStudentTSetterValidation:
    def test_nu_rejects_negative(self):
        t = pylibstats.StudentT()
        with pytest.raises(ValueError):
            t.nu = -5.0


class TestGeometricSetterValidation:
    def test_p_rejects_zero(self):
        g = pylibstats.Geometric()
        with pytest.raises(ValueError):
            g.p = 0.0

    def test_p_rejects_negative(self):
        g = pylibstats.Geometric()
        with pytest.raises(ValueError):
            g.p = -0.1

    def test_p_rejects_greater_than_one(self):
        g = pylibstats.Geometric()
        with pytest.raises(ValueError):
            g.p = 1.5

    def test_valid_setter_applies(self):
        g = pylibstats.Geometric()
        g.p = 0.7
        assert g.p == pytest.approx(0.7)


class TestLaplaceSetterValidation:
    def test_b_rejects_zero(self):
        lap = pylibstats.Laplace()
        with pytest.raises(ValueError):
            lap.b = 0.0

    def test_b_rejects_negative(self):
        lap = pylibstats.Laplace()
        with pytest.raises(ValueError):
            lap.b = -1.0

    def test_mu_rejects_inf(self):
        lap = pylibstats.Laplace()
        with pytest.raises(ValueError):
            lap.mu = math.inf

    def test_valid_setters_apply(self):
        lap = pylibstats.Laplace()
        lap.mu = 3.0
        lap.b = 2.0
        assert lap.mu == pytest.approx(3.0)
        assert lap.b == pytest.approx(2.0)


class TestCauchySetterValidation:
    def test_gamma_rejects_zero(self):
        c = pylibstats.Cauchy()
        with pytest.raises(ValueError):
            c.gamma = 0.0

    def test_gamma_rejects_negative(self):
        c = pylibstats.Cauchy()
        with pytest.raises(ValueError):
            c.gamma = -0.5

    def test_x0_rejects_nan(self):
        c = pylibstats.Cauchy()
        with pytest.raises(ValueError):
            c.x0 = float("nan")

    def test_valid_setters_apply(self):
        c = pylibstats.Cauchy()
        c.x0 = 1.0
        c.gamma = 0.5
        assert c.x0 == pytest.approx(1.0)
        assert c.gamma == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fit() input validation
# ---------------------------------------------------------------------------

# Use Gaussian as representative; the validation path is shared.

class TestFitValidation:
    def test_empty_array_raises(self):
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError, match="at least one"):
            g.fit(np.array([]))

    def test_nan_data_raises(self):
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError, match="finite"):
            g.fit(np.array([1.0, float("nan"), 3.0]))

    def test_inf_data_raises(self):
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError, match="finite"):
            g.fit(np.array([1.0, math.inf, 3.0]))

    def test_valid_fit_succeeds(self):
        g = pylibstats.Gaussian()
        g.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert g.mu == pytest.approx(3.0)

    def test_generator_raises_value_error_not_type_error(self):
        # S-4: a generator has no len(); _validate_fit_data must convert via
        # np.asarray first, otherwise TypeError leaks instead of ValueError.
        g = pylibstats.Gaussian()
        with pytest.raises(ValueError, match="at least one"):
            g.fit(x for x in [])  # empty generator

    def test_generator_with_data_accepted(self):
        # S-4: non-empty generator should work (np.asarray materialises it).
        g = pylibstats.Gaussian()
        g.fit(x for x in [1.0, 2.0, 3.0, 4.0, 5.0])
        assert g.mu == pytest.approx(3.0)

    def test_fit_validation_all_distributions(self):
        """Verify every distribution rejects empty data."""
        dists = [
            pylibstats.Gaussian(),
            pylibstats.Exponential(),
            pylibstats.Uniform(),
            pylibstats.Poisson(),
            pylibstats.DiscreteUniform(),
            pylibstats.Gamma(),
            pylibstats.Beta(),
            pylibstats.ChiSquared(),
            pylibstats.StudentT(),
            pylibstats.Geometric(),
            pylibstats.Laplace(),
            pylibstats.Cauchy(),
        ]
        empty = np.array([])
        for dist in dists:
            with pytest.raises(ValueError, match="at least one"):
                dist.fit(empty)
