"""Shared fixtures for pylibstats tests."""

import numpy as np
import pytest

import pylibstats


@pytest.fixture
def gaussian():
    """Standard normal distribution N(0, 1)."""
    return pylibstats.Gaussian(0.0, 1.0)


@pytest.fixture
def exponential():
    """Standard exponential distribution Exp(1)."""
    return pylibstats.Exponential(1.0)


@pytest.fixture
def uniform():
    """Unit uniform distribution U(0, 1)."""
    return pylibstats.Uniform(0.0, 1.0)


@pytest.fixture
def poisson():
    """Poisson distribution Pois(3)."""
    return pylibstats.Poisson(3.0)


@pytest.fixture
def discrete_uniform():
    """Fair six-sided die {1, 2, 3, 4, 5, 6}."""
    return pylibstats.DiscreteUniform(1, 6)


@pytest.fixture
def gamma_dist():
    """Gamma(2, 1) distribution."""
    return pylibstats.Gamma(2.0, 1.0)


@pytest.fixture
def beta_dist():
    """Beta(2, 5) distribution."""
    return pylibstats.Beta(2.0, 5.0)


@pytest.fixture
def chi_squared():
    """Chi-squared distribution with 5 degrees of freedom."""
    return pylibstats.ChiSquared(5.0)


@pytest.fixture
def student_t():
    """Student's t distribution with 10 degrees of freedom."""
    return pylibstats.StudentT(10.0)


@pytest.fixture
def lognormal():
    """Log-normal distribution LogN(0, 1)."""
    return pylibstats.LogNormal(0.0, 1.0)


@pytest.fixture
def pareto():
    """Pareto distribution Pareto(scale=1, alpha=2)."""
    return pylibstats.Pareto(1.0, 2.0)


@pytest.fixture
def weibull():
    """Weibull distribution W(shape=2, scale=1)."""
    return pylibstats.Weibull(2.0, 1.0)


@pytest.fixture
def rayleigh():
    """Rayleigh distribution R(sigma=1)."""
    return pylibstats.Rayleigh(1.0)


@pytest.fixture
def von_mises():
    """Von Mises distribution VM(mu=0, kappa=1)."""
    return pylibstats.VonMises(0.0, 1.0)


@pytest.fixture
def binomial():
    """Binomial distribution B(n=10, p=0.5)."""
    return pylibstats.Binomial(10, 0.5)


@pytest.fixture
def negative_binomial():
    """Negative Binomial distribution NB(r=2, p=0.5)."""
    return pylibstats.NegativeBinomial(2.0, 0.5)


@pytest.fixture
def rng():
    """Deterministic NumPy random generator for reproducible test data."""
    return np.random.default_rng(seed=42)
