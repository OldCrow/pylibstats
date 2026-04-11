"""Shared fixtures for pylibstats tests."""

import numpy as np
import pytest

import pylibstats


@pytest.fixture
def gaussian():
    """Standard normal distribution."""
    return pylibstats.Gaussian(0.0, 1.0)


@pytest.fixture
def rng():
    """Deterministic NumPy random generator for reproducible test data."""
    return np.random.default_rng(seed=42)
