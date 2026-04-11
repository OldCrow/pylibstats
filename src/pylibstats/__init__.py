"""pylibstats — Python bindings for libstats.

SIMD-accelerated statistical distributions with NumPy integration.
"""

from pylibstats._core import (
    Beta,
    ChiSquared,
    DiscreteUniform,
    Exponential,
    Gamma,
    Gaussian,
    Poisson,
    StudentT,
    Uniform,
)

# SciPy-familiar alias
Normal = Gaussian

__all__ = [
    "Beta",
    "ChiSquared",
    "DiscreteUniform",
    "Exponential",
    "Gamma",
    "Gaussian",
    "Normal",
    "Poisson",
    "StudentT",
    "Uniform",
]
