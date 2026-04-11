"""Type stubs for pylibstats package."""

from pylibstats._core import (
    Beta as Beta,
    ChiSquared as ChiSquared,
    DiscreteUniform as DiscreteUniform,
    Exponential as Exponential,
    Gamma as Gamma,
    Gaussian as Gaussian,
    Poisson as Poisson,
    StudentT as StudentT,
    Uniform as Uniform,
)

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
