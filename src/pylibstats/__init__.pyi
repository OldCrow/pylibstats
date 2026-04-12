"""Type stubs for pylibstats package.

All distribution classes validate parameters in Python before delegating
to the native C++ extension.  Invalid parameters raise ``ValueError``.
"""

from typing import overload

import numpy as np
from numpy.typing import NDArray

from pylibstats import _core

# ---------------------------------------------------------------------------
# Common method signatures inherited by every distribution
# ---------------------------------------------------------------------------
# Each class below also exposes:
#   pdf, log_pdf, cdf   — scalar (float) and batch (NDArray) overloads
#   ppf                 — scalar only
#   fit                 — NDArray input
#   sample              — returns NDArray
#   mean, variance, std, skewness, kurtosis  — read-only float properties
#   name, support, is_discrete, num_parameters — read-only metadata
#   __repr__
# ---------------------------------------------------------------------------

class Gaussian(_core.Gaussian):
    """Gaussian (normal) distribution N(mu, sigma)."""
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None: ...
    @property
    def mu(self) -> float: ...
    @mu.setter
    def mu(self, value: float) -> None: ...
    @property
    def sigma(self) -> float: ...
    @sigma.setter
    def sigma(self, value: float) -> None: ...

class Exponential(_core.Exponential):
    """Exponential distribution Exp(lambda)."""
    def __init__(self, lam: float = 1.0) -> None: ...
    @property
    def lam(self) -> float: ...
    @lam.setter
    def lam(self, value: float) -> None: ...

class Uniform(_core.Uniform):
    """Continuous uniform distribution U(a, b)."""
    def __init__(self, a: float = 0.0, b: float = 1.0) -> None: ...
    @property
    def a(self) -> float: ...
    @a.setter
    def a(self, value: float) -> None: ...
    @property
    def b(self) -> float: ...
    @b.setter
    def b(self, value: float) -> None: ...

class Poisson(_core.Poisson):
    """Poisson distribution Pois(lambda)."""
    def __init__(self, lam: float = 1.0) -> None: ...
    @property
    def lam(self) -> float: ...
    @lam.setter
    def lam(self, value: float) -> None: ...

class DiscreteUniform(_core.DiscreteUniform):
    """Discrete uniform distribution over integers {a, a+1, ..., b}."""
    def __init__(self, a: int = 0, b: int = 1) -> None: ...
    @property
    def a(self) -> int: ...
    @a.setter
    def a(self, value: int) -> None: ...
    @property
    def b(self) -> int: ...
    @b.setter
    def b(self, value: int) -> None: ...

class Gamma(_core.Gamma):
    """Gamma distribution Gamma(alpha, beta)."""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None: ...
    @property
    def alpha(self) -> float: ...
    @alpha.setter
    def alpha(self, value: float) -> None: ...
    @property
    def beta(self) -> float: ...
    @beta.setter
    def beta(self, value: float) -> None: ...

class Beta(_core.Beta):
    """Beta distribution Beta(alpha, beta)."""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None: ...
    @property
    def alpha(self) -> float: ...
    @alpha.setter
    def alpha(self, value: float) -> None: ...
    @property
    def beta(self) -> float: ...
    @beta.setter
    def beta(self, value: float) -> None: ...

class ChiSquared(_core.ChiSquared):
    """Chi-squared distribution χ²(k)."""
    def __init__(self, k: float = 1.0) -> None: ...
    @property
    def k(self) -> float: ...
    @k.setter
    def k(self, value: float) -> None: ...

class StudentT(_core.StudentT):
    """Student's t distribution t(nu)."""
    def __init__(self, nu: float = 1.0) -> None: ...
    @property
    def nu(self) -> float: ...
    @nu.setter
    def nu(self, value: float) -> None: ...

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
