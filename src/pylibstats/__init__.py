"""pylibstats — Python bindings for libstats.

SIMD-accelerated statistical distributions with NumPy integration.
"""

import math

from pylibstats import _core


# ---------------------------------------------------------------------------
# Parameter validation helpers
# ---------------------------------------------------------------------------
# Validation is done in Python to avoid C++ exception-handling segfaults
# caused by libc++ ABI incompatibilities on macOS (Homebrew LLVM vs Apple
# clang).  See libstats core/error_handling.h for background.
# ---------------------------------------------------------------------------

def _require_finite(value, name):
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _require_positive_finite(value, name):
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


# ---------------------------------------------------------------------------
# Validated wrapper classes
# ---------------------------------------------------------------------------

class Gaussian(_core.Gaussian):
    """Gaussian (normal) distribution N(mu, sigma)."""

    __slots__ = ()

    def __init__(self, mu=0.0, sigma=1.0):
        _require_finite(mu, "Mean")
        _require_positive_finite(sigma, "Standard deviation")
        super().__init__(mu=mu, sigma=sigma)


class Exponential(_core.Exponential):
    """Exponential distribution Exp(lambda)."""

    __slots__ = ()

    def __init__(self, lam=1.0):
        _require_positive_finite(lam, "Lambda (rate parameter)")
        super().__init__(lam=lam)


class Uniform(_core.Uniform):
    """Continuous uniform distribution U(a, b)."""

    __slots__ = ()

    def __init__(self, a=0.0, b=1.0):
        _require_finite(a, "Lower bound (a)")
        _require_finite(b, "Upper bound (b)")
        if a >= b:
            raise ValueError(
                "Upper bound (b) must be strictly greater than lower bound (a)"
            )
        super().__init__(a=a, b=b)


class Poisson(_core.Poisson):
    """Poisson distribution Pois(lambda)."""

    __slots__ = ()

    def __init__(self, lam=1.0):
        _require_positive_finite(lam, "Lambda (rate parameter)")
        super().__init__(lam=lam)


class DiscreteUniform(_core.DiscreteUniform):
    """Discrete uniform distribution over integers {a, a+1, ..., b}."""

    __slots__ = ()

    def __init__(self, a=0, b=1):
        if a > b:
            raise ValueError(
                "Upper bound (b) must be greater than or equal to lower bound (a)"
            )
        super().__init__(a=a, b=b)


class Gamma(_core.Gamma):
    """Gamma distribution Gamma(alpha, beta)."""

    __slots__ = ()

    def __init__(self, alpha=1.0, beta=1.0):
        _require_positive_finite(alpha, "Alpha (shape parameter)")
        _require_positive_finite(beta, "Beta (rate parameter)")
        super().__init__(alpha=alpha, beta=beta)


class Beta(_core.Beta):
    """Beta distribution Beta(alpha, beta)."""

    __slots__ = ()

    def __init__(self, alpha=1.0, beta=1.0):
        _require_positive_finite(alpha, "Alpha (shape parameter)")
        _require_positive_finite(beta, "Beta (shape parameter)")
        super().__init__(alpha=alpha, beta=beta)


class ChiSquared(_core.ChiSquared):
    """Chi-squared distribution \u03c7\u00b2(k)."""

    __slots__ = ()

    def __init__(self, k=1.0):
        _require_positive_finite(k, "Degrees of freedom (k)")
        super().__init__(k=k)


class StudentT(_core.StudentT):
    """Student's t distribution t(nu)."""

    __slots__ = ()

    def __init__(self, nu=1.0):
        _require_positive_finite(nu, "Degrees of freedom (nu)")
        super().__init__(nu=nu)


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
