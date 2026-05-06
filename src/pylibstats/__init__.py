"""pylibstats — Python bindings for libstats.

SIMD-accelerated statistical distributions with NumPy integration.
"""

import math

import numpy as np

from . import _core


# ---------------------------------------------------------------------------
# Parameter validation helpers
# ---------------------------------------------------------------------------
# Validation is done in Python to avoid C++ exception-handling segfaults
# caused by libc++ ABI incompatibilities on macOS (Homebrew LLVM vs Apple
# clang).  See libstats core/error_handling.h for background.
# ---------------------------------------------------------------------------

def _require_finite(value, name):
    """Raise ValueError if *value* is NaN or infinite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _require_positive_finite(value, name):
    """Raise ValueError if *value* is not a positive finite number."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def _validate_fit_data(data):
    """Raise ValueError if *data* is empty or contains non-finite values."""
    if len(data) == 0:
        raise ValueError("fit() requires at least one data point")
    if not np.all(np.isfinite(data)):
        raise ValueError("fit() data must contain only finite values")


def _validated_prop(parent_prop, validator, param_name, doc=None):
    """Create a property that validates before delegating to a nanobind property setter."""

    def getter(self):
        # nanobind property descriptors require explicit __get__/__set__ calls
        # (not getattr/setattr) to dispatch correctly from a Python subclass.
        return parent_prop.__get__(self)

    def setter(self, value):
        validator(value, param_name)
        parent_prop.__set__(self, value)

    return property(getter, setter, doc=doc)


# ---------------------------------------------------------------------------
# Validated wrapper classes
# ---------------------------------------------------------------------------

class Gaussian(_core.Gaussian):
    """Gaussian (normal) distribution N(mu, sigma).

    Parameters
    ----------
    mu : float
        Mean parameter μ (any finite value, default 0).
    sigma : float
        Standard deviation parameter σ (must be positive, default 1).

    Raises
    ------
    ValueError
        If *mu* is not finite or *sigma* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, mu=0.0, sigma=1.0):
        _require_finite(mu, "Mean")
        _require_positive_finite(sigma, "Standard deviation")
        super().__init__(mu=mu, sigma=sigma)

    mu = _validated_prop(_core.Gaussian.mu, _require_finite, "Mean",
                         "Mean parameter μ.")
    sigma = _validated_prop(_core.Gaussian.sigma, _require_positive_finite,
                            "Standard deviation", "Standard deviation parameter σ.")

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class Exponential(_core.Exponential):
    """Exponential distribution Exp(lambda).

    Parameters
    ----------
    lam : float
        Rate parameter λ (must be positive, default 1).

    Raises
    ------
    ValueError
        If *lam* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, lam=1.0):
        _require_positive_finite(lam, "Lambda (rate parameter)")
        super().__init__(lam=lam)

    lam = _validated_prop(_core.Exponential.lam, _require_positive_finite,
                          "Lambda (rate parameter)", "Rate parameter λ.")

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class Uniform(_core.Uniform):
    """Continuous uniform distribution U(a, b).

    Parameters
    ----------
    a : float
        Lower bound (must be finite, default 0).
    b : float
        Upper bound (must be finite and strictly greater than *a*, default 1).

    Raises
    ------
    ValueError
        If *a* or *b* is not finite, or *a* >= *b*.
    """

    __slots__ = ()

    def __init__(self, a=0.0, b=1.0):
        _require_finite(a, "Lower bound (a)")
        _require_finite(b, "Upper bound (b)")
        if a >= b:
            raise ValueError(
                "Upper bound (b) must be strictly greater than lower bound (a)"
            )
        super().__init__(a=a, b=b)

    @property
    def a(self):
        """Lower bound a."""
        return _core.Uniform.a.__get__(self)

    @a.setter
    def a(self, value):
        _require_finite(value, "Lower bound (a)")
        if value >= self.b:
            raise ValueError(
                "Lower bound (a) must be strictly less than upper bound (b)"
            )
        _core.Uniform.a.__set__(self, value)

    @property
    def b(self):
        """Upper bound b."""
        return _core.Uniform.b.__get__(self)

    @b.setter
    def b(self, value):
        _require_finite(value, "Upper bound (b)")
        if value <= self.a:
            raise ValueError(
                "Upper bound (b) must be strictly greater than lower bound (a)"
            )
        _core.Uniform.b.__set__(self, value)

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class Poisson(_core.Poisson):
    """Poisson distribution Pois(lambda).

    Parameters
    ----------
    lam : float
        Rate parameter λ (must be positive, default 1).

    Raises
    ------
    ValueError
        If *lam* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, lam=1.0):
        _require_positive_finite(lam, "Lambda (rate parameter)")
        super().__init__(lam=lam)

    lam = _validated_prop(_core.Poisson.lam, _require_positive_finite,
                          "Lambda (rate parameter)", "Rate parameter λ.")

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class DiscreteUniform(_core.DiscreteUniform):
    """Discrete uniform distribution over integers {a, a+1, ..., b}.

    Parameters
    ----------
    a : int
        Lower bound (default 0).
    b : int
        Upper bound (must be >= *a*, default 1).

    Raises
    ------
    ValueError
        If *a* > *b*.
    """

    __slots__ = ()

    def __init__(self, a=0, b=1):
        if a > b:
            raise ValueError(
                "Upper bound (b) must be greater than or equal to lower bound (a)"
            )
        super().__init__(a=a, b=b)

    @property
    def a(self):
        """Lower bound a (integer)."""
        return _core.DiscreteUniform.a.__get__(self)

    @a.setter
    def a(self, value):
        if value > self.b:
            raise ValueError(
                "Lower bound (a) must be less than or equal to upper bound (b)"
            )
        _core.DiscreteUniform.a.__set__(self, value)

    @property
    def b(self):
        """Upper bound b (integer)."""
        return _core.DiscreteUniform.b.__get__(self)

    @b.setter
    def b(self, value):
        if value < self.a:
            raise ValueError(
                "Upper bound (b) must be greater than or equal to lower bound (a)"
            )
        _core.DiscreteUniform.b.__set__(self, value)

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class Gamma(_core.Gamma):
    """Gamma distribution Gamma(alpha, beta).

    Parameters
    ----------
    alpha : float
        Shape parameter α (must be positive, default 1).
    beta : float
        Rate parameter β (must be positive, default 1).

    Raises
    ------
    ValueError
        If *alpha* or *beta* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, alpha=1.0, beta=1.0):
        _require_positive_finite(alpha, "Alpha (shape parameter)")
        _require_positive_finite(beta, "Beta (rate parameter)")
        super().__init__(alpha=alpha, beta=beta)

    alpha = _validated_prop(_core.Gamma.alpha, _require_positive_finite,
                            "Alpha (shape parameter)", "Shape parameter α.")
    beta = _validated_prop(_core.Gamma.beta, _require_positive_finite,
                           "Beta (rate parameter)", "Rate parameter β.")

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class Beta(_core.Beta):
    """Beta distribution Beta(alpha, beta).

    Parameters
    ----------
    alpha : float
        Shape parameter α (must be positive, default 1).
    beta : float
        Shape parameter β (must be positive, default 1).

    Raises
    ------
    ValueError
        If *alpha* or *beta* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, alpha=1.0, beta=1.0):
        _require_positive_finite(alpha, "Alpha (shape parameter)")
        _require_positive_finite(beta, "Beta (shape parameter)")
        super().__init__(alpha=alpha, beta=beta)

    alpha = _validated_prop(_core.Beta.alpha, _require_positive_finite,
                            "Alpha (shape parameter)", "Shape parameter α.")
    beta = _validated_prop(_core.Beta.beta, _require_positive_finite,
                           "Beta (shape parameter)", "Shape parameter β.")

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class ChiSquared(_core.ChiSquared):
    """Chi-squared distribution χ²(k).

    Parameters
    ----------
    k : float
        Degrees of freedom (must be positive, default 1).

    Raises
    ------
    ValueError
        If *k* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, k=1.0):
        _require_positive_finite(k, "Degrees of freedom (k)")
        super().__init__(k=k)

    k = _validated_prop(_core.ChiSquared.k, _require_positive_finite,
                        "Degrees of freedom (k)", "Degrees of freedom k.")

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


class StudentT(_core.StudentT):
    """Student's t distribution t(nu).

    Parameters
    ----------
    nu : float
        Degrees of freedom ν (must be positive, default 1).

    Raises
    ------
    ValueError
        If *nu* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, nu=1.0):
        _require_positive_finite(nu, "Degrees of freedom (nu)")
        super().__init__(nu=nu)

    nu = _validated_prop(_core.StudentT.nu, _require_positive_finite,
                         "Degrees of freedom (nu)", "Degrees of freedom ν.")

    def fit(self, data):
        _validate_fit_data(data)
        super().fit(data)


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
