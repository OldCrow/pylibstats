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


def _coerce_batch_input(x):
    """Normalize input for pdf / log_pdf / cdf.

    Scalars (int, float, 0-d array, NumPy scalar) become Python float so that
    nanobind dispatches to the scalar C++ overload.  Array-like inputs (lists,
    int arrays, strided views) are converted to a C-contiguous float64 ndarray
    so they reach the batch C++ overload.  This matches the behaviour of
    ``fit()``, which also accepts any array-like via ``_validate_fit_data``.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return float(x)
        return np.ascontiguousarray(x, dtype=np.float64)
    if isinstance(x, (int, float, np.generic)):
        return float(x)
    # list, tuple, or other sequence
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return np.ascontiguousarray(arr)


def _validate_fit_data(data):
    """Convert *data* to float64, then raise ValueError if empty or non-finite.

    Accepts any array-like including generators; always returns a 1-D
    float64 ndarray so callers can pass it directly to the C++ binding.
    """
    # np.asarray cannot consume generators; materialise them to a list first.
    if not isinstance(data, (np.ndarray, list, tuple)):
        data = list(data)
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("fit() requires at least one data point")
    if not np.all(np.isfinite(arr)):
        raise ValueError("fit() data must contain only finite values")
    return arr


def _require_non_negative_finite(value, name):
    """Raise ValueError if *value* is not a non-negative finite number."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _require_probability(value, name):
    """Raise ValueError if *value* is not in [0, 1]."""
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_positive_probability(value, name):
    """Raise ValueError if *value* is not in (0, 1]."""
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in (0, 1]")


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
        super().fit(_validate_fit_data(data))


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
        super().fit(_validate_fit_data(data))


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
        super().fit(_validate_fit_data(data))


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
        super().fit(_validate_fit_data(data))


class DiscreteUniform(_core.DiscreteUniform):
    """Discrete uniform distribution over integers {a, a+1, ..., b}.

    Parameters
    ----------
    a : int
        Lower bound (default 0).
    b : int
        Upper bound (must be >= *a*; a == b is a valid degenerate distribution
        with a single outcome, default 1).

    Raises
    ------
    ValueError
        If *a* > *b*.
    """

    __slots__ = ()

    def __init__(self, a=0, b=1):
        if a > b:
            raise ValueError(
                "Upper bound (b) must be >= lower bound (a)"
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
                "Lower bound (a) must be <= upper bound (b)"
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
                "Upper bound (b) must be >= lower bound (a)"
            )
        _core.DiscreteUniform.b.__set__(self, value)

    def fit(self, data):
        super().fit(_validate_fit_data(data))


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
        super().fit(_validate_fit_data(data))


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
        super().fit(_validate_fit_data(data))


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
        super().fit(_validate_fit_data(data))


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
        super().fit(_validate_fit_data(data))


class LogNormal(_core.LogNormal):
    """Log-normal distribution LogN(mu, sigma).

    Parameters
    ----------
    mu : float
        Location parameter \u03bc (log-mean, any finite value, default 0).
    sigma : float
        Scale parameter \u03c3 (log-stddev, must be positive, default 1).

    Raises
    ------
    ValueError
        If *mu* is not finite or *sigma* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, mu=0.0, sigma=1.0):
        _require_finite(mu, "Location parameter (mu)")
        _require_positive_finite(sigma, "Scale parameter (sigma)")
        super().__init__(mu=mu, sigma=sigma)

    mu = _validated_prop(_core.LogNormal.mu, _require_finite,
                         "Location parameter (mu)", "Location parameter \u03bc (log-mean).")
    sigma = _validated_prop(_core.LogNormal.sigma, _require_positive_finite,
                            "Scale parameter (sigma)", "Scale parameter \u03c3 (log-stddev).")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class Pareto(_core.Pareto):
    """Pareto distribution Pareto(scale, alpha).

    Parameters
    ----------
    scale : float
        Minimum value x_m (must be positive, default 1).
    alpha : float
        Shape/tail parameter \u03b1 (must be positive, default 1).

    Raises
    ------
    ValueError
        If *scale* or *alpha* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, scale=1.0, alpha=1.0):
        _require_positive_finite(scale, "Scale parameter (scale)")
        _require_positive_finite(alpha, "Shape parameter (alpha)")
        super().__init__(scale=scale, alpha=alpha)

    scale = _validated_prop(_core.Pareto.scale, _require_positive_finite,
                            "Scale parameter (scale)", "Minimum value x_m.")
    alpha = _validated_prop(_core.Pareto.alpha, _require_positive_finite,
                            "Shape parameter (alpha)", "Shape parameter \u03b1 (tail index).")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class Weibull(_core.Weibull):
    """Weibull distribution W(shape, scale).

    Parameters
    ----------
    shape : float
        Shape parameter k (must be positive, default 1).
    scale : float
        Scale parameter \u03bb (must be positive, default 1).

    Raises
    ------
    ValueError
        If *shape* or *scale* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, shape=1.0, scale=1.0):
        _require_positive_finite(shape, "Shape parameter (shape)")
        _require_positive_finite(scale, "Scale parameter (scale)")
        super().__init__(shape=shape, scale=scale)

    shape = _validated_prop(_core.Weibull.shape, _require_positive_finite,
                            "Shape parameter (shape)", "Shape parameter k.")
    scale = _validated_prop(_core.Weibull.scale, _require_positive_finite,
                            "Scale parameter (scale)", "Scale parameter \u03bb.")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class Rayleigh(_core.Rayleigh):
    """Rayleigh distribution R(sigma).

    Parameters
    ----------
    sigma : float
        Scale parameter \u03c3 (must be positive, default 1).

    Raises
    ------
    ValueError
        If *sigma* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, sigma=1.0):
        _require_positive_finite(sigma, "Scale parameter (sigma)")
        super().__init__(sigma=sigma)

    sigma = _validated_prop(_core.Rayleigh.sigma, _require_positive_finite,
                            "Scale parameter (sigma)", "Scale parameter \u03c3.")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class VonMises(_core.VonMises):
    """Von Mises distribution VM(mu, kappa).

    Parameters
    ----------
    mu : float
        Mean direction \u03bc (any finite real; stored wrapped to (-\u03c0, \u03c0], default 0).
    kappa : float
        Concentration parameter \u03ba (\u2265 0; 0 = uniform circular, default 1).

    Raises
    ------
    ValueError
        If *mu* is not finite or *kappa* is negative or not finite.
    """

    __slots__ = ()

    def __init__(self, mu=0.0, kappa=1.0):
        _require_finite(mu, "Mean direction (mu)")
        _require_non_negative_finite(kappa, "Concentration (kappa)")
        super().__init__(mu=mu, kappa=kappa)

    mu = _validated_prop(_core.VonMises.mu, _require_finite,
                         "Mean direction (mu)", "Mean direction \u03bc (wrapped to (-\u03c0, \u03c0]).")
    kappa = _validated_prop(_core.VonMises.kappa, _require_non_negative_finite,
                            "Concentration (kappa)", "Concentration \u03ba (\u2265 0).")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class Binomial(_core.Binomial):
    """Binomial distribution B(n, p).

    Parameters
    ----------
    n : int
        Number of trials (must be a positive integer, default 10).
    p : float
        Success probability (must be in [0, 1], default 0.5).

    Raises
    ------
    ValueError
        If *n* is not a positive integer or *p* is not in [0, 1].
    """

    __slots__ = ()

    def __init__(self, n=10, p=0.5):
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n must be a positive integer")
        _require_probability(p, "Probability (p)")
        super().__init__(n=int(n), p=p)

    @property
    def n(self):
        """Number of trials n (positive integer)."""
        return _core.Binomial.n.__get__(self)

    @n.setter
    def n(self, value):
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("n must be a positive integer")
        _core.Binomial.n.__set__(self, int(value))

    p = _validated_prop(_core.Binomial.p, _require_probability,
                        "Probability (p)", "Success probability p (in [0, 1]).")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class NegativeBinomial(_core.NegativeBinomial):
    """Negative Binomial distribution NB(r, p).

    Parameters
    ----------
    r : float
        Number of successes (real-valued, must be positive, default 1).
    p : float
        Success probability (must be in (0, 1], default 0.5).

    Raises
    ------
    ValueError
        If *r* is not positive and finite, or *p* is not in (0, 1].
    """

    __slots__ = ()

    def __init__(self, r=1.0, p=0.5):
        _require_positive_finite(r, "Success count (r)")
        _require_positive_probability(p, "Probability (p)")
        super().__init__(r=r, p=p)

    r = _validated_prop(_core.NegativeBinomial.r, _require_positive_finite,
                        "Success count (r)", "Number of successes r (positive, real-valued).")
    p = _validated_prop(_core.NegativeBinomial.p, _require_positive_probability,
                        "Probability (p)", "Success probability p (in (0, 1]).")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class Geometric(_core.Geometric):
    """Geometric distribution Geo(p).

    Models the number of failures before the first success in a sequence of
    independent Bernoulli trials.

    Parameters
    ----------
    p : float
        Success probability (must be in (0, 1], default 0.5).

    Raises
    ------
    ValueError
        If *p* is not in (0, 1].
    """

    __slots__ = ()

    def __init__(self, p=0.5):
        _require_positive_probability(p, "Probability (p)")
        super().__init__(p=p)

    p = _validated_prop(_core.Geometric.p, _require_positive_probability,
                        "Probability (p)", "Success probability p (in (0, 1]).")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class Laplace(_core.Laplace):
    """Laplace (double-exponential) distribution Laplace(mu, b).

    Parameters
    ----------
    mu : float
        Location parameter \u03bc (any finite value, default 0).
    b : float
        Scale parameter b (must be positive, default 1).

    Raises
    ------
    ValueError
        If *mu* is not finite or *b* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, mu=0.0, b=1.0):
        _require_finite(mu, "Location parameter (mu)")
        _require_positive_finite(b, "Scale parameter (b)")
        super().__init__(mu=mu, b=b)

    mu = _validated_prop(_core.Laplace.mu, _require_finite,
                         "Location parameter (mu)", "Location parameter \u03bc.")
    b = _validated_prop(_core.Laplace.b, _require_positive_finite,
                        "Scale parameter (b)", "Scale parameter b.")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


class Cauchy(_core.Cauchy):
    """Cauchy distribution Cauchy(x0, gamma).

    Mean, variance, skewness, and kurtosis are undefined (NaN).

    Parameters
    ----------
    x0 : float
        Location parameter (any finite value, default 0).
    gamma : float
        Scale parameter \u03b3 (must be positive, default 1).

    Raises
    ------
    ValueError
        If *x0* is not finite or *gamma* is not positive and finite.
    """

    __slots__ = ()

    def __init__(self, x0=0.0, gamma=1.0):
        _require_finite(x0, "Location parameter (x0)")
        _require_positive_finite(gamma, "Scale parameter (gamma)")
        super().__init__(x0=x0, gamma=gamma)

    x0 = _validated_prop(_core.Cauchy.x0, _require_finite,
                         "Location parameter (x0)", "Location parameter x\u2080.")
    gamma = _validated_prop(_core.Cauchy.gamma, _require_positive_finite,
                            "Scale parameter (gamma)", "Scale parameter \u03b3.")

    def fit(self, data):
        super().fit(_validate_fit_data(data))


# SciPy-familiar alias
Normal = Gaussian


# ---------------------------------------------------------------------------
# P2 fix: coerce pdf / log_pdf / cdf inputs across all distribution classes.
#
# The C++ batch overload requires a C-contiguous float64 NDArray; nanobind
# rejects int arrays, lists, and strided views with TypeError.  By contrast,
# fit() already accepts any array-like via _validate_fit_data.  The patch
# below makes pdf / log_pdf / cdf equally permissive: scalars are normalised
# to Python float (scalar C++ overload); everything else is coerced to a
# C-contiguous float64 ndarray (batch C++ overload).
# ---------------------------------------------------------------------------

def _install_batch_coercion(cls, core_cls):
    """Inject pdf / log_pdf / cdf coercion wrappers into *cls*."""
    for _name in ('pdf', 'log_pdf', 'cdf'):
        _m = getattr(core_cls, _name)

        def _make(m, method_name):
            def _wrapper(self, x):
                return m(self, _coerce_batch_input(x))
            _wrapper.__name__ = method_name
            _wrapper.__qualname__ = f'{cls.__qualname__}.{method_name}'
            return _wrapper

        setattr(cls, _name, _make(_m, _name))


for _dist_cls, _core_cls in [
    (Gaussian, _core.Gaussian),
    (Exponential, _core.Exponential),
    (Uniform, _core.Uniform),
    (Poisson, _core.Poisson),
    (DiscreteUniform, _core.DiscreteUniform),
    (Gamma, _core.Gamma),
    (Beta, _core.Beta),
    (ChiSquared, _core.ChiSquared),
    (StudentT, _core.StudentT),
    (LogNormal, _core.LogNormal),
    (Pareto, _core.Pareto),
    (Weibull, _core.Weibull),
    (Rayleigh, _core.Rayleigh),
    (VonMises, _core.VonMises),
    (Binomial, _core.Binomial),
    (NegativeBinomial, _core.NegativeBinomial),
    (Geometric, _core.Geometric),
    (Laplace, _core.Laplace),
    (Cauchy, _core.Cauchy),
]:
    _install_batch_coercion(_dist_cls, _core_cls)
del _dist_cls, _core_cls

__all__ = [
    "Beta",
    "Binomial",
    "Cauchy",
    "ChiSquared",
    "DiscreteUniform",
    "Exponential",
    "Gamma",
    "Gaussian",
    "Geometric",
    "Laplace",
    "LogNormal",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "Rayleigh",
    "StudentT",
    "Uniform",
    "VonMises",
    "Weibull",
]
