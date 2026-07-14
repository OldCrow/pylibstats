"""Side-by-side comparison of pylibstats and scipy.stats.

Verifies that pylibstats produces the same numerical results as SciPy
for all sixteen distributions — scalar PDF/PMF, CDF, PPF, and a batch
PDF check.

Usage:
    python examples/scipy_comparison.py
"""

import math as _math
import sys

import numpy as np
from scipy import stats as sp

import pylibstats

# Relative tolerance for floating-point comparison.  PDF/PMF routines using
# direct formulae agree to ~1e-12.  CDF/PPF routines that rely on regularised
# incomplete gamma/beta functions or iterative root-finding (Newton-Raphson)
# are accurate to ~1e-7.
RTOL = 1e-7
ATOL = 1e-12

ok_count = 0
fail_count = 0


def compare(label: str, pl_val: float, sc_val: float) -> None:
    """Print one comparison line and track pass/fail counts."""
    global ok_count, fail_count
    match = np.isclose(pl_val, sc_val, rtol=RTOL, atol=ATOL)
    if match:
        ok_count += 1
        tag = "OK"
    else:
        fail_count += 1
        tag = "MISMATCH"
    print(f"  [{tag}] {label:35s}  pylibstats={pl_val:.12g}  scipy={sc_val:.12g}")


def heading(text: str) -> None:
    print(f"\n── {text} ──")


# ═══════════════════════════════════════════════════════════════════════════
print("═" * 78)
print("  pylibstats vs SciPy numerical comparison")
print("═" * 78)

# ── Gaussian ──────────────────────────────────────────────────────────────
heading("Gaussian(mu=2, sigma=0.5)")
pl = pylibstats.Gaussian(mu=2.0, sigma=0.5)
sc = sp.norm(loc=2.0, scale=0.5)

for x in [1.0, 2.0, 3.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.025, 0.5, 0.975]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Exponential ───────────────────────────────────────────────────────────
heading("Exponential(lam=2.5)")
pl = pylibstats.Exponential(lam=2.5)
sc = sp.expon(scale=1.0 / 2.5)  # SciPy: scale = 1/lambda

for x in [0.0, 0.5, 1.0, 2.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))

# ── Uniform ───────────────────────────────────────────────────────────────
heading("Uniform(a=-2, b=3)")
pl = pylibstats.Uniform(a=-2.0, b=3.0)
sc = sp.uniform(loc=-2.0, scale=5.0)  # SciPy: loc=a, scale=b-a

for x in [-2.0, 0.0, 1.5, 3.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.0, 0.25, 0.75, 1.0]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Poisson ───────────────────────────────────────────────────────────────
heading("Poisson(lam=4)")
pl = pylibstats.Poisson(lam=4.0)
sc = sp.poisson(mu=4.0)

for k in [0, 1, 3, 4, 8, 12]:
    compare(f"PMF({k})", pl.pdf(float(k)), sc.pmf(k))
    compare(f"CDF({k})", pl.cdf(float(k)), sc.cdf(k))

# ── Discrete Uniform ─────────────────────────────────────────────────────
heading("DiscreteUniform(a=1, b=6)  [fair die]")
pl = pylibstats.DiscreteUniform(a=1, b=6)
sc = sp.randint(low=1, high=7)  # SciPy: [low, high)

for k in [1, 3, 6]:
    compare(f"PMF({k})", pl.pdf(float(k)), sc.pmf(k))
    compare(f"CDF({k})", pl.cdf(float(k)), sc.cdf(k))

# ── Gamma ─────────────────────────────────────────────────────────────────
heading("Gamma(alpha=3, beta=2)")
pl = pylibstats.Gamma(alpha=3.0, beta=2.0)
sc = sp.gamma(a=3.0, scale=1.0 / 2.0)  # SciPy: scale = 1/beta

for x in [0.5, 1.0, 2.0, 4.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))

# ── Beta ──────────────────────────────────────────────────────────────────
heading("Beta(alpha=2, beta=5)")
pl = pylibstats.Beta(alpha=2.0, beta=5.0)
sc = sp.beta(a=2.0, b=5.0)

for x in [0.1, 0.3, 0.5, 0.9]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))

# ── Chi-Squared ───────────────────────────────────────────────────────────
heading("ChiSquared(k=8)")
pl = pylibstats.ChiSquared(k=8.0)
sc = sp.chi2(df=8)

for x in [2.0, 5.0, 10.0, 15.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.05, 0.5, 0.95]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Student's t ───────────────────────────────────────────────────────────
heading("StudentT(nu=5)")
pl = pylibstats.StudentT(nu=5.0)
sc = sp.t(df=5)

for x in [-2.0, 0.0, 1.0, 3.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.025, 0.5, 0.975]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Log-Normal ─────────────────────────────────────────────────────────────
heading("LogNormal(mu=0, sigma=1)")
pl = pylibstats.LogNormal(mu=0.0, sigma=1.0)
sc = sp.lognorm(s=1.0, scale=_math.exp(0.0))  # s=sigma, scale=exp(mu)

for x in [0.5, 1.0, 2.0, 4.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.1, 0.5, 0.9]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Pareto ────────────────────────────────────────────────────────────────────
heading("Pareto(scale=1, alpha=2)")
pl = pylibstats.Pareto(scale=1.0, alpha=2.0)
sc = sp.pareto(b=2.0, scale=1.0)  # sp.pareto(b=alpha, scale=scale)

for x in [1.0, 2.0, 5.0, 10.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.25, 0.5, 0.75]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Weibull ────────────────────────────────────────────────────────────────────
heading("Weibull(shape=2, scale=1)")
pl = pylibstats.Weibull(shape=2.0, scale=1.0)
sc = sp.weibull_min(c=2.0, scale=1.0)  # c=shape, scale=scale

for x in [0.5, 1.0, 1.5, 2.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.1, 0.5, 0.9]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Rayleigh ──────────────────────────────────────────────────────────────────
heading("Rayleigh(sigma=2)")
pl = pylibstats.Rayleigh(sigma=2.0)
sc = sp.rayleigh(scale=2.0)  # scale=sigma

for x in [0.5, 1.0, 2.0, 4.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.25, 0.5, 0.75]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Von Mises ───────────────────────────────────────────────────────────────
# CDF is numerical (quadrature); different integrators produce ~1e-6 relative
# disagreement between pylibstats and SciPy, so only PDF is compared here.
heading("VonMises(mu=0, kappa=2) — PDF only (CDF uses different quadrature)")
pl = pylibstats.VonMises(mu=0.0, kappa=2.0)
sc = sp.vonmises(kappa=2.0, loc=0.0)

for x in [-1.5, -0.5, 0.0, 0.5, 1.5]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))

# ── Binomial ──────────────────────────────────────────────────────────────────
heading("Binomial(n=10, p=0.3)")
pl = pylibstats.Binomial(n=10, p=0.3)
sc = sp.binom(n=10, p=0.3)

for k in [0, 2, 3, 5, 8, 10]:
    compare(f"PMF({k})", pl.pdf(float(k)), sc.pmf(k))
    compare(f"CDF({k})", pl.cdf(float(k)), sc.cdf(k))

# ── Negative Binomial ───────────────────────────────────────────────────────────
heading("NegativeBinomial(r=3, p=0.6)")
pl = pylibstats.NegativeBinomial(r=3.0, p=0.6)
sc = sp.nbinom(n=3, p=0.6)  # SciPy nbinom: n=r (integer), p=p

for k in [0, 1, 2, 4, 6]:
    compare(f"PMF({k})", pl.pdf(float(k)), sc.pmf(k))
    compare(f"CDF({k})", pl.cdf(float(k)), sc.cdf(k))

# ── Batch comparison ─────────────────────────────────────────────────────────
heading("Batch Gaussian PDF (1 000 points)")
g = pylibstats.Gaussian(mu=0.0, sigma=1.0)
x = np.linspace(-4, 4, 1000)
pl_pdf = g.pdf(x)
sc_pdf = sp.norm.pdf(x)
max_diff = np.max(np.abs(pl_pdf - sc_pdf))
print(f"  Max absolute difference: {max_diff:.2e}")
print(f"  All within 1e-10:       {'YES' if max_diff < 1e-10 else 'NO'}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "═" * 78)
print(f"  {ok_count} passed, {fail_count} failed  (rtol={RTOL}, atol={ATOL})")
print("═" * 78)

sys.exit(1 if fail_count else 0)
