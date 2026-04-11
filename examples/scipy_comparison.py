"""Side-by-side comparison of pylibstats and scipy.stats.

Shows that pylibstats produces the same numerical results as SciPy
while offering a more concise API for common operations.
"""

import numpy as np
from scipy import stats as sp

import pylibstats

TOL = 1e-10

def compare(label: str, pylibstats_val: float, scipy_val: float) -> None:
    match = abs(pylibstats_val - scipy_val) < max(TOL, TOL * abs(scipy_val))
    tag = "OK" if match else "MISMATCH"
    print(f"  [{tag}] {label:30s}  pylibstats={pylibstats_val:.10f}  scipy={scipy_val:.10f}")


print("═" * 78)
print("  pylibstats vs SciPy numerical comparison")
print("═" * 78)

# ── Gaussian ─────────────────────────────────────────────────────────────────

print("\n── Gaussian(mu=2, sigma=0.5) ──")
pl = pylibstats.Gaussian(mu=2.0, sigma=0.5)
sc = sp.norm(loc=2.0, scale=0.5)

for x in [1.0, 2.0, 3.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))
for p in [0.025, 0.5, 0.975]:
    compare(f"PPF({p})", pl.ppf(p), sc.ppf(p))

# ── Exponential ──────────────────────────────────────────────────────────────

print("\n── Exponential(lambda=2.5) ──")
pl = pylibstats.Exponential(lam=2.5)
sc = sp.expon(scale=1.0 / 2.5)  # SciPy uses scale = 1/lambda

for x in [0.0, 0.5, 1.0, 2.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))

# ── Beta ─────────────────────────────────────────────────────────────────────

print("\n── Beta(alpha=2, beta=5) ──")
pl = pylibstats.Beta(alpha=2.0, beta=5.0)
sc = sp.beta(a=2.0, b=5.0)

for x in [0.1, 0.3, 0.5, 0.9]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))

# ── Chi-Squared ──────────────────────────────────────────────────────────────

print("\n── ChiSquared(k=8) ──")
pl = pylibstats.ChiSquared(k=8.0)
sc = sp.chi2(df=8)

for x in [2.0, 5.0, 10.0, 15.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))
    compare(f"CDF({x})", pl.cdf(x), sc.cdf(x))

# ── Student's t ──────────────────────────────────────────────────────────────

print("\n── StudentT(nu=5) ──")
pl = pylibstats.StudentT(nu=5.0)
sc = sp.t(df=5)

for x in [-2.0, 0.0, 1.0, 3.0]:
    compare(f"PDF({x})", pl.pdf(x), sc.pdf(x))

# ── Batch comparison ─────────────────────────────────────────────────────────

print("\n── Batch Gaussian PDF (1000 points) ──")
g = pylibstats.Gaussian(mu=0.0, sigma=1.0)
x = np.linspace(-4, 4, 1000)
pl_pdf = g.pdf(x)
sc_pdf = sp.norm.pdf(x)
max_diff = np.max(np.abs(pl_pdf - sc_pdf))
print(f"  Max absolute difference: {max_diff:.2e}")
print(f"  All within 1e-10: {'YES' if max_diff < 1e-10 else 'NO'}")

print("\n" + "═" * 78)
