"""Benchmark pylibstats batch operations against SciPy.

Measures wall-clock time for PDF and CDF evaluation on arrays of 100k and 1M
elements across all continuous distributions. Prints a formatted table with
per-distribution speedup ratios.

Note: Von Mises, Binomial, and NegativeBinomial use scalar loops (no SIMD)
but still benefit from cached loop-invariants vs repeated SciPy calls.

Usage:
    python examples/benchmark.py
"""

import time

import numpy as np
from scipy import stats as sp

import pylibstats


def bench(fn, *, warmup: int = 2, repeats: int = 5) -> float:
    """Return median wall-clock seconds for fn()."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ── Distribution configs ─────────────────────────────────────────────────────
# Each entry: (name, pylibstats instance, scipy frozen instance, x_range)

DISTRIBUTIONS = [
    ("Gaussian(0,1)",
     pylibstats.Gaussian(0.0, 1.0),
     sp.norm(0.0, 1.0),
     (-4.0, 4.0)),

    ("Exponential(1)",
     pylibstats.Exponential(1.0),
     sp.expon(scale=1.0),
     (0.01, 10.0)),

    ("Uniform(0,1)",
     pylibstats.Uniform(0.0, 1.0),
     sp.uniform(0.0, 1.0),
     (0.0, 1.0)),

    ("Gamma(2,1)",
     pylibstats.Gamma(2.0, 1.0),
     sp.gamma(a=2.0, scale=1.0),
     (0.01, 10.0)),

    ("Beta(2,5)",
     pylibstats.Beta(2.0, 5.0),
     sp.beta(a=2.0, b=5.0),
     (0.01, 0.99)),

    ("ChiSquared(5)",
     pylibstats.ChiSquared(5.0),
     sp.chi2(df=5),
     (0.1, 20.0)),

    ("StudentT(10)",
     pylibstats.StudentT(10.0),
     sp.t(df=10),
     (-5.0, 5.0)),

    ("LogNormal(0,1)",
     pylibstats.LogNormal(0.0, 1.0),
     sp.lognorm(s=1.0, scale=1.0),
     (0.01, 8.0)),

    ("Pareto(1,2)",
     pylibstats.Pareto(1.0, 2.0),
     sp.pareto(b=2.0, scale=1.0),
     (1.0, 20.0)),

    ("Weibull(2,1)",
     pylibstats.Weibull(2.0, 1.0),
     sp.weibull_min(c=2.0, scale=1.0),
     (0.01, 4.0)),

    ("Rayleigh(1)",
     pylibstats.Rayleigh(1.0),
     sp.rayleigh(scale=1.0),
     (0.01, 6.0)),

    ("VonMises(0,2)",
     pylibstats.VonMises(0.0, 2.0),
     sp.vonmises(kappa=2.0, loc=0.0),
     (-3.14, 3.14)),
]

SIZES = [100_000, 1_000_000]


def run_benchmarks() -> None:
    header = (f"{'Distribution':>20s}  {'N':>10s}  {'Op':>4s}  "
              f"{'pylibstats (ms)':>15s}  {'SciPy (ms)':>12s}  {'Speedup':>8s}")
    sep = "─" * len(header)

    print()
    print(header)
    print(sep)

    for name, pl_dist, sc_dist, (lo, hi) in DISTRIBUTIONS:
        for n in SIZES:
            x = np.linspace(lo, hi, n)

            # PDF — bind loop vars via default args to avoid late-binding
            t_pl = bench(lambda d=pl_dist, arr=x: d.pdf(arr)) * 1000
            t_sc = bench(lambda d=sc_dist, arr=x: d.pdf(arr)) * 1000
            ratio = t_sc / t_pl if t_pl > 0 else float("inf")
            print(f"{name:>20s}  {n:>10,d}  {'PDF':>4s}  "
                  f"{t_pl:>15.2f}  {t_sc:>12.2f}  {ratio:>7.1f}x")

            # CDF
            t_pl = bench(lambda d=pl_dist, arr=x: d.cdf(arr)) * 1000
            t_sc = bench(lambda d=sc_dist, arr=x: d.cdf(arr)) * 1000
            ratio = t_sc / t_pl if t_pl > 0 else float("inf")
            print(f"{'':>20s}  {'':>10s}  {'CDF':>4s}  "
                  f"{t_pl:>15.2f}  {t_sc:>12.2f}  {ratio:>7.1f}x")

    print(sep)
    print()


if __name__ == "__main__":
    import scipy
    print("pylibstats vs SciPy batch performance benchmark")
    print(f"NumPy {np.__version__}, SciPy {scipy.__version__}")
    print(f"Repeats: 5 (median), Warmup: 2")
    run_benchmarks()
