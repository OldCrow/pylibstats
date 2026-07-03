#!/usr/bin/env python3
"""pylibstats vs scipy.stats — throughput and accuracy benchmark.

Usage
-----
    python benchmarks/scipy_comparison.py                        # full suite
    python benchmarks/scipy_comparison.py --quick                # N ≤ 100k, fewer reps
    python benchmarks/scipy_comparison.py --sizes 1000,100000    # custom batch sizes
    python benchmarks/scipy_comparison.py --fit                  # include MLE fitting
    python benchmarks/scipy_comparison.py --sizes 1000000        # bandwidth cliff probe

Covers 16 distributions across the pylibstats / scipy.stats shared API.
Not included in this first cut:
  - DiscreteUniform  (no direct scipy equivalent)
  - Geometric        (0-indexed vs 1-indexed support mismatch with scipy.stats.geom)
  - NegativeBinomial (scipy.stats.nbinom requires integer n; pylibstats accepts real r)
  - ppf throughput   (pylibstats ppf is scalar-only; batch comparison is inherently
                      a Python-loop vs vectorised C comparison and measured separately)

Related: https://github.com/OldCrow/pylibstats/issues/2
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy
import scipy.stats as sp

import pylibstats as pls


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BATCH_SIZES_FULL  = [1_000, 10_000, 100_000, 1_000_000]
BATCH_SIZES_QUICK = [1_000, 10_000, 100_000]

# Default reps per batch size.  _default_reps() handles arbitrary N.
REPS_DEFAULT: dict[int, int] = {
    1_000:     200,
    10_000:     60,
    100_000:    20,
    1_000_000:   8,
}
WARMUP_REPS = 3   # discarded before timing begins
ACC_N       = 50_000  # batch size for accuracy measurements


def _default_reps(n: int) -> int:
    """Return a sensible rep count for any batch size."""
    if n in REPS_DEFAULT:
        return REPS_DEFAULT[n]
    if n <= 1_000:       return 200
    if n <= 10_000:      return 60
    if n <= 100_000:     return 20
    if n <= 1_000_000:   return 8
    return 3


# ---------------------------------------------------------------------------
# Distribution catalogue
# ---------------------------------------------------------------------------

@dataclass
class DistSpec:
    """Specification for one distribution in the benchmark."""
    name: str
    make_py: Callable[[], object]             # () → pylibstats instance
    make_sc: Callable[[], object]             # () → frozen scipy rv
    make_x:  Callable[[int], np.ndarray]      # (N) → float64 input array
    make_p:  Callable[[int], np.ndarray]      # (N) → probabilities in (0,1)
    scipy_pdf_meth:    str = "pdf"            # "pdf" or "pmf"
    scipy_logpdf_meth: str = "logpdf"        # "logpdf" or "logpmf"
    # Optional per-distribution notes shown in the accuracy section
    note: str = ""


def _p_grid(n: int) -> np.ndarray:
    """N uniformly spaced probabilities strictly inside (0, 1)."""
    return np.linspace(0.001, 0.999, n)


DISTRIBUTIONS: list[DistSpec] = [
    # ── Continuous: symmetric / unbounded ──────────────────────────────────
    DistSpec(
        name="Gaussian",
        make_py=lambda: pls.Gaussian(mu=2.0, sigma=1.5),
        make_sc=lambda: sp.norm(loc=2.0, scale=1.5),
        make_x=lambda n: np.linspace(-2.0, 6.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="StudentT",
        make_py=lambda: pls.StudentT(nu=5.0),
        make_sc=lambda: sp.t(df=5.0),
        make_x=lambda n: np.linspace(-5.0, 5.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="Cauchy",
        make_py=lambda: pls.Cauchy(x0=0.0, gamma=1.0),
        make_sc=lambda: sp.cauchy(loc=0.0, scale=1.0),
        make_x=lambda n: np.linspace(-8.0, 8.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="Laplace",
        make_py=lambda: pls.Laplace(mu=1.0, b=0.5),
        make_sc=lambda: sp.laplace(loc=1.0, scale=0.5),
        make_x=lambda n: np.linspace(-3.0, 5.0, n),
        make_p=_p_grid,
    ),
    # ── Continuous: positive support ───────────────────────────────────────
    DistSpec(
        name="Exponential",
        # pylibstats: lam = rate;  scipy: scale = 1/rate
        make_py=lambda: pls.Exponential(lam=2.0),
        make_sc=lambda: sp.expon(scale=0.5),
        make_x=lambda n: np.linspace(0.001, 4.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="Gamma",
        # pylibstats: beta = rate;  scipy: scale = 1/beta
        make_py=lambda: pls.Gamma(alpha=2.0, beta=1.0),
        make_sc=lambda: sp.gamma(a=2.0, scale=1.0),
        make_x=lambda n: np.linspace(0.001, 12.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="ChiSquared",
        make_py=lambda: pls.ChiSquared(k=4.0),
        make_sc=lambda: sp.chi2(df=4.0),
        make_x=lambda n: np.linspace(0.001, 14.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="LogNormal",
        # pylibstats: mu=log-mean, sigma=log-std;  scipy: s=sigma, scale=exp(mu)
        make_py=lambda: pls.LogNormal(mu=0.0, sigma=0.5),
        make_sc=lambda: sp.lognorm(s=0.5, scale=math.exp(0.0)),
        make_x=lambda n: np.linspace(0.01, 6.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="Weibull",
        # pylibstats: Weibull(shape=k, scale=λ);  scipy: weibull_min(c=k, scale=λ)
        make_py=lambda: pls.Weibull(shape=2.0, scale=1.5),
        make_sc=lambda: sp.weibull_min(c=2.0, scale=1.5),
        make_x=lambda n: np.linspace(0.001, 5.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="Rayleigh",
        # pylibstats: Rayleigh(sigma);  scipy: rayleigh(scale=sigma)
        make_py=lambda: pls.Rayleigh(sigma=1.5),
        make_sc=lambda: sp.rayleigh(scale=1.5),
        make_x=lambda n: np.linspace(0.001, 7.0, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="Pareto",
        # pylibstats: Pareto(scale=xm, alpha);  scipy: pareto(b=alpha, scale=xm, loc=0)
        make_py=lambda: pls.Pareto(scale=1.0, alpha=2.0),
        make_sc=lambda: sp.pareto(b=2.0, scale=1.0, loc=0.0),
        make_x=lambda n: np.linspace(1.001, 6.0, n),
        make_p=_p_grid,
    ),
    # ── Continuous: bounded ────────────────────────────────────────────────
    DistSpec(
        name="Beta",
        make_py=lambda: pls.Beta(alpha=2.0, beta=3.0),
        make_sc=lambda: sp.beta(a=2.0, b=3.0),
        make_x=lambda n: np.linspace(0.001, 0.999, n),
        make_p=_p_grid,
    ),
    DistSpec(
        name="Uniform",
        # pylibstats: Uniform(a, b);  scipy: uniform(loc=a, scale=b-a)
        make_py=lambda: pls.Uniform(a=0.0, b=5.0),
        make_sc=lambda: sp.uniform(loc=0.0, scale=5.0),
        make_x=lambda n: np.linspace(-0.5, 5.5, n),
        make_p=_p_grid,
    ),
    # ── Circular ───────────────────────────────────────────────────────────
    DistSpec(
        name="VonMises",
        # pylibstats: VonMises(mu, kappa);  scipy: vonmises(kappa, loc=mu)
        make_py=lambda: pls.VonMises(mu=0.0, kappa=2.0),
        make_sc=lambda: sp.vonmises(kappa=2.0, loc=0.0),
        make_x=lambda n: np.linspace(-math.pi, math.pi, n),
        make_p=_p_grid,
        note="scipy vonmises CDF uses numerical integration; ppf excluded",
    ),
    # ── Discrete ───────────────────────────────────────────────────────────
    DistSpec(
        name="Poisson",
        make_py=lambda: pls.Poisson(lam=8.0),
        make_sc=lambda: sp.poisson(mu=8.0),
        # integer-valued inputs, broadcast as float
        make_x=lambda n: np.arange(n, dtype=np.float64) % 30,
        make_p=_p_grid,
        scipy_pdf_meth="pmf",
        scipy_logpdf_meth="logpmf",
    ),
    DistSpec(
        name="Binomial",
        make_py=lambda: pls.Binomial(n=20, p=0.4),
        make_sc=lambda: sp.binom(n=20, p=0.4),
        make_x=lambda n: np.arange(n, dtype=np.float64) % 21,
        make_p=_p_grid,
        scipy_pdf_meth="pmf",
        scipy_logpdf_meth="logpmf",
    ),
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_op(fn: Callable, reps: int) -> float:
    """Return median wall-clock time (seconds) for one call to fn()."""
    # Warmup
    for _ in range(WARMUP_REPS):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _throughput(n: int, t: float) -> float:
    """Elements per second."""
    return n / t if t > 0 else float("inf")


def _fmt_throughput(n: int, t: float) -> str:
    """Format pylibstats throughput compactly: 123k, 456M, 1.2G."""
    eps = _throughput(n, t)
    if eps >= 1e9:
        return f"{eps/1e9:.1f}G"
    if eps >= 1e6:
        return f"{eps/1e6:.0f}M"
    if eps >= 1e3:
        return f"{eps/1e3:.0f}k"
    return f"{eps:.0f}"


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

# (label, pls_method, sc_meth_key)
# sc_meth_key values: "pdf" → spec.scipy_pdf_meth, "logpdf" → spec.scipy_logpdf_meth,
# or a literal attribute name ("cdf") used as-is on the frozen scipy rv.
OPERATIONS = [
    ("pdf",     "pdf",     "pdf"),
    ("log_pdf", "log_pdf", "logpdf"),
    ("cdf",     "cdf",     "cdf"),
]


def run_throughput(specs: list[DistSpec], sizes: list[int]) -> dict:
    """Return nested dict: results[spec.name][op_label][N] = (pls_s, sc_s)."""
    results = {}
    for spec in specs:
        results[spec.name] = {}
        py = spec.make_py()
        sc = spec.make_sc()

        for op_label, pls_meth, sc_meth_override in OPERATIONS:
            results[spec.name][op_label] = {}
            pls_fn = getattr(py, pls_meth)
            # Resolve the scipy method, respecting pmf/logpmf for discrete dists
            if sc_meth_override == "pdf":
                sc_fn = getattr(sc, spec.scipy_pdf_meth)
            elif sc_meth_override == "logpdf":
                sc_fn = getattr(sc, spec.scipy_logpdf_meth)
            else:
                sc_fn = getattr(sc, sc_meth_override)  # "cdf" used as-is

            for n in sizes:
                x = spec.make_x(n)
                reps = _default_reps(n)

                pls_t = _time_op(lambda: pls_fn(x), reps)
                sc_t  = _time_op(lambda: sc_fn(x), reps)
                results[spec.name][op_label][n] = (pls_t, sc_t)

    return results


def print_throughput_table(results: dict, sizes: list[int]) -> None:
    """Print speedup ratios and pylibstats absolute throughput side by side.

    Layout constants
    ----------------
    RATIO_W = 5   "17.0×"  (f"{r:4.1f}×")
    INNER   = 2   intra-super-column gap between × and pls/s
    TPUT_W  = 5   " 800M"  (right-justified in 5; fits "pls/s" as label)
    CELL_W  = 12  = RATIO_W + INNER + TPUT_W  (one super-column width)
    OUTER   = 4   inter-super-column gap (2× the inner gap)

    All three rows — N= header, ×/pls/s sub-header, data — are CELL_W+OUTER
    chars wide per super-column, so the separator is always exact.
    """
    dist_names = list(results.keys())
    ops        = list(results[dist_names[0]].keys())

    RATIO_W = 5
    INNER   = 2
    TPUT_W  = 5
    CELL_W  = RATIO_W + INNER + TPUT_W   # 12
    OUTER   = 4
    outer_s = " " * OUTER

    for op in ops:
        print(f"\n── {op} ──")
        # Row 1: N= labels, each centred over its CELL_W super-column
        h1 = f"{'Distribution':<16}" + "".join(
            f"{outer_s}{_fmt_n(n):^{CELL_W}}" for n in sizes
        )
        # Row 2: sub-column labels; × right-justified in RATIO_W, pls/s right in TPUT_W
        h2 = f"{'':16}" + "".join(
            f"{outer_s}{'×':>{RATIO_W}}  {'pls/s':>{TPUT_W}}" for _ in sizes
        )
        sep = "─" * len(h1)
        print(h1)
        print(h2)
        print(sep)
        for name in dist_names:
            row = f"{name:<16}"
            for n in sizes:
                pls_t, sc_t = results[name][op][n]
                ratio = sc_t / pls_t if pls_t > 0 else float("inf")
                tput  = _fmt_throughput(n, pls_t)
                row += f"{outer_s}{ratio:4.1f}×  {tput:>{TPUT_W}}"
            print(row)
        print("  (>1× = pylibstats faster; pls/s = absolute pylibstats throughput)")


def _fmt_n(n: int) -> str:
    if n >= 1_000_000:
        return f"N={n//1_000_000}M"
    if n >= 1_000:
        return f"N={n//1_000}k"
    return f"N={n}"


# ---------------------------------------------------------------------------
# Accuracy benchmark
# ---------------------------------------------------------------------------

def run_accuracy(specs: list[DistSpec]) -> dict:
    """Return dict: accuracy[spec.name][op] = max_rel_err (float).

    Relative error is computed only over interior points where the scipy
    reference is finite and non-negligible (abs > 1e-10).  This avoids
    misleading numbers at domain boundaries where both libraries return 0
    or -inf and the denominator collapses.
    """
    accuracy = {}
    for spec in specs:
        accuracy[spec.name] = {}
        py = spec.make_py()
        sc = spec.make_sc()
        x  = spec.make_x(ACC_N)

        checks = [
            ("pdf",     getattr(py, "pdf"),     getattr(sc, spec.scipy_pdf_meth)),
            ("log_pdf", getattr(py, "log_pdf"), getattr(sc, spec.scipy_logpdf_meth)),
            ("cdf",     getattr(py, "cdf"),     getattr(sc, "cdf")),
        ]
        for op, pls_fn, sc_fn in checks:
            with np.errstate(invalid="ignore", divide="ignore"):
                pls_vals = pls_fn(x)
                sc_vals  = sc_fn(x)
                abs_ref  = np.abs(sc_vals)
                abs_diff = np.abs(pls_vals - sc_vals)
                # Restrict to interior points: both finite, reference non-negligible.
                valid = (
                    np.isfinite(pls_vals) &
                    np.isfinite(sc_vals)  &
                    (abs_ref > 1e-10)
                )
            if valid.any():
                rel_err = float(np.max(abs_diff[valid] / abs_ref[valid]))
            else:
                rel_err = float("nan")  # no valid comparison points on this grid
            accuracy[spec.name][op] = rel_err

    return accuracy


def print_accuracy_table(accuracy: dict) -> None:
    print(f"\n{'Distribution':<16}  {'pdf':>12}  {'log_pdf':>12}  {'cdf':>12}")
    print("─" * 58)
    FLAG = 1e-10  # flag anything larger than this
    for name, ops in accuracy.items():
        row = f"{name:<16}"
        for op in ("pdf", "log_pdf", "cdf"):
            e = ops[op]
            mark = " !" if e > FLAG else "  "
            row += f"  {e:>10.2e}{mark}"
        print(row)
    print("(! = relative error > 1e-10; investigate parameterisation or approximation)")


# ---------------------------------------------------------------------------
# MLE fitting benchmark
# ---------------------------------------------------------------------------

# Scipy fit() returns (param..., loc, scale); the mapping back to pylibstats
# parameters is distribution-specific. We only time here — not compare accuracy.
# Full parameter-recovery accuracy (M=100 replicates) is deferred to a follow-up.

@dataclass
class FitSpec:
    name: str
    make_py_fresh: Callable[[], object]   # fresh pylibstats instance to fit into
    sc_dist: object                       # unfrozen scipy distribution class
    sc_fit_kwargs: dict                   # passed to sc_dist.fit()
    make_data: Callable[[int], np.ndarray]


FIT_DISTRIBUTIONS: list[FitSpec] = [
    FitSpec(
        name="Gaussian",
        make_py_fresh=lambda: pls.Gaussian(),
        sc_dist=sp.norm,
        sc_fit_kwargs={},
        make_data=lambda n: sp.norm(loc=2.0, scale=1.5).rvs(n, random_state=42),
    ),
    FitSpec(
        name="Exponential",
        make_py_fresh=lambda: pls.Exponential(),
        sc_dist=sp.expon,
        sc_fit_kwargs={"floc": 0},          # fix loc=0 for fair rate comparison
        make_data=lambda n: sp.expon(scale=0.5).rvs(n, random_state=42),
    ),
    FitSpec(
        name="Gamma",
        make_py_fresh=lambda: pls.Gamma(),
        sc_dist=sp.gamma,
        sc_fit_kwargs={"floc": 0},
        make_data=lambda n: sp.gamma(a=2.0, scale=1.0).rvs(n, random_state=42),
    ),
    FitSpec(
        name="Beta",
        make_py_fresh=lambda: pls.Beta(),
        sc_dist=sp.beta,
        sc_fit_kwargs={"floc": 0, "fscale": 1},
        make_data=lambda n: sp.beta(a=2.0, b=3.0).rvs(n, random_state=42),
    ),
    FitSpec(
        name="LogNormal",
        make_py_fresh=lambda: pls.LogNormal(),
        sc_dist=sp.lognorm,
        sc_fit_kwargs={"floc": 0},
        make_data=lambda n: sp.lognorm(s=0.5, scale=1.0).rvs(n, random_state=42),
    ),
]

FIT_SIZES = [1_000, 10_000, 100_000]
FIT_REPS  = {1_000: 50, 10_000: 15, 100_000: 5}


def run_fit(fit_specs: list[FitSpec]) -> dict:
    """Return nested dict: fit_results[name][N] = (pls_t, sc_t)."""
    fit_results = {}
    for fs in fit_specs:
        fit_results[fs.name] = {}
        for n in FIT_SIZES:
            data = fs.make_data(n)
            reps = FIT_REPS[n]

            def pls_fit():
                d = fs.make_py_fresh()
                d.fit(data)

            def sc_fit():
                fs.sc_dist.fit(data, **fs.sc_fit_kwargs)

            pls_t = _time_op(pls_fit, reps)
            sc_t  = _time_op(sc_fit,  reps)
            fit_results[fs.name][n] = (pls_t, sc_t)

    return fit_results


def print_fit_table(fit_results: dict) -> None:
    sizes = FIT_SIZES
    print(f"\n{'Distribution':<16}" + "".join(f"  {_fmt_n(n):>9}" for n in sizes))
    print("─" * (16 + len(sizes) * 11))
    for name, ns in fit_results.items():
        row = f"{name:<16}"
        for n in sizes:
            pls_t, sc_t = ns[n]
            ratio = sc_t / pls_t if pls_t > 0 else float("inf")
            row += f"  {ratio:>8.1f}x"
        print(row)
    print("(>1× = pylibstats fit() faster)")


# ---------------------------------------------------------------------------
# Header / footer helpers
# ---------------------------------------------------------------------------

def _cpu_name() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        return r.stdout.strip() if r.returncode == 0 else platform.processor()
    except Exception:
        return platform.processor()


def print_header() -> None:
    print("=" * 72)
    print("pylibstats vs scipy.stats — throughput and accuracy benchmark")
    print("=" * 72)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  CPU     : {_cpu_name()}")
    try:
        print(f"  pylibstats: {pls.__version__}")  # type: ignore[attr-defined]
    except AttributeError:
        pass
    print(f"  scipy   : {scipy.__version__}")
    print(f"  numpy   : {np.__version__}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="pylibstats vs scipy.stats throughput and accuracy benchmark"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Limit to N ≤ 100k and reduce repetitions for a fast smoke-test",
    )
    parser.add_argument(
        "--sizes",
        metavar="N[,N,...]",
        default=None,
        help="Comma-separated batch sizes to benchmark, e.g. 1000,100000,1000000. "
             "Overrides --quick. Rep counts are chosen automatically per N.",
    )
    parser.add_argument(
        "--fit", action="store_true",
        help="Include MLE fitting timing section (adds ~30-60s)",
    )
    args = parser.parse_args()

    if args.sizes:
        try:
            sizes = [int(s.strip()) for s in args.sizes.split(",")]
            if not sizes:
                parser.error("--sizes requires at least one value")
        except ValueError:
            parser.error("--sizes values must be integers, e.g. 1000,100000,1000000")
    elif args.quick:
        sizes = BATCH_SIZES_QUICK
    else:
        sizes = BATCH_SIZES_FULL

    print_header()

    # ── Throughput ──────────────────────────────────────────────────────────
    print("Throughput  (ratio = scipy_time / pylibstats_time;  "
          "pls/s = pylibstats elements per second)")
    print(f"Median of {WARMUP_REPS} warmup + timed reps; "
          f"rep counts: {', '.join(f'{_fmt_n(n)}={_default_reps(n)}' for n in sizes)}")
    print()

    results = run_throughput(DISTRIBUTIONS, sizes)
    print_throughput_table(results, sizes)

    # ── Accuracy ────────────────────────────────────────────────────────────
    print(f"\nAccuracy (max relative error vs scipy, N={ACC_N:,})")
    print("  max |pylibstats(x) - scipy(x)| / |scipy(x)|  over a uniform grid")
    print()

    accuracy = run_accuracy(DISTRIBUTIONS)
    print_accuracy_table(accuracy)

    # ── Notes on excluded ops / distributions ──────────────────────────────
    print("\nNotes")
    print("  ppf: pylibstats ppf() is scalar-only; batch comparison deferred.")
    print("  Geometric: 0-indexed (pylibstats) vs 1-indexed (scipy.stats.geom).")
    print("  NegativeBinomial: scipy.stats.nbinom requires integer n; "
          "pylibstats accepts real r.")
    print("  DiscreteUniform: no direct scipy equivalent.")
    for spec in DISTRIBUTIONS:
        if spec.note:
            print(f"  {spec.name}: {spec.note}")

    # ── Fitting (optional) ──────────────────────────────────────────────────
    if args.fit:
        print("\nMLE fitting (speedup ratio: pylibstats.fit() vs scipy.dist.fit())")
        print("Note: per-call overhead includes fresh distribution object creation "
              "for pylibstats. Full parameter-recovery accuracy (M=100 replicates) "
              "deferred to follow-up.")
        fit_results = run_fit(FIT_DISTRIBUTIONS)
        print_fit_table(fit_results)

    print()


if __name__ == "__main__":
    main()
