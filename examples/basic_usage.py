"""Basic pylibstats usage examples."""


import numpy as np

import pylibstats

# ── Construction ─────────────────────────────────────────────────────────────

gaussian = pylibstats.Gaussian(mu=0.0, sigma=1.0)
gamma = pylibstats.Gamma(alpha=2.0, beta=0.5)
beta = pylibstats.Beta(alpha=2.0, beta=5.0)
poisson = pylibstats.Poisson(lam=4.0)
lognormal = pylibstats.LogNormal(mu=0.0, sigma=1.0)
pareto = pylibstats.Pareto(scale=1.0, alpha=2.0)
weibull = pylibstats.Weibull(shape=2.0, scale=1.0)
rayleigh = pylibstats.Rayleigh(sigma=1.0)
von_mises = pylibstats.VonMises(mu=0.0, kappa=2.0)
binomial = pylibstats.Binomial(n=10, p=0.5)
neg_binom = pylibstats.NegativeBinomial(r=2.0, p=0.5)

# Normal is an alias for Gaussian
normal = pylibstats.Normal(mu=100.0, sigma=15.0)

print("── All 16 distributions ──")
for d in [gaussian, gamma, beta, poisson, lognormal, pareto,
          weibull, rayleigh, von_mises, binomial, neg_binom]:
    print(f"  {d!r}")

# ── Scalar operations ────────────────────────────────────────────────────────

print("\n── Scalar PDF / CDF / PPF ──")
print(f"  Gaussian PDF(0)           = {gaussian.pdf(0.0):.6f}")
print(f"  Gaussian CDF(0)           = {gaussian.cdf(0.0):.6f}")
print(f"  Gaussian PPF(0.975)       = {gaussian.ppf(0.975):.4f}")
print(f"  Gamma PDF(2)              = {gamma.pdf(2.0):.6f}")
print(f"  Beta CDF(0.3)             = {beta.cdf(0.3):.6f}")
print(f"  Poisson PMF(4)            = {poisson.pdf(4.0):.6f}")
print(f"  LogNormal CDF(1) [=0.5]   = {lognormal.cdf(1.0):.6f}")
print(f"  Pareto PDF(scale=1)       = {pareto.pdf(1.0):.6f}")
print(f"  Weibull PDF(1)            = {weibull.pdf(1.0):.6f}")
print(f"  Rayleigh PDF(sigma)       = {rayleigh.pdf(1.0):.6f}")
print(f"  VonMises PDF(mu=0)        = {von_mises.pdf(0.0):.6f}")
print(f"  Binomial PMF(5) [=252/1024]= {binomial.pdf(5.0):.6f}")
print(f"  NegBinom PMF(0) [=0.25]   = {neg_binom.pdf(0.0):.6f}")

# ── Batch operations (SIMD-accelerated) ──────────────────────────────────────

x = np.linspace(-4, 4, 10)
pdf_values = gaussian.pdf(x)
cdf_values = gaussian.cdf(x)

print("\n── Batch operations (10 points) ──")
print(f"  x   = {x}")
print(f"  pdf = {pdf_values}")
print(f"  cdf = {cdf_values}")

# ── Moment properties ────────────────────────────────────────────────────────

print("\n── Moment properties ──")
for name, dist in [
    ("Gaussian(0,1)",   gaussian),
    ("Gamma(2,0.5)",    gamma),
    ("Beta(2,5)",       beta),
    ("Poisson(4)",      poisson),
    ("LogNormal(0,1)",  lognormal),
    ("Pareto(1,2)",     pareto),
    ("Weibull(2,1)",    weibull),
    ("Rayleigh(1)",     rayleigh),
    ("VonMises(0,2)",   von_mises),
    ("Binomial(10,0.5)", binomial),
    ("NegBinom(2,0.5)", neg_binom),
]:
    print(f"  {name:20s}  mean={dist.mean:.4f}  var={dist.variance:.4f}  "
          f"skew={dist.skewness:.4f}  kurt={dist.kurtosis:.4f}")

# ── Sampling ─────────────────────────────────────────────────────────────────

samples = gaussian.sample(n=10_000, seed=42)
print("\n── Sampling ──")
print(f"  10,000 Gaussian samples: mean={samples.mean():.4f}, std={samples.std():.4f}")

beta_samples = beta.sample(n=10_000, seed=42)
print(f"  10,000 Beta(2,5) samples: mean={beta_samples.mean():.4f}, "
      f"min={beta_samples.min():.4f}, max={beta_samples.max():.4f}")

# ── Fitting ──────────────────────────────────────────────────────────────────

data = np.random.default_rng(42).normal(loc=5.0, scale=2.0, size=5000)
fitted = pylibstats.Gaussian()
fitted.fit(data)
print("\n── Fitting ──")
print("  Fitted Gaussian to 5,000 samples from N(5, 2):")
print(f"  mu={fitted.mu:.4f}  sigma={fitted.sigma:.4f}")

# ── Parameter modification ───────────────────────────────────────────────────

print("\n── Parameter modification ──")
dist = pylibstats.Exponential(lam=1.0)
print(f"  Before: lam={dist.lam}, mean={dist.mean:.4f}")
dist.lam = 3.0
print(f"  After:  lam={dist.lam}, mean={dist.mean:.4f}")
