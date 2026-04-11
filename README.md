# pylibstats

Python bindings for [libstats](https://github.com/OldCrow/libstats) — a C++20 statistical distributions library with SIMD batch operations.

## Features

- **9 distributions**: Gaussian, Exponential, Uniform, Poisson, Discrete Uniform, Gamma, Beta, Chi-Squared, Student's t
- **NumPy integration**: pass arrays directly to `pdf()`, `cdf()`, `log_pdf()` — the SIMD/parallel batch path runs automatically
- **GIL-releasing**: batch operations release the Python GIL for concurrent workloads
- **SciPy-compatible naming**: `pdf`, `cdf`, `ppf`, `fit`, `sample`

## Quick start

```python
import numpy as np
import pylibstats

dist = pylibstats.Gaussian(mu=0.0, sigma=1.0)

# Scalar
dist.pdf(1.0)
dist.cdf(0.0)        # 0.5
dist.ppf(0.975)      # ~1.96

# Batch (SIMD-accelerated)
x = np.linspace(-4, 4, 100_000)
densities = dist.pdf(x)

# Sampling
samples = dist.sample(n=10_000, seed=42)

# Fitting
dist.fit(samples)
```

## Building from source

Requires Python ≥3.9, CMake ≥3.20, and a C++20 compiler.

```bash
pip install .
```

This fetches libstats v1.0.0 via CMake FetchContent if not already installed.

## Running tests

```bash
pip install ".[test]"
pytest
```

## License

MIT
