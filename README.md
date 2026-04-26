# pylibstats

Python bindings for [libstats](https://github.com/OldCrow/libstats) — a C++20 statistical distributions library with SIMD batch operations.

## Features

- **9 distributions**: Gaussian, Exponential, Uniform, Poisson, Discrete Uniform, Gamma, Beta, Chi-Squared, Student's t
- **NumPy integration**: pass arrays directly to `pdf()`, `cdf()`, `log_pdf()` — the SIMD/parallel batch path runs automatically
- **GIL-releasing**: batch operations release the Python GIL for concurrent workloads
- **SciPy-compatible naming**: `pdf`, `cdf`, `ppf`, `fit`, `sample`
- **Input validation**: all constructor, setter, and `fit()` parameters are validated in Python with clear `ValueError` messages

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

Requires Python ≥3.11, CMake ≥3.20, and a C++20 compiler.

```bash
pip install .
```

This fetches libstats v1.1.2 via CMake FetchContent if not already installed.

### Building against a local libstats

To link against a locally built libstats (e.g. a development branch), install
libstats to a prefix and point `pip` at it:

```bash
# In the libstats repo
cmake --install build --prefix /path/to/libstats/install

# In this repo — use libstats_DIR, not CMAKE_PREFIX_PATH
# (overriding CMAKE_PREFIX_PATH breaks nanobind discovery)
pip install --no-build-isolation -ve . \
    -Ccmake.define.libstats_DIR=/path/to/libstats/install/lib/cmake/libstats
```

`--no-build-isolation` requires build deps in the active environment:

```bash
pip install "scikit-build-core>=0.10" "nanobind>=2.0"
```

## Running tests

```bash
pip install ".[test]"
pytest
```

## Examples

See the `examples/` directory:

- `basic_usage.py` — scalar/batch operations, sampling, and fitting
- `benchmark.py` — wall-clock comparison against SciPy (PDF and CDF)
- `scipy_comparison.py` — numerical accuracy verification across all 9 distributions

## Known limitations

- **Beta CDF performance**: the regularised incomplete beta function in libstats is slower than SciPy's implementation (~0.5× speedup). All other distribution/operation combinations are faster.

## Contributing

### macOS ABI note

On macOS, libstats may be compiled with Homebrew LLVM while Python extensions use Apple clang. These ship different `libc++` versions whose exception-handling ABIs are incompatible — C++ exceptions thrown from libstats segfault during stack unwinding instead of propagating normally.

pylibstats works around this by validating all parameters in pure Python (in `__init__.py`) *before* calling into the C++ layer, so the error path never crosses the ABI boundary. If you add new parameters or distribution classes, follow the same pattern: validate in Python, then delegate to `_core`.

See `libstats/include/core/error_handling.h` for the upstream discussion.

## License

MIT
