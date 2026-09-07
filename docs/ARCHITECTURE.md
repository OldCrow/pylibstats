# pylibstats architecture

## Build model
`_core` is a single nanobind extension module built via scikit-build-core.
`CMakeLists.txt` tries `find_package(libstats)` first at its declared version
floor; if not found, it falls back to `FetchContent` at its `GIT_TAG`, and
accepts a `libstats_DIR` override for local development builds (no implicit
sibling-directory preference — unlike pylibhmm's `../libhmm` behavior, this
is deliberate; see PLAN.md).

## `_common.h` — NumPy ⇔ libstats conversion
Batch **input** is genuinely zero-copy: `pdf`/`log_pdf`/`cdf` construct a
`std::span<const double>` directly over the NumPy array's buffer
(`x.data()`) and pass it straight to libstats' span-based batch methods
(`getProbability(span, span)` etc.) — no intermediate copy, consistent with
libstats' own batch-API design and its SIMD/parallel auto-dispatch. This is
safe because the span is constructed and consumed synchronously within the
same bound call (GIL released only around the libstats call itself), while
the NumPy array argument is kept alive by the Python call frame for the
duration of that call.

Batch **output** is a fresh heap allocation (`new double[n]`) wrapped in a
NumPy array via an `nb::capsule` that deletes the buffer when the array is
garbage-collected — ownership transfers to Python, not shared with C++.

Two exceptions to the zero-copy input path:
- `fit()` copies its input into an owned `std::vector<double>`, since
  libstats' `fit()` takes a vector, not a span.
- Batch `ppf` loops over `p.data()` element-by-element (no span overload)
  because libstats does not expose a batch quantile method — only PDF,
  LogPDF, and CDF have span-based batch overloads.

## `__init__.py` — why the Python wrapper layer exists
`__init__.py` subclasses each `_core` type to add parameter validation
(clear `ValueError` messages) and dtype/shape coercion (`_coerce_batch_input`
distinguishes scalar-like inputs, dispatched to the scalar C++ overload,
from array-likes, coerced to a C-contiguous float64 ndarray for the batch
overload). Per the module's own comment: this validation layer's original
motivation — v1.x ABI safety between Homebrew LLVM and Apple Clang builds —
no longer applies since v2.0 (both libstats and pylibstats now always build
with the same system AppleClang libc++); it's retained purely for UX
consistency (clean `ValueError`s instead of raw C++ exception text).

## Type stubs
`__init__.pyi` and `_core.pyi` are hand-written — no stub-generator
invocation exists in `CMakeLists.txt`, `pyproject.toml`, or CI. Update them
manually whenever `_core.cpp` bindings change.
