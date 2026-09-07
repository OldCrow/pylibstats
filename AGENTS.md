# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project Overview

`pylibstats` provides Python bindings for `libstats` via `nanobind` and `scikit-build-core`.

Core goals:

- expose the `libstats` distribution API in Python
- keep NumPy-based batch paths fast and simple
- keep Python wrappers, native bindings, and stubs synchronized

Key files:

- `CMakeLists.txt` — native extension build and `libstats` dependency wiring
- `pyproject.toml` — package metadata and build backend config
- `src/pylibstats/_core.cpp` — nanobind bindings
- `src/pylibstats/_common.h` — NumPy conversion helpers
- `src/pylibstats/__init__.py` — Python wrappers and validation
- `src/pylibstats/__init__.pyi`, `src/pylibstats/_core.pyi` — typing stubs
- `tests/` — pytest suite

Dependency notes:

- Build first tries `find_package(libstats)` at the version floor in `CMakeLists.txt`.
- If not found, CMake fetches `libstats` from the `GIT_TAG` in `CMakeLists.txt`.
- Those two version strings live only in `CMakeLists.txt` and are deliberately
  not restated here; a `pin-currency` CI job asserts they agree with each other
  and with libstats' newest release (see PLAN.md).
- For local development against a custom `libstats` install, pass `libstats_DIR` (do not override `CMAKE_PREFIX_PATH`, which can break nanobind discovery).

## Architecture

### Build model
`_core` is a single nanobind extension module built via scikit-build-core.
`CMakeLists.txt` tries `find_package(libstats)` first at its declared version
floor; if not found, it falls back to `FetchContent` at its `GIT_TAG`, and
accepts a `libstats_DIR` override for local development builds (no implicit
sibling-directory preference — unlike pylibhmm's `../libhmm` behavior, this
is deliberate; see PLAN.md).

### `_common.h` — NumPy ⇔ libstats conversion
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

### `__init__.py` — why the Python wrapper layer exists
`__init__.py` subclasses each `_core` type to add parameter validation
(clear `ValueError` messages) and dtype/shape coercion (`_coerce_batch_input`
distinguishes scalar-like inputs, dispatched to the scalar C++ overload,
from array-likes, coerced to a C-contiguous float64 ndarray for the batch
overload). Per the module's own comment: this validation layer's original
motivation — v1.x ABI safety between Homebrew LLVM and Apple Clang builds —
no longer applies since v2.0 (both libstats and pylibstats now always build
with the same system AppleClang libc++); it's retained purely for UX
consistency (clean `ValueError`s instead of raw C++ exception text).

### Type stubs
`__init__.pyi` and `_core.pyi` are hand-written — no stub-generator
invocation exists in `CMakeLists.txt`, `pyproject.toml`, or CI. Update them
manually whenever `_core.cpp` bindings change.

## Session Start

**Requires Python ≥ 3.11.** Follow the standard architecture-check ritual:
[SESSION-START.md](https://github.com/OldCrow/standards/blob/main/SESSION-START.md).

## Build Commands

Run from the repository root:

```bash
# macOS/Linux
python -m pip install -e ".[test]"   # installs package (editable) + test dependencies
python -m pytest tests -q
```

```powershell
# Windows
python -m pip install -e ".[test]"
python -m pytest tests -q
```

Local `libstats` build override (replace `<libstats-install>` with the path to your `libstats` CMake installation):

```bash
# macOS/Linux
python -m pip install --no-build-isolation -ve . \
  -Ccmake.define.libstats_DIR=<libstats-install>/lib/cmake/libstats
```

```powershell
# Windows
python -m pip install --no-build-isolation -ve . `
  -Ccmake.define.libstats_DIR=<libstats-install>\lib\cmake\libstats
```

### CMake standard

Full rules: [CMake House Style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md)
in the fleet standards repo; this section is self-sufficient for this repo. pylibstats is built via
scikit-build-core (the `pip install -e` path above is primary and
authoritative); `CMakePresets.json` (schema 6, min CMake 3.25) exists only
for direct-CMake dev/debugging, not the normal workflow: `release` →
`build/`, `debug` → `build-debug/`. No project extras, no `generator`
field. Deviation: prefers an installed `find_package(libstats)` over a
local sibling checkout (deliberate contrast to pylibhmm — see Build model
above and PLAN.md); falls back to the pinned `FetchContent` tag only
when no installed libstats is found.

## Platform-Specific Notes

**Minimum macOS:** 13 Ventura (AppleClang 15 / Xcode 15). macOS Catalina (10.15) is not supported from `pylibstats` v0.3.0 / `libstats` v2.0.0 onwards.

### macOS

- Use the active Python environment for install/test.
- If linking to local `libstats`, ensure that `libstats` was built for the same machine architecture (`arm64` vs `x86_64`).

```bash
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -q
```

### Linux

- Requires GCC ≥ 13 or Clang ≥ 17 for C++20 support.
- If `libstats` is not found locally, CMake fetches it automatically at the pinned `GIT_TAG` (see `CMakeLists.txt`).

```bash
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -q
```

### Windows (MSVC)

- **Minimum toolchain**: Visual Studio 2022 (17.x) or later with the C++
  desktop workload — any MSVC toolset with full C++20 support. Verified
  through VS 2026 (v18, MSVC 14.5x). Build Tools or a full IDE edition both
  work.
- Don't pin a generator locally — see
  [WINDOWS-TOOLCHAIN.md §3](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md)
  and [CMAKE-HOUSE-STYLE.md](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md).
- If using `libstats_DIR`, the referenced `libstats` install must be built
  with compatible MSVC/x64 settings.

```powershell
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -q
```

#### Windows toolchain setup

pylibstats needs no per-session `vcvars` activation: the Visual Studio CMake
generator locates its own toolchain (the "VS generator" case in
[WINDOWS-TOOLCHAIN.md](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md) —
not the case that requires activation, which applies only to non-VS
generators or direct `cl.exe` use). See that doc for one-time setup, the
Smart App Control note, and the CMake version requirement.

## Coding Conventions

1. Keep `_core.cpp`, `__init__.py`, and `.pyi` stubs consistent (hand-edit — see Architecture).
2. Add or update tests for any behavior/API change.
3. Keep documentation concise and accurate.

### Linting

**Python** (`src/pylibstats/__init__.py`, `tests/`, `examples/`): ruff, config
in `pyproject.toml`. Scoped to `E`/`F`/`I`/`UP`/`B` — unlike pylibhmm, `B`
(flake8-bugbear) is clean here and fully enabled, no deferral needed.
```bash
ruff check src/pylibstats tests examples
ruff format src/pylibstats tests examples
```

**Type checking**: pyright via the editor/agent language server only, not
CI. `[tool.pyright]` in `pyproject.toml` points it at `.venv` so `numpy`
and the editable install resolve, and silences the "no source" warning for
the compiled `_core` module. `_core.pyi` must declare every `def_prop_rw`
parameter property: `__init__.py` reads the nanobind descriptors off the
`_core` classes to build its validated wrappers, so an undeclared property
is a pyright error at each use site. `__init__.py` carries a file-level
`# pyright: reportIncompatibleMethodOverride=false` because pyright rejects
`name = _validated_prop(...)` as an override of a property regardless of
type; consumers type-check against `__init__.pyi`, not the source.
Baseline: `pyright src/pylibstats` reports 0 errors.

**C++ binding layer** (`_core.cpp`, `_common.h`): its own cppcheck
invocation, `scripts/lint-cpp.sh` — not a copy of libstats' own CI
cppcheck, because (a) `_common.h` is a header and needs `--language=c++`
explicit or cppcheck misparses it as C, and (b) libstats' own cppcheck is
informational only (`--error-exitcode=0`, "not enforced yet"); this script
enforces (`--error-exitcode=1`) since it's new tooling for this repo, not
inherited from a stricter parent convention.
```bash
bash scripts/lint-cpp.sh
```

## CI / Validation

Fleet-wide workflow rules (runner budget, bounded parallelism, ISA hazards on
hosted runners, action pinning, wheel builds):
[CI House Style](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md).

Three workflows:

- `ci.yml` — `build-and-test` over an explicit include-list, not a
  cross-product: all three Python versions on Linux, oldest+newest only on
  macOS and Windows (the version axis rarely diverges by OS for a nanobind
  extension, and those runners cost 10x/2x Linux minutes). Plus a combined
  ASan+UBSan job on Linux, and a `lint` job (ruff check + format check,
  `scripts/lint-cpp.sh` with libstats headers cloned at the pinned
  `GIT_TAG` parsed from `CMakeLists.txt`).
- `wheels.yml` — cibuildwheel (pinned), gated on `v*` tags and
  `workflow_dispatch`, never on PRs. Follows the fleet wheel contract
  ([CI House Style §9](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md#9-wheel-builds-pylibhmm-pylibstats)),
  settled here at v0.5.0: `CIBW_BUILD` is an allowlist *defined* as the
  interpreters `ci.yml` tests — the `CIBW_SKIP` denylist it replaced
  failed open twice in the v0.5.0 tag run (`cp314-*` missing the
  free-threaded `cp314t-` prefix, and nothing naming a cp315 that did
  not exist when the line was written). `requires-python` moves in the
  same change as the built set. The cp312 wheel is Stable ABI, and
  `wheel.py-api` in `pyproject.toml` plus `SKBUILD_SABI_COMPONENT` in
  CMake are one mechanism in two files — set both, or the result is an
  abi3-tagged, version-locked wheel that cibuildwheel is structurally
  unable to catch. `musllinux` stays a `CIBW_SKIP` entry (an ABI axis
  orthogonal to the interpreter set, applied after BUILD).
- `lint-workflows.yml` — actionlint + zizmor (`--min-severity medium`), on
  workflow-file changes only.

The monthly canary (`schedule`) also runs `pin-currency`, which compares the
`find_package(libstats ...)` floor and the `GIT_TAG` pin in `CMakeLists.txt`
against libstats' newest release. Buildability and currency are different
questions: every run answers the first, only the canary answers the second.
That job is why the pin value is **not** restated in prose anywhere — the
check reads `CMakeLists.txt`, so `CMakeLists.txt` is the single source of
truth.

## Open Items
See PLAN.md for current status, in-progress work, and open questions.
