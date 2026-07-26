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

**Requires Python ≥ 3.11.** At the start of every session, do these steps in order:

1. Verify machine architecture (OS + CPU) and Python architecture.
2. Select the platform-specific build path for this host (see Platform-Specific Notes).
3. Build/install/test only after the architecture check is complete.

Architecture checks:

```bash
# macOS/Linux shells
uname -m
uname -s
python -c "import platform, struct; print(platform.system(), platform.machine(), struct.calcsize('P')*8)"
```

```powershell
# Windows PowerShell
[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
[System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
python -c "import platform, struct; print(platform.system(), platform.machine(), struct.calcsize('P')*8)"
```

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

- Visual Studio 2022 (Build Tools or full IDE) is required as the C++ compiler. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`.
- Use the VS 2022 x64 generator for reproducible MSVC builds (`-Ccmake.define.CMAKE_GENERATOR="Visual Studio 17 2022"`).
- If using `libstats_DIR`, the referenced `libstats` install must be built with compatible MSVC/x64 settings.

```powershell
python -m pip install -e ".[test]" `
  -Ccmake.define.CMAKE_GENERATOR="Visual Studio 17 2022" `
  -Ccmake.define.CMAKE_GENERATOR_PLATFORM=x64 `
  -Ccmake.build-type=Release
python -m pytest tests -q
```

#### Windows toolchain setup

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). The paths below are common defaults — adjust for your installation. VS Build Tools and full VS editions use different default directories.

Activate the MSVC toolchain once per PowerShell session before building:

```powershell
# Default path for VS 2022 Build Tools. For full VS (Community/Professional/Enterprise),
# replace "BuildTools" with your edition under "C:\Program Files\Microsoft Visual Studio\2022\".
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# Auto-detect any edition instead:
# $vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
# $vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
$envVars = cmd /c "`"$vcvars`" > nul && set"
foreach ($line in $envVars) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
```

**One-time setup:**
- Visual Studio 2022 Build Tools (not full IDE) is sufficient. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`.
  - Build Tools default path: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\`
  - Full VS default path: `C:\Program Files\Microsoft Visual Studio\2022\{edition}\`
- **Smart App Control must be Off** (Windows Security → App & Browser Control → SAC settings). SAC blocks locally compiled executables and cannot be re-enabled without a Windows reset.
- CMake ≥ 3.25: https://cmake.org/download/, `winget install Kitware.CMake`, or `choco install cmake`.

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
ruff format src/pylibstats tests examples   # not yet applied repo-wide; see PLAN.md
```

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

## Open Items
See PLAN.md for current status, in-progress work, and open questions.
