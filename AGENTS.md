# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project purpose

`pylibstats` provides Python bindings for `libstats` via `nanobind` and `scikit-build-core`.

Core goals:

- expose the `libstats` distribution API in Python
- keep NumPy-based batch paths fast and simple
- keep Python wrappers, native bindings, and stubs synchronized

## Key files

- `CMakeLists.txt` — native extension build and `libstats` dependency wiring
- `pyproject.toml` — package metadata and build backend config
- `src/pylibstats/_core.cpp` — nanobind bindings
- `src/pylibstats/_common.h` — NumPy conversion helpers
- `src/pylibstats/__init__.py` — Python wrappers and validation
- `src/pylibstats/__init__.pyi`, `src/pylibstats/_core.pyi` — typing stubs
- `tests/` — pytest suite

## Dependency notes

- Build first tries `find_package(libstats 2.0.3)`.
- If not found, CMake fetches `libstats` from GitHub tag `v2.0.3`.
- For local development against a custom `libstats` install, pass `libstats_DIR` (do not override `CMAKE_PREFIX_PATH`, which can break nanobind discovery).

## Session Start Baseline Workflow (Required)

**Requires Python ≥ 3.11.** At the start of every session, do these steps in order:

1. Verify machine architecture (OS + CPU) and Python architecture.
2. Select the platform-specific build path for this host.
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

## Platform-specific build requirements

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
- If `libstats` is not found locally, CMake fetches it automatically at v2.0.1.

```bash
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -q
```

### Windows (MSVC)

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). See libstats `AGENTS.md` for full path alternatives and auto-detection via `vswhere.exe`.

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

## Common commands

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

## Editing rules

1. Keep `_core.cpp`, `__init__.py`, and `.pyi` stubs consistent.
2. Add or update tests for any behavior/API change.
3. Keep docs concise and accurate.
