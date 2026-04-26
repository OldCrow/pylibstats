# WARP.md

Guidance for working in `pylibstats`.

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

- Build first tries `find_package(libstats)`.
- If not found, CMake fetches `libstats` from GitHub tag `v1.1.1`.
- For local development against a custom `libstats` install, pass `libstats_DIR` (do not override `CMAKE_PREFIX_PATH`, which can break nanobind discovery).

## Common commands

```powershell
python -m pip install -e C:\Users\gdwol\Development\pylibstats
python -m pip install ".[test]"
python -m pytest C:\Users\gdwol\Development\pylibstats\tests -q
```

Local `libstats` build override example:

```powershell
python -m pip install --no-build-isolation -ve C:\Users\gdwol\Development\pylibstats `
  -Ccmake.define.libstats_DIR=C:\path\to\libstats\install\lib\cmake\libstats
```

## Editing rules

1. Keep `_core.cpp`, `__init__.py`, and `.pyi` stubs consistent.
2. Add or update tests for any behavior/API change.
3. Keep docs concise and accurate.
