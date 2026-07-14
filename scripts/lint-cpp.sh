#!/usr/bin/env bash
# scripts/lint-cpp.sh
# Static analysis for the C++ nanobind binding layer (src/pylibstats/_core.cpp,
# src/pylibstats/_common.h): cppcheck warning/style/performance/portability checks.
#
# Not a copy of libstats' own CI cppcheck invocation, for two reasons:
#   1. libstats' invocation relies on file-extension inference for C++ (it only
#      scans include/src, mostly .cpp with headers reached via -I). _common.h
#      is a header passed directly; without --language=c++ explicit, cppcheck
#      misparses it as C and raises a false `nb::capsule` syntaxError.
#   2. libstats' own cppcheck is informational only (--error-exitcode=0,
#      "configured but not enforced yet" per its CI comment). This script
#      enforces (--error-exitcode=1) since it's new tooling for this repo's
#      binding layer, not inherited from a stricter parent convention.
#
# No third-party-header suppression is needed here (verified 2026-07-14):
# unlike pylibhmm, no findings were attributed to libstats' own headers.
#
# Usage:
#   ./scripts/lint-cpp.sh
#
# Prerequisites:
#   - cppcheck on PATH
#   - a local ../libstats checkout (for its headers), or set LIBSTATS_INCLUDE
#     to an alternate include directory

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBSTATS_INCLUDE="${LIBSTATS_INCLUDE:-$REPO_ROOT/../libstats/include}"

if [[ ! -d "$LIBSTATS_INCLUDE" ]]; then
    echo "ERROR: $LIBSTATS_INCLUDE not found — set LIBSTATS_INCLUDE or place a"
    echo "       local ../libstats checkout alongside this repo."
    exit 1
fi

echo ""
echo "==> Running cppcheck (src/pylibstats/_core.cpp, src/pylibstats/_common.h)..."

cppcheck --enable=warning,style,performance,portability --error-exitcode=1 \
    --suppress=missingIncludeSystem --inline-suppr \
    --std=c++20 --language=c++ \
    -I "$LIBSTATS_INCLUDE" -I "$REPO_ROOT/src/pylibstats" \
    "$REPO_ROOT/src/pylibstats/_core.cpp" "$REPO_ROOT/src/pylibstats/_common.h"

echo ""
echo "==> cppcheck clean."
