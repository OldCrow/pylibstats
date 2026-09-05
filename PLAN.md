# pylibstats — Plan / Status

## Decided [DERIVED]
- Bindings via nanobind + scikit-build-core.
- Dependency resolution: `find_package` first, `FetchContent` fallback,
  `libstats_DIR` override for local development (no implicit
  sibling-directory preference, unlike pylibhmm's `../libhmm` behavior —
  this is a deliberate difference; do not "fix" pylibstats to match
  pylibhmm's pattern without a specific reason).
- NumPy ⇔ libstats conversion: batch input is genuinely zero-copy
  (`std::span` constructed directly over the NumPy buffer, matching
  libstats' span-based batch API); batch output is a fresh heap
  allocation transferred to Python via `nb::capsule`. Two exceptions:
  `fit()` copies into an owned `std::vector<double>` (libstats' `fit()`
  takes a vector, not a span); batch `ppf` loops scalar-wise since
  libstats has no batch quantile method. See AGENTS.md Architecture.
- `__init__.pyi` / `_core.pyi` are hand-written, not tool-generated.
- pyright is the editor/agent type checker (2026-09-01): config in
  `pyproject.toml` `[tool.pyright]`, venv-aware, not run in CI. `_core.pyi`
  declares the `def_prop_rw` parameter properties it previously omitted,
  since `__init__.py` reads them off the `_core` classes; `__init__.py`
  disables `reportIncompatibleMethodOverride` file-wide because pyright
  rejects assigned-property overrides regardless of type.
- Python-level parameter validation in `__init__.py` is retained for UX
  consistency (clean `ValueError`s) only — its original v1.x motivation
  (Homebrew LLVM vs AppleClang ABI mismatch) stopped applying in v2.0,
  per the module's own comment.
- Python tooling: ruff adopted (`E`/`F`/`I`/`UP`/`B`), config in
  `pyproject.toml`. All five categories are clean here — no deferral
  needed (contrast with pylibhmm, where `B` was deferred for 5
  blind-exception test findings). `ruff format` applied repo-wide
  2026-09-02 (16 files, closed #7); check + format --check wired into
  CI the same day (closed #5), ruff pinned 0.16.5 there.
- C++ binding tooling: `scripts/lint-cpp.sh` — own cppcheck invocation
  (not a copy of libstats' CI cppcheck), requiring `--language=c++` for
  `_common.h`. Enforced (`--error-exitcode=1`) even though libstats' own
  cppcheck is informational-only, since this is new tooling for this
  repo, not inherited from a stricter parent. Verified clean 2026-07-14.
  Wired into CI 2026-09-02 (lint job clones libstats headers at the
  GIT_TAG parsed from CMakeLists.txt). Two suppressions added the same
  day: libstats-header findings are suppressed by a `libstats/`
  path-segment glob (18 appeared since the July baseline — pin and
  cppcheck both moved; all libstats' concern, none in the binding
  layer; an absolute-path glob fails on Windows drive-letter colons),
  and one inline memleak suppression in `_common.h` `vec_to_numpy`
  (Ubuntu 24.04's older cppcheck cannot see the nb::capsule ownership
  transfer; local 2.21 doesn't flag it).

## Version Pin Verification (Task 1, 2026-07-14) [DERIVED]
The brief for this session assumed a three-way inconsistency (AGENTS.md
"Dependency notes" at 2.0.3, AGENTS.md Linux section at v2.0.1, actual
current libstats release v2.0.4). On inspection, that inconsistency did
not exist in AGENTS.md — a prior session (commit `3d675ca`, "docs: align
AGENTS.md structure, update libstats version pin, add CLAUDE.md import
shim") had already bumped both AGENTS.md mentions and `CMakeLists.txt`
(`find_package(libstats 2.0.4)`, `FetchContent` `GIT_TAG v2.0.4`) to
v2.0.4, matching libstats' actual current release exactly — confirmed by
cross-checking libstats' own tags directly, not by trusting either
repo's prose. One genuine stale mention was found and fixed this
session: `README.md` still said "v2.0.3" in the FetchContent build
instructions; corrected to v2.0.4. No version-pin gap remains to defer.

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-09-02.
- GitHub is the collaborator-facing source for issues and milestones; this
  PLAN.md is the agent-facing durable project state. Keep both in sync.
- When creating, closing, reopening, retitling, or moving a GitHub issue or
  milestone, update this section in the same change set or note why it could
  not be updated.
- Reconcile this section against live GitHub state when either is true:
  (a) the task at hand involves reading the backlog to decide what to work
  on next, or creating/closing/retitling/moving an issue or milestone, or
  (b) more than 7 days have passed since the "Last reconciled" date above.
  Skip the check for tasks that don't touch the backlog or this file at
  all — a per-session or per-task refresh regardless of relevance is
  wasted effort in one direction and a rubber stamp in the other. Update
  the "Last reconciled" date whenever this section is actually re-checked,
  whether or not anything had drifted.
- Convention: open (actionable) milestones/issues are fully itemized here;
  closed/historical ones are summarized as counts only.

## GitHub Milestones [DERIVED]
- None currently exist in this repository (checked 2026-07-14).

## GitHub Issues Without Milestone [DERIVED]
- Open issues:
  - #6 Decide whether to adopt mypy for the Python surface (filed 2026-07-14)
- Closed issues: 4 as of 2026-09-02 (#5 and #7 closed 2026-09-02 by the
  catch-up commits `da098f4`/`d15687a`; CI green at `3b211c9`).

## In Progress [OPEN]
- **0.7.0 SHIPPED 2026-09-05**: PR #18 squash-merged (`6ac8233`,
  user-merged in the UI; full 13-job matrix green incl. 3.14t and
  ASan/UBSan) — libstats pin v2.3.1 → v2.4.0 plus bindings for the
  v2.4.0 eight (19 → 27; suite 424 → 632). Signed tag v0.7.0 pushed
  [user-approved — tag push IS the PyPI trigger]; wheels green on all
  five targets; PyPI 0.7.0 live (16 files incl. sdist); GitHub release
  published. Parameterization findings recorded in the tag message and
  PR #18: Erlang `lam` is a RATE, InverseGamma `beta` is a SCALE
  (probed vs scipy before asserting; the header's internal "RATE"
  comment refers to the Gamma delegate), TruncatedNormal bounds are
  ABSOLUTE (scipy truncnorm standardizes).

## Known Gaps [OPEN]
- [2026-08-16, resolved same day] The pylibhmm `wheels.yml` denylist defect
  flagged here was closed by pylibhmm v0.10.0 (`c48008c`): `CIBW_BUILD`
  allowlist, abi3 pairing completed, cibuildwheel pinned.
- [2026-08-16, resolved same day] The missing 3.14t row in `ci.yml` flagged
  here was closed by `73f7f49`: Linux-only free-threaded row matching
  pylibhmm v0.10.0, CI green including the new `3.14t` job.
- mypy is not adopted for the Python surface — not evaluated this
  session (ruff covers lint/format; typing strictness is a separate,
  undecided question). Tracked as issue #6.
- [resolved 2026-09-02] CI lint wiring (#5) and the deferred ruff
  format pass (#7) both landed — see Decided and Next Steps.

## Cross-Repo Dependencies [OPEN]
Depends on libstats via `find_package` (preferred) or `FetchContent`
(fallback). **The pinned version is deliberately not restated here.**
`CMakeLists.txt` holds the only two copies that matter — the
`find_package` version floor and the `FetchContent` `GIT_TAG` — and both
must agree with each other and with libstats' newest release.

This repo has already been burned by a restated copy: the 2026-07-14
audit (see "Version Pin Verification" above) found `README.md` still
advertising v2.0.3 while the actual pin and release were v2.0.4. It took
a manual cross-check against libstats' tags to catch, and it was fixed by
re-syncing the copy — the approach that guarantees recurrence. The copies
are now removed instead.

Currency and internal consistency are enforced mechanically by the
`pin-currency` job in the monthly CI canary (also runnable on demand via
`workflow_dispatch`): it fails if the floor and the tag disagree, or if
the tag has fallen behind libstats' newest release. Bumping remains a
deliberate act — move the floor and the tag together.

[2026-08-16] Bumped to libstats **v2.2.0** (floor and tag together), and
pylibstats goes to **0.5.0** rather than 0.4.1 in the same change. The
minor bump is the point: libstats #97 means every consumer of the
installed v2.1.0 package — this one included — was silently compiling the
Tier 2 Bessel fallback, so `bessel_i0(10)` carried ~1.3e-08 relative error
where the library itself measures 1.5e-16. Wheel output changes by eight
orders of magnitude on the VonMises paths that reach it. That is a
behaviour change users can observe, not packaging bookkeeping, and a patch
number would have understated it exactly the way 2.1.1 would have
understated the libstats release it tracks.

Note the pin bump also raises this package's effective CMake floor to
libstats' new 3.20 → 3.25 minimum. No action needed: pylibstats already
declares `cmake_minimum_required(VERSION 3.25)`, checked before bumping.

The first v0.5.0 tag push failed to publish, and **not because of the pin**
— the extension built against libstats v2.2.0 on all four platforms. The
wheels workflow was building six interpreters where it intended three; cp315
has no scipy wheel, so `CIBW_TEST_REQUIRES` fell back to a source build and
died on missing OpenBLAS. `publish` needs both wheel jobs, so nothing reached
PyPI and the version number was never burned. See "Wheel targets" below for
what the investigation turned up and what shipped instead.

[2026-08-22] Bumped to libstats **v2.3.0** (floor and tag together), and
pylibstats goes to **0.6.0** rather than 0.5.1 — **released 2026-08-22**
(tag v0.6.0, wheels + CI + pin-currency green, PyPI 0.6.0, GitHub
release) — on the 0.5.0 reasoning:
libstats v2.3.0 changes numbers users observe — LogNormal and Gaussian CDF
lower tails no longer collapse to 0 (libstats #49 pattern), von Mises CDF
via the Bessel series (#51), the closed-form Cauchy CDF (#48), and
clean-room SIMD cos/sin at every x86 tier (#95). No binding-surface change.
libstats v2.3.1 (the correctness patch carrying #105, LogNormal batch
cdf(NaN) = 1) is imminent; re-bump when it is cut. The bump was verified
locally through the FetchContent path (no libstats installed here).

[2026-08-25] Bumped to libstats **v2.3.1** (floor and tag together,
8ef6a2b), and pylibstats goes to **0.6.1** — a patch, since libstats
v2.3.1 is itself a correctness patch with no API change. Numbers users
observe change for the better: batch pdf/logpdf/cdf(NaN) now return NaN
on every SIMD tier (libstats #105/#102 — 0.6.0 returned finite garbage,
e.g. LogNormal batch cdf(NaN) = 1), the von Mises CDF is correct across
the ±π seam for κ > 1000 (#106), and NegBin/Geometric quantiles work past
INT_MAX (#116). No binding-surface change. Verified locally on Windows
through the FetchContent path: 424/424 pytest against the fetched v2.3.1
tag. Released 2026-08-26 (tag v0.6.1 → wheels.yml all green — sdist +
linux x86_64/aarch64 + macos + windows — → PyPI 0.6.1 via trusted
publishing, 16 artifacts; GitHub release).

## Wheel targets and the Stable ABI (2026-08-16) [DERIVED]

**3.14 has been shipping since 0.1.5 and was never deliberately withdrawn.**
Every published release, 0.1.5 (2026-04-26) through 0.4.0, carries 25 wheels
including full `cp314-cp314` and `cp314-cp314t` sets. `CIBW_SKIP` gained a
`cp314-*` entry in `fa6ea59`, four hours *after* the v0.4.0 tag — and no tag
has been cut since, so the intent to stop never reached PyPI. Verified against
the PyPI JSON API, not inferred from the workflow.

Dropping a target that already ships is worse than it looks. `requires-python`
is `>=3.11`, which still admits those users, so pip would offer them the new
version's **sdist** and attempt to compile libstats on their machine — an
error, where being left on the previous release would merely have been stale.
That is why 3.11 and both 3.14 variants stay in the allowlist. cp315 is out
until scipy publishes a wheel for it; the blocker is `CIBW_TEST_REQUIRES`, not
anything this package builds.

**Adopted the Stable ABI**, so 3.12/3.13/3.14 converge on one `cp312-abi3`
wheel: 25 wheels per release becomes 3 distinct wheels per arch
(`cp311-cp311`, `cp312-abi3`, `cp314-cp314t`), and 3.15+ needs no workflow
change once its ecosystem catches up. `nanobind_add_module` had carried
`STABLE_ABI` since the beginning with no effect whatever.

**The trap, because it is silent and would ship broken wheels.**
`wheel.py-api` only TAGS a wheel abi3; it does not check that one was built.
nanobind gates `STABLE_ABI` on `TARGET Python::SABIModule`, which exists only
if `find_package(Python)` was given `${SKBUILD_SABI_COMPONENT}`. Set the
pyproject key without the CMake argument and you get a wheel *tagged*
`cp312-abi3` around a *version-locked* binary — installs on 3.13+, then fails
to import. Nothing warns at any layer. Proven with `dumpbin` on two wheels
with byte-identical filenames:

| | module inside | links |
|---|---|---|
| `wheel.py-api` alone | `_core.cp312-win_amd64.pyd` | `python312.dll` |
| plus `${SKBUILD_SABI_COMPONENT}` | `_core.pyd` | `python3.dll` |

CI would not have caught it: cibuildwheel tests each wheel on the interpreter
that built it, which is the one version where the broken form works. Hence the
allowlist keeps listing cp313 and cp314 even though they produce no additional
wheel — installing the abi3 wheel on an interpreter it was not built with is
the only check that closes this class.

scikit-build-core disables limited-API by itself where it cannot apply — on
3.11 (`target_minor <= sys.version_info.minor` is false), on PyPy, and on
free-threaded builds — matching nanobind's own gate, so no per-version
configuration is needed.

Published as a fleet standard 2026-08-16:
[CI House Style §9](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md#9-wheel-builds-pylibhmm-pylibstats),
merged with pylibhmm's twin incidents. AGENTS.md's `wheels.yml` bullet was
rewritten the same day (it still described the retired `CIBW_SKIP`
upper-bound rule); this section stays as the incident record behind the
rules.

## Next Steps
Bindings catch-up track, decided 2026-09-02 (user delegated the
catch-up-vs-widen call; catch-up chosen — the C++ fleet is at a natural
pause after corvus v1.0.0, and clearing the tooling backlog now keeps
the adoption-era sessions to pin bumps only). Format before CI wiring so
a format check can go in green:
1. ~~#7: ruff format pass~~ — DONE 2026-09-02 (`da098f4`, 424/424 green).
2. ~~#5: CI lint wiring~~ — DONE 2026-09-02 (`d15687a` + suppression fix
   `3b211c9`; full matrix + lint green).
- DEFERRED past the adoption round: #6 mypy (a decision + annotation
  question; no drift cost to waiting).
- ~~The libstats v2.4.0 catch-up~~ **DONE 2026-09-04** (PR #18 /
  0.7.0, see In Progress): the eight new distributions are bound, so
  the v2.5.0 adoption-era session is back to a pin bump only. That
  bump stays MINOR, not patch — the corvus swap changes numbers users
  observe (the v2.2.0/0.5.0 Bessel precedent), so the release notes
  carry the behavior change. v2.6.0's new distributions will need the
  same bindings + hand-written stubs treatment when they ship.

## Build-Stack Standardization (2026-07-23) [DERIVED]
Cross-repo effort tracked in the fleet standards repo
([record](https://github.com/OldCrow/standards/blob/main/records/BUILD-STANDARDIZATION-PLAN.md)).
Commit `870877d` (minimal CMakePresets.json, CMake minimum bumped to 3.25)
was the only change this repo received — no Phase 3 work touched it (it
consumes libstats via `find_package`/pinned `FetchContent`, not option
names). AGENTS.md's CMake-standard section checked post-Phase-3 and is
still accurate.

## Local Branch Cleanup (2026-07-14) [DERIVED]
The two stale local branches (`bench/issue-2-scipy-comparison`,
`fix/audit-v1.5.2`) were confirmed via GitHub PR history to be
squash-merged already — PR #3 (merged 2026-07-04) and PR #1 (merged
2026-06-19) respectively. Squash merges diverge from the source
branch's commit hash, so `git branch --merged main` didn't recognize
them and they survived local cleanup after their remote counterparts
were deleted post-merge. Deleted locally with `git branch -D`; no
GitHub issue filed since this was local-only housekeeping with no
collaborator-facing artifact.
