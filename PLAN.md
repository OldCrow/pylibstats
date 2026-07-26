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
- Python-level parameter validation in `__init__.py` is retained for UX
  consistency (clean `ValueError`s) only — its original v1.x motivation
  (Homebrew LLVM vs AppleClang ABI mismatch) stopped applying in v2.0,
  per the module's own comment.
- Python tooling: ruff adopted (`E`/`F`/`I`/`UP`/`B`), config in
  `pyproject.toml`. All five categories are clean here — no deferral
  needed (contrast with pylibhmm, where `B` was deferred for 5
  blind-exception test findings).
- C++ binding tooling: `scripts/lint-cpp.sh` — own cppcheck invocation
  (not a copy of libstats' CI cppcheck), requiring `--language=c++` for
  `_common.h`. Enforced (`--error-exitcode=1`) even though libstats' own
  cppcheck is informational-only, since this is new tooling for this
  repo, not inherited from a stricter parent. Verified clean 2026-07-14.

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
Last reconciled against live GitHub state: 2026-07-14.
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
- Open issues (filed 2026-07-14, from this session's Known Gaps/Next Steps):
  - #5 Wire ruff check and lint-cpp.sh into CI
  - #6 Decide whether to adopt mypy for the Python surface
  - #7 Run deferred ruff format pass across src/pylibstats, tests, examples
- Closed issues: 1 as of 2026-07-14.

## In Progress [OPEN]
- (none currently tracked — populate as work starts).

## Known Gaps [OPEN]
- mypy is not adopted for the Python surface — not evaluated this
  session (ruff covers lint/format; typing strictness is a separate,
  undecided question). Tracked as issue #6.
- Neither `ruff check` nor `scripts/lint-cpp.sh` are wired into CI yet
  (`ci.yml`, `wheels.yml` only build and test). Tracked as issue #5.
- `ruff format` would reformat 16 files under the new config — not
  applied in this pass since it's a large, purely cosmetic diff that
  deserves its own visible change. Tracked as issue #7.

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

## Next Steps
- #5: Decide when to wire `ruff check` and `scripts/lint-cpp.sh` into CI.
- #7: Run the deferred `ruff format` pass as its own reviewable change.
- #6: Decide whether to adopt mypy.

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
