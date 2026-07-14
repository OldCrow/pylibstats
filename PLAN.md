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
- (none currently tracked — populate as work starts). Two old local
  branches exist (`bench/issue-2-scipy-comparison`, last commit
  2026-07-02; `fix/audit-v1.5.2`, last commit 2026-06-19, predates the
  v2.0 libstats pin) but both are stale relative to `main`'s current tip
  and show no signs of active work — not treated as in-progress.

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
(fallback), currently pinned at v2.0.4 — confirmed current against
libstats' actual latest release tag as of 2026-07-14 (see Version Pin
Verification above). Check libstats' own PLAN.md/AGENTS.md for its
current release before assuming this repo's pin is still current on any
future session.

## Next Steps
- #5: Decide when to wire `ruff check` and `scripts/lint-cpp.sh` into CI.
- #7: Run the deferred `ruff format` pass as its own reviewable change.
- #6: Decide whether to adopt mypy.
- Prune or revive the two stale local branches (`bench/issue-2-scipy-comparison`,
  `fix/audit-v1.5.2`) — not filed as a GitHub issue since both branches'
  remote counterparts are already gone (`origin` shows `: gone` for both);
  this is local-only housekeeping, not collaborator-facing.
