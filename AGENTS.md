# Repository Guidelines

## Project Structure & Module Organization

- `doc.go` defines the top-level package docs and architecture overview (plan-based API: create a plan once, call `Solve` repeatedly).
- Core packages live at the repo root: `poisson/` (Poisson/Helmholtz solvers), `r2r/` (real-to-real transforms), `grid/` (shape/stride utilities), `fd/` (finite-difference operators).
- `cmd/acoustics-wasm/` is the WASM entry point; `demo/` is its Vite/TypeScript web frontend (with a Web Worker driving the simulation).
- `examples/` contains small runnable programs, one per boundary-condition/solver scenario.
- Tests are alongside code as `*_test.go` (e.g., `poisson/`, `r2r/`, `grid/`, `fd/`).
- Tooling/config: `justfile` (task runner), `treefmt.toml` (formatters), `go.mod`/`go.sum` (deps).
- FFT kernels come from `github.com/cwbudde/algo-fft`.

## Build, Test, and Development Commands

Use `just` targets or Go tooling directly:

- `just test` / `go test ./...`: run all unit tests.
- `just test-v`: verbose tests.
- `just test-race`: race detector.
- `just test-cov`: generate `coverage.html`.
- `just bench` / `just bench-pkg pkg=poisson`: run benchmarks.
- `just build`: compile all packages.
- `just lint` / `just lint-fix`: run `golangci-lint` (with fixes).
- `just fmt` / `just fmt-check`: run `treefmt` (gofumpt + gci + prettier).
- `just fix`: lint-fix + fmt in one step.
- `just check`: lint + test + build (run before committing).
- `just wasm`: build `demo/public/acoustics.wasm`; `just demo-dev` / `just demo-build`: run/build the web demo (requires npm).
- `just profile-cpu pkg bench` / `just profile-mem pkg bench`: profile a benchmark via pprof.

## Implementing New Features

- `PLAN.md` is a roadmap of **pending** work only (reset July 2026; completed history lives in git). Work through its phases in priority order.
- If a feature is not in `PLAN.md`, add a new section describing it before implementing.
- Write tests alongside new code in `*_test.go` files.
- Once completed, tick the feature's checkboxes in `PLAN.md`.

## Coding Style & Naming Conventions

- Go formatting uses `gofumpt` and import ordering via `gci` (see `treefmt.toml`).
- Keep identifiers idiomatic Go: `MixedBoundary` types, `NewPlan*` constructors, `Solve`/`SolveInPlace` methods.
- Prefer package-level tests named `TestXxx` and benchmarks `BenchmarkXxx`.

## Testing Guidelines

- Primary framework is Go’s standard `testing` package (`go test`).
- Add tests next to the code they cover; keep file names descriptive, e.g., `eigenvalues_test.go`.
- Run `just test-race` for concurrency-sensitive changes.

## Commit & Pull Request Guidelines

- Git history uses Conventional Commits (`feat:`). Follow that pattern when possible.
- PRs should include: a concise description, linked issue (if any), test command/results, and benchmark notes for performance changes.

## Security & Configuration Notes

- No runtime secrets or external services are configured. Keep APIs deterministic and avoid network calls in tests.

## Releasing, and Not Drifting

This module is part of the `github.com/cwbudde/algo-*` family, which is co-developed
across separate repositories. That arrangement failed once already, and the rules below
exist to stop it failing the same way twice.

**What went wrong (August 2026).** The family had drifted onto three different `algo-fft`
versions simultaneously — `algo-pde` on v0.6.15, `algo-dsp` on v0.7.3, `algo-acoustics` on
v0.6.11 — while `algo-fft`'s own `main` sat 97 commits past its latest tag and its
CHANGELOG documented a `v0.7.5` that had never been tagged. Because `algo-fft`'s generic
`PlanReal2D`/`PlanReal3D` had changed signature between the v0.6 and v0.7 lines, _no single
upgrade anywhere would compile_. Untangling it took a day and four coordinated releases.

Three separate mistakes combined to produce that. Each now has a check.

### 1. Do not let work pile up untagged

Work that only exists on `main` cannot be consumed. If you finish something a sibling repo
needs, tag it — do not wait for a milestone.

```bash
just check-unreleased     # how much is sitting past the latest tag?
```

A scheduled CI job (`.github/workflows/dep-drift.yml`) reports this weekly.

### 2. Do not sit on stale siblings

```bash
just check-deps           # are all github.com/cwbudde/* deps at their latest tags?
```

This is wired into the repo's aggregate check recipe, and the same scheduled job files a
GitHub issue when it starts failing. If a bump is _deliberately_ deferred, write down why in
`PLAN.md` — an undocumented old pin is indistinguishable from an forgotten one.

Renovate (`.github/renovate.json`) opens the bump PRs automatically and groups the whole
`cwbudde` family into a single PR on purpose: an incompatible `algo-fft` can reach a
consumer through two different dependency paths at once, so bumping them one PR at a time
produces intermediate combinations that never build.

### 3. Never remove or change exported API without the version saying so

Always release through the guard rather than by hand:

```bash
just tag-release v0.8.0       # runs every precondition, then tags and pushes
```

It refuses to tag when the tree is dirty, when `HEAD` is not a pushed `main`, when the tag
already exists or does not sort after the current one, when siblings are stale, when
`CHANGELOG.md` has no section for the version, or when the exported API changed
incompatibly without the version reflecting it.

**That last rule is stricter than semver, deliberately.** Semver exempts `v0.x` — "anything
MAY change at any time" — so `gorelease` will happily approve a _patch_ bump across a
removed symbol. Every module in this family is `v0.x`, so that exemption is exactly the hole
we fell through: `KernelEightStep` was removed and `PlanReal2D` became generic, and nothing
in the version numbers said so. The guard therefore requires a **minor** bump for any
incompatible change while on `v0.x`.

When you do break API, say so in the CHANGELOG in the form a consumer needs: the old
signature, the new signature, and the call-site rewrite. "Refactored plans" does not help
anyone; `NewPlanReal2D(rows, cols)` → `NewPlanReal2D64(rows, cols)` does.

### Order of operations for a cross-repo change

Releases must flow up the dependency graph, never sideways:

```
algo-vecmath ─┐
algo-approx  ─┼─→ algo-dsp ─┐
algo-fft ─────┴─→ algo-pde ─┴─→ algo-acoustics
```

Tag the dependency first, then bump and tag its consumers, then the consumers' consumers.
Bumping a consumer before its dependency is tagged is what forces pseudo-versions into
`go.mod`, and those are how a repo quietly ends up pinned to a commit nobody can find later.
