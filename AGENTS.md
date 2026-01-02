# Repository Guidelines

## Project Structure & Module Organization

- `doc.go` defines the top-level package docs and architecture overview.
- Core packages live at the repo root: `poisson/` (solvers), `r2r/` (real-to-real transforms), `grid/` (shape/stride utilities), `fd/` (finite-difference operators).
- Tests are alongside code as `*_test.go` (e.g., `poisson/`, `r2r/`, `grid/`, `fd/`).
- Tooling/config: `justfile` (task runner), `treefmt.toml` (formatters), `go.mod`/`go.sum` (deps).

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
