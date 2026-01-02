# algo-pde

Fast spectral Poisson and Helmholtz solvers for Go, built on top of `algo-fft`. The library uses plan-based APIs (like FFTW) to precompute eigenvalues and reuse transform plans for many solves on the same grid.

## Features
- O(N log N) solvers for Poisson and Helmholtz on 1D/2D/3D regular grids.
- Boundary conditions per axis: Periodic, Dirichlet, Neumann, and mixed.
- Real-to-real transforms (DST/DCT) implemented via FFT for physical boundaries.
- Nullspace handling options for periodic/Neumann problems.
- Zero-allocation solve path with reusable plans and work buffers.

## Install

```bash
go get github.com/MeKo-Tech/algo-pde
```

## Quick Start

```go
package main

import (
	"log"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func main() {
	// 2D periodic Poisson solve on a 128x128 grid.
	plan, err := poisson.NewPlan2D(128, 128, 1.0/128, 1.0/128, poisson.Periodic, poisson.Periodic)
	if err != nil {
		log.Fatal(err)
	}

	rhs := make([]float64, 128*128)
	sol := make([]float64, 128*128)
	// fill rhs...

	if err := plan.Solve(sol, rhs); err != nil {
		log.Fatal(err)
	}
}
```

## Helmholtz Example

```go
plan, err := poisson.NewHelmholtzPlan2D(
	128, 128,
	1.0/128, 1.0/128,
	poisson.Dirichlet, poisson.Dirichlet,
	poisson.WithAlpha(1.5),
)
```

## Package Layout
- `poisson/`: Poisson/Helmholtz solvers, plans, boundary handling.
- `r2r/`: DST/DCT transforms and plans.
- `grid/`: Shape, stride, indexing utilities.
- `fd/`: Finite-difference eigenvalues and validation helpers.
- `examples/`: End-to-end examples (periodic, Dirichlet, Neumann, mixed, Helmholtz).

## Usage Notes
- Reuse plans when solving multiple RHS on the same grid.
- Periodic/Neumann problems have a nullspace; configure handling via options such as `WithNullspace` or `WithSubtractMean`.
- Data layout is row-major for 2D/3D, stored in flat `[]float64` slices.

## Development

Common tasks use `just` (or run the Go commands directly):

```bash
just test       # go test ./...
just test-race  # go test -race ./...
just bench      # go test -bench=. -benchmem ./...
just lint       # golangci-lint run
just fmt        # treefmt (gofumpt + gci + prettier)
```

## Performance
- Expected complexity: O(N log N).
- Plans precompute eigenvalues and buffers to avoid per-solve allocations.
- Benchmarks live alongside packages and can be run with `just bench`.

## License

TBD.
