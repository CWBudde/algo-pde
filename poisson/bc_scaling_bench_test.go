package poisson_test

import (
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// benchmarkSolve2DBC measures a square 2D solve for a given per-axis boundary
// condition. Before the FFT-based DST-III/DCT-III inverses, the Neumann case
// (DCT-II inverse) was ~500-650x slower than Dirichlet/periodic at n=1024
// because its inverse was an O(N^2) double loop. These benchmarks let
// `just bench-pkg pkg=poisson` confirm the three BC types now scale
// comparably (all O(N^2 log N) per solve).
func benchmarkSolve2DBC(b *testing.B, n int, bc poisson.BCType) {
	var h float64
	switch bc {
	case poisson.Dirichlet:
		h = 1.0 / float64(n+1) // vertex-centered
	default:
		h = 1.0 / float64(n) // cell-centered / periodic
	}

	plan, err := poisson.NewPlan(
		2,
		[]int{n, n},
		[]float64{h, h},
		[]poisson.BCType{bc, bc},
	)
	if err != nil {
		b.Fatalf("NewPlan failed: %v", err)
	}

	shape := grid.NewShape2D(n, n)
	u := make([]float64, n*n)
	for i := range u {
		u[i] = float64((i*7)%13) * 0.01
	}

	rhs := make([]float64, n*n)
	fd.Apply2D(rhs, u, shape, [2]float64{h, h}, [2]poisson.BCType{bc, bc})

	dst := make([]float64, n*n)

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := plan.Solve(dst, rhs); err != nil {
			b.Fatalf("Solve failed: %v", err)
		}
	}
}

func BenchmarkSolve2D_Dirichlet_256(b *testing.B)  { benchmarkSolve2DBC(b, 256, poisson.Dirichlet) }
func BenchmarkSolve2D_Dirichlet_512(b *testing.B)  { benchmarkSolve2DBC(b, 512, poisson.Dirichlet) }
func BenchmarkSolve2D_Dirichlet_1024(b *testing.B) { benchmarkSolve2DBC(b, 1024, poisson.Dirichlet) }

func BenchmarkSolve2D_Neumann_256(b *testing.B)  { benchmarkSolve2DBC(b, 256, poisson.Neumann) }
func BenchmarkSolve2D_Neumann_512(b *testing.B)  { benchmarkSolve2DBC(b, 512, poisson.Neumann) }
func BenchmarkSolve2D_Neumann_1024(b *testing.B) { benchmarkSolve2DBC(b, 1024, poisson.Neumann) }

func BenchmarkSolve2D_Periodic_256(b *testing.B)  { benchmarkSolve2DBC(b, 256, poisson.Periodic) }
func BenchmarkSolve2D_Periodic_512(b *testing.B)  { benchmarkSolve2DBC(b, 512, poisson.Periodic) }
func BenchmarkSolve2D_Periodic_1024(b *testing.B) { benchmarkSolve2DBC(b, 1024, poisson.Periodic) }
