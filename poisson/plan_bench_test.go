package poisson_test

import (
	"testing"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func BenchmarkPlanSolve2D_Dirichlet(b *testing.B) {
	nx, ny := 128, 128
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
	)
	if err != nil {
		b.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, nx*ny)
	for i := range rhs {
		rhs[i] = float64(i % 7)
	}
	dst := make([]float64, nx*ny)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := plan.Solve(dst, rhs); err != nil {
			b.Fatalf("Solve failed: %v", err)
		}
	}
}
