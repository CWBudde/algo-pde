//go:build !race

package poisson_test

import (
	"testing"

	"github.com/cwbudde/algo-pde/poisson"
)

// TestSolveComplex_AllocParity confirms SolveComplex adds no per-call
// allocation over the real Solve on the same plan configuration. Race and
// atomic-coverage instrumentation inflate these counts differently, so—as with
// the other allocation-contract tests—this file is excluded under -race.
func TestSolveComplex_AllocParity(t *testing.T) {
	const (
		nx, ny = 32, 32
		hx, hy = 1.0 / 33, 1.0 / 33
		alpha  = 2.0
	)
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}
	f := randomField(7, nx*ny)

	realPlan, err := poisson.NewHelmholtzPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, alpha, poisson.WithWorkers(1))
	if err != nil {
		t.Fatalf("real plan: %v", err)
	}
	realDst := make([]float64, nx*ny)
	if err := realPlan.Solve(realDst, f); err != nil {
		t.Fatalf("warm real solve: %v", err)
	}
	realAllocs := testing.AllocsPerRun(50, func() { _ = realPlan.Solve(realDst, f) })

	cplxPlan, err := poisson.NewComplexHelmholtzPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, complex(alpha, 0), poisson.WithWorkers(1))
	if err != nil {
		t.Fatalf("complex plan: %v", err)
	}
	dst := make([]complex128, nx*ny)
	if err := cplxPlan.SolveComplex(dst, f); err != nil {
		t.Fatalf("warm complex solve: %v", err)
	}
	cplxAllocs := testing.AllocsPerRun(50, func() { _ = cplxPlan.SolveComplex(dst, f) })

	if cplxAllocs > realAllocs {
		t.Fatalf("SolveComplex allocates %v/op vs real Solve %v/op — complex path must not allocate more", cplxAllocs, realAllocs)
	}
}
