//go:build !race

package poisson_test

import (
	"testing"

	"github.com/CWBudde/algo-pde/poisson"
)

// TestVariableCoeff_AllocFree pins the steady-state allocation behaviour of the
// PCG solve. The wrapper itself must add no allocations: after warm-up, every
// Solve draws its work vectors from the resident pool and calls the inner
// spectral solve once per iteration. The inner spectral Plan.Solve is not itself
// zero-alloc (see shape_alloc_test), so the honest bound is "no more than the
// inner solves cost, plus a tiny constant" — a growing wrapper (e.g. a per-call
// scratch allocation) would blow past it. Gated on !race because the race
// detector inflates allocation counts.
func TestVariableCoeff_AllocFree(t *testing.T) {
	const nx, ny = 24, 20
	size := nx * ny
	n := []int{nx, ny}
	h := []float64{0.1, 0.12}
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Neumann} // no nullspace

	a := positiveCoeff(4, size)
	plan, err := poisson.NewVariableCoeffPlan(2, n, h, bcs, a, poisson.WithParallelism(1))
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}

	// A standalone spectral plan with the same shape/BCs is what each PCG
	// iteration invokes as the preconditioner; measure its per-solve allocations.
	inner, err := poisson.NewPlan(2, n, h, bcs, poisson.WithWorkers(1))
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}

	rhs := randomField(1, size)
	dst := make([]float64, size)
	tmp := make([]float64, size)

	// Warm both pools.
	if _, err := plan.Solve(dst, rhs); err != nil {
		t.Fatalf("warm-up Solve: %v", err)
	}
	if err := inner.Solve(tmp, rhs); err != nil {
		t.Fatalf("warm-up inner Solve: %v", err)
	}

	var innerErr error
	innerAllocs := testing.AllocsPerRun(20, func() { innerErr = inner.Solve(tmp, rhs) })
	if innerErr != nil {
		t.Fatalf("inner Solve: %v", innerErr)
	}

	stats, err := plan.Solve(dst, rhs)
	if err != nil {
		t.Fatalf("Solve: %v", err)
	}
	iters := stats.Iterations

	var solveErr error
	outerAllocs := testing.AllocsPerRun(20, func() { _, solveErr = plan.Solve(dst, rhs) })
	if solveErr != nil {
		t.Fatalf("Solve: %v", solveErr)
	}

	// The outer solve performs at most iters+1 inner solves; anything beyond
	// (iters+1)*innerAllocs is allocation the PCG wrapper introduced itself.
	budget := float64(iters+1)*innerAllocs + 8
	t.Logf("outer=%.0f allocs, inner=%.0f allocs/solve, iters=%d, budget=%.0f",
		outerAllocs, innerAllocs, iters, budget)
	if outerAllocs > budget {
		t.Fatalf("PCG wrapper allocates beyond the inner solves: outer=%.0f > budget=%.0f",
			outerAllocs, budget)
	}
}
