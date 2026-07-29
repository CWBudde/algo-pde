//go:build !race

package poisson_test

import (
	"testing"

	"github.com/cwbudde/algo-pde/grid"
	"github.com/cwbudde/algo-pde/poisson"
)

// These allocation guards protect the zero-per-solve property of the unified
// slice-backed grid.Shape: a slice-backed Shape allocates on construction, so
// every plan builds its shape ONCE (at construction) and reuses it. Rebuilding
// the shape inside Solve would add one allocation per call. They are gated on
// !race because the race detector inflates allocation counts, which would make
// the tight bounds flaky under `just test-race`.

// TestPlan_SolveShapeCached pins the <=3D Plan.Solve hot path. With a single
// worker the per-solve overhead is minimal (5 allocs/op), so a per-solve shape
// rebuild is directly observable: it lifts the count to 6.
func TestPlan_SolveShapeCached(t *testing.T) {
	n := 64
	plan, err := poisson.NewPlan(2, []int{n, n},
		[]float64{1.0 / float64(n), 1.0 / float64(n)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
		poisson.WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}

	rhs := make([]float64, n*n)
	dst := make([]float64, n*n)
	for i := range rhs {
		rhs[i] = float64(i%5) - 2
	}

	if err := plan.Solve(dst, rhs); err != nil { // warm the workspace pool
		t.Fatal(err)
	}

	var solveErr error
	got := testing.AllocsPerRun(100, func() {
		solveErr = plan.Solve(dst, rhs)
	})
	if solveErr != nil {
		t.Fatalf("Solve returned error during measurement: %v", solveErr)
	}
	t.Logf("2D Dirichlet workers=1 allocs/op = %v", got)

	if got > 5 {
		t.Fatalf("Plan.Solve allocs/op = %v, want <= 5 (per-solve shape allocation regression?)", got)
	}
}

// TestPlanNDPeriodic_SolveShapeCached pins the >3D periodic Solve path. Its
// absolute count is dominated by the per-line FFT worker pool bookkeeping and
// carries headroom; a per-solve shape allocation would still be a regression
// off the recorded baseline.
func TestPlanNDPeriodic_SolveShapeCached(t *testing.T) {
	dims := grid.NewShapeND(4, 5, 6, 7)
	h := []float64{0.25, 0.2, 1.0 / 6, 1.0 / 7}
	plan, err := poisson.NewPlanNDPeriodic(dims, h, poisson.WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}

	size := dims.Size()
	rhs := make([]float64, size)
	dst := make([]float64, size)
	for i := range rhs {
		rhs[i] = float64(i%7) - 3
	}
	mean := 0.0
	for _, v := range rhs {
		mean += v
	}
	mean /= float64(size)
	for i := range rhs {
		rhs[i] -= mean
	}

	if err := plan.Solve(dst, rhs); err != nil { // warm the pools
		t.Fatal(err)
	}

	var solveErr error
	got := testing.AllocsPerRun(50, func() {
		solveErr = plan.Solve(dst, rhs)
	})
	if solveErr != nil {
		t.Fatalf("Solve returned error during measurement: %v", solveErr)
	}
	t.Logf("ND workers=1 allocs/op = %v", got)

	if got > 640 {
		t.Fatalf("PlanNDPeriodic.Solve allocs/op = %v, want <= 640 (per-solve shape allocation regression?)", got)
	}
}
