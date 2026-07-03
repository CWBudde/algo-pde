package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const dirichlet1dTol = 1e-10

func TestPlan1DDirichlet_Solve_Fundamental(t *testing.T) {
	n := 64
	h := 1.0 / float64(n+1)
	L := float64(n+1) * h

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := float64(i+1) * h
		u[i] = math.Sin(math.Pi * x / L)
	}

	rhs := make([]float64, n)
	if err := fd.Apply1D(rhs, u, h, poisson.Dirichlet); err != nil {
		t.Fatal(err)
	}

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if maxErr := maxAbsDiff(got, u); maxErr > dirichlet1dTol {
		t.Fatalf("max error %g exceeds tol %g", maxErr, dirichlet1dTol)
	}

	assertDirichletBoundaryDecay(t, got)
}

func TestPlan1DDirichlet_Solve_Combination(t *testing.T) {
	n := 96
	h := 1.0 / float64(n+1)
	L := float64(n+1) * h

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := float64(i+1) * h
		u[i] = math.Sin(math.Pi*x/L) + 0.3*math.Sin(2.0*math.Pi*x/L)
	}

	rhs := make([]float64, n)
	if err := fd.Apply1D(rhs, u, h, poisson.Dirichlet); err != nil {
		t.Fatal(err)
	}

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if maxErr := maxAbsDiff(got, u); maxErr > dirichlet1dTol {
		t.Fatalf("max error %g exceeds tol %g", maxErr, dirichlet1dTol)
	}

	assertDirichletBoundaryDecay(t, got)
}

// assertDirichletBoundaryDecay checks that the SOLVED field decays to the
// homogeneous Dirichlet boundary value (0). The nodes are vertex-centered at
// (i+1)h, so the physical boundaries at x=0 and x=(n+1)h lie one grid step
// outside got[0] and got[n-1]. Linearly extrapolating the solver output to those
// off-grid boundaries must recover ~0. This exercises the actual solution rather
// than the compile-time constant math.Sin(0)/math.Sin(math.Pi).
func assertDirichletBoundaryDecay(t *testing.T, got []float64) {
	t.Helper()
	if len(got) < 2 {
		return
	}
	n := len(got)

	// For a field vanishing at the boundary the extrapolation residual is
	// O(h²·u''(0)); for these manufactured modes u''(0)=0, so it is ~1e-4.
	const boundaryTol = 1e-2

	left := 2*got[0] - got[1]
	right := 2*got[n-1] - got[n-2]
	if math.Abs(left) > boundaryTol || math.Abs(right) > boundaryTol {
		t.Fatalf("Dirichlet boundary not ~0: left=%g right=%g (tol %g)", left, right, boundaryTol)
	}
}
