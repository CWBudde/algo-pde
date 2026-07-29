package poisson_test

import (
	"math"
	"testing"

	"github.com/CWBudde/algo-pde/fd"
	"github.com/CWBudde/algo-pde/grid"
	"github.com/CWBudde/algo-pde/poisson"
)

// TestWithInPlace_MatchesDefault exercises the WithInPlace option (previously
// unreferenced) and confirms an in-place-configured plan produces the same
// solution as a default plan, both for a non-aliased Solve and for SolveInPlace.
func TestWithInPlace_MatchesDefault(t *testing.T) {
	nx, ny := 32, 24
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := float64(j+1) * hy
			u[i*ny+j] = math.Sin(math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly)
		}
	}

	rhs := make([]float64, nx*ny)
	if err := fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Dirichlet, poisson.Dirichlet,
	}); err != nil {
		t.Fatal(err)
	}

	ref, err := poisson.NewPlan(2, []int{nx, ny}, []float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})
	if err != nil {
		t.Fatalf("reference plan: %v", err)
	}
	inplace, err := poisson.NewPlan(2, []int{nx, ny}, []float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, poisson.WithInPlace(true))
	if err != nil {
		t.Fatalf("in-place plan: %v", err)
	}

	refSol := make([]float64, nx*ny)
	if err := ref.Solve(refSol, rhs); err != nil {
		t.Fatalf("reference solve: %v", err)
	}

	ipSol := make([]float64, nx*ny)
	if err := inplace.Solve(ipSol, rhs); err != nil {
		t.Fatalf("in-place plan solve: %v", err)
	}
	if e := maxAbsDiff(ipSol, refSol); e > 1e-12 {
		t.Fatalf("in-place plan Solve differs from default: %g", e)
	}
	if e := maxAbsDiff(refSol, u); e > 1e-9 {
		t.Fatalf("solution does not match manufactured field: %g", e)
	}

	// Aliased SolveInPlace with the in-place plan.
	buf := make([]float64, nx*ny)
	copy(buf, rhs)
	if err := inplace.SolveInPlace(buf); err != nil {
		t.Fatalf("SolveInPlace: %v", err)
	}
	if e := maxAbsDiff(buf, refSol); e > 1e-12 {
		t.Fatalf("SolveInPlace differs from default solve: %g", e)
	}
}

// TestWithNullspace_Functional exercises the WithNullspace option in its
// non-error modes with a real solve (the existing test only covers the
// construction-time rejection of NullspaceError).
func TestWithNullspace_Functional(t *testing.T) {
	const n = 48
	const h = 1.0 / float64(n)

	t.Run("SubtractMean", func(t *testing.T) {
		plan, err := poisson.NewPlan(1, []int{n}, []float64{h},
			[]poisson.BCType{poisson.Neumann},
			poisson.WithNullspace(poisson.NullspaceSubtractMean))
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		// A non-zero-mean RHS must be accepted (the mode subtracts the mean).
		rhs := randomField(7, n)
		got := make([]float64, n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		// Reapplying the operator reproduces the mean-projected RHS.
		want := make([]float64, n)
		copy(want, rhs)
		mean := sliceMean(want)
		for i := range want {
			want[i] -= mean
		}
		residual := make([]float64, n)
		if err := fd.Apply1D(residual, got, h, poisson.Neumann); err != nil {
			t.Fatal(err)
		}
		if e := relResidualError(residual, want); e > randomResidualRelTol {
			t.Fatalf("residual rel error %g exceeds tol %g", e, randomResidualRelTol)
		}
	})

	t.Run("ZeroMode", func(t *testing.T) {
		plan, err := poisson.NewPlan(1, []int{n}, []float64{h},
			[]poisson.BCType{poisson.Neumann},
			poisson.WithNullspace(poisson.NullspaceZeroMode))
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		// A compatible (mean-projected) RHS must solve under the zero-mode policy.
		rhs := randomField(9, n)
		mean := sliceMean(rhs)
		for i := range rhs {
			rhs[i] -= mean
		}
		got := make([]float64, n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}
		residual := make([]float64, n)
		if err := fd.Apply1D(residual, got, h, poisson.Neumann); err != nil {
			t.Fatal(err)
		}
		if e := relResidualError(residual, rhs); e > randomResidualRelTol {
			t.Fatalf("residual rel error %g exceeds tol %g", e, randomResidualRelTol)
		}
	})
}

// TestPlan2DPeriodic_SolveInPlace_Correctness is a dedicated (non-concurrency)
// correctness check for Plan2DPeriodic.SolveInPlace.
func TestPlan2DPeriodic_SolveInPlace_Correctness(t *testing.T) {
	nx, ny := 32, 32
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		for j := range ny {
			y := float64(j) * hy
			u[i*ny+j] = math.Sin(2.0*math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly)
		}
	}

	rhs := make([]float64, nx*ny)
	if err := fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Periodic, poisson.Periodic,
	}); err != nil {
		t.Fatal(err)
	}

	ref := make([]float64, nx*ny)
	if err := plan.Solve(ref, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	buf := make([]float64, nx*ny)
	copy(buf, rhs)
	if err := plan.SolveInPlace(buf); err != nil {
		t.Fatalf("SolveInPlace failed: %v", err)
	}

	if e := maxAbsDiff(buf, ref); e > 1e-12 {
		t.Fatalf("SolveInPlace differs from Solve: %g", e)
	}
	if e := maxAbsDiff(buf, u); e > periodic2dTol {
		t.Fatalf("SolveInPlace result does not match manufactured field: %g", e)
	}
}

// TestPlan3DPeriodic_SolveInPlace_Correctness is a dedicated (non-concurrency)
// correctness check for Plan3DPeriodic.SolveInPlace.
func TestPlan3DPeriodic_SolveInPlace_Correctness(t *testing.T) {
	n := 16
	h := 1.0 / float64(n)
	L := float64(n) * h

	plan, err := poisson.NewPlan3DPeriodic(n, n, n, h, h, h)
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	u := make([]float64, n*n*n)
	for i := range n {
		x := float64(i) * h
		for j := range n {
			y := float64(j) * h
			for k := range n {
				z := float64(k) * h
				u[(i*n+j)*n+k] = math.Sin(2.0*math.Pi*x/L) *
					math.Sin(2.0*math.Pi*y/L) *
					math.Cos(2.0*math.Pi*z/L)
			}
		}
	}

	rhs := make([]float64, n*n*n)
	if err := fd.Apply3D(rhs, u, grid.NewShape3D(n, n, n), [3]float64{h, h, h}, [3]poisson.BCType{
		poisson.Periodic, poisson.Periodic, poisson.Periodic,
	}); err != nil {
		t.Fatal(err)
	}

	ref := make([]float64, n*n*n)
	if err := plan.Solve(ref, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	buf := make([]float64, n*n*n)
	copy(buf, rhs)
	if err := plan.SolveInPlace(buf); err != nil {
		t.Fatalf("SolveInPlace failed: %v", err)
	}

	if e := maxAbsDiff(buf, ref); e > 1e-12 {
		t.Fatalf("SolveInPlace differs from Solve: %g", e)
	}
	if e := maxAbsDiff(buf, u); e > 1e-9 {
		t.Fatalf("SolveInPlace result does not match manufactured field: %g", e)
	}
}
