package poisson_test

import (
	"errors"
	"testing"

	"github.com/CWBudde/algo-pde/poisson"
)

// TestPlan2D_SolveWithBC_QuarterWavePlanar pins the inhomogeneous per-face
// lifting on mixed quarter-wave axes at machine precision. A planar field
// u = a·x + c·y + b is linear, so the second-difference stencil and the ghost
// reflections (u₋₁ = 2g − u₀ on a Dirichlet face, u₋₁ = u₀ − h·g on a Neumann
// face) are all exact for it. The discrete solve must therefore reproduce u
// exactly — but only if the Dirichlet face of a mixed axis lifts with the
// quarter-wave factor 2 (a factor-1 lift fails for any non-zero boundary value).
//
// x is DirichletNeumann (Dirichlet low, Neumann high); y is NeumannDirichlet
// (Neumann low, Dirichlet high). Both use the cell-centered grid x_i=(i+½)h.
func TestPlan2D_SolveWithBC_QuarterWavePlanar(t *testing.T) {
	// Extents avoid a factor of 5 so the 4N quarter-wave FFT is sound.
	nx, ny := 8, 6
	hx, hy := 0.1, 0.15
	Ly := float64(ny) * hy
	const a, c, b = 1.3, -0.7, 0.5

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.DirichletNeumann, poisson.NeumannDirichlet},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	xAt := func(i int) float64 { return (float64(i) + 0.5) * hx }
	yAt := func(j int) float64 { return (float64(j) + 0.5) * hy }

	u := make([]float64, nx*ny)
	for i := range nx {
		for j := range ny {
			u[i*ny+j] = a*xAt(i) + c*yAt(j) + b
		}
	}

	// x DirichletNeumann: low face is Dirichlet (u at x=0), high face Neumann
	// (∂u/∂x = a). Face values are indexed by the tangential y node.
	xLow := make([]float64, ny)  // Dirichlet value u(0, y_j) = c·y_j + b
	xHigh := make([]float64, ny) // Neumann value ∂u/∂x = a
	for j := range ny {
		xLow[j] = c*yAt(j) + b
		xHigh[j] = a
	}

	// y NeumannDirichlet: low face Neumann (∂u/∂y = c), high face Dirichlet
	// (u at y=Ly), indexed by the tangential x node.
	yLow := make([]float64, nx)  // Neumann value ∂u/∂y = c
	yHigh := make([]float64, nx) // Dirichlet value u(x_i, Ly) = a·x_i + c·Ly + b
	for i := range nx {
		yLow[i] = c
		yHigh[i] = a*xAt(i) + c*Ly + b
	}

	bc := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: xLow},
		{Face: poisson.XHigh, Type: poisson.Neumann, Values: xHigh},
		{Face: poisson.YLow, Type: poisson.Neumann, Values: yLow},
		{Face: poisson.YHigh, Type: poisson.Dirichlet, Values: yHigh},
	}

	// -Δu = 0 for a planar field.
	rhs := make([]float64, nx*ny)

	got := make([]float64, nx*ny)
	if err := plan.SolveWithBC(got, rhs, bc); err != nil {
		t.Fatalf("SolveWithBC failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > inhomAPITol {
		t.Fatalf("max error %g exceeds tol %g (mixed-axis inhomogeneous lift)", max, inhomAPITol)
	}
}

// TestPlan_SolveWithBC_MixedFaceTypeValidation checks that a mixed axis requires
// the correct BC type per face: on a DirichletNeumann axis the low face must be
// Dirichlet and the high face Neumann; the swapped pairing is rejected.
func TestPlan_SolveWithBC_MixedFaceTypeValidation(t *testing.T) {
	nx, ny := 8, 6
	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{0.1, 0.1},
		[]poisson.BCType{poisson.DirichletNeumann, poisson.Neumann},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, nx*ny)
	got := make([]float64, nx*ny)

	// Low face of a DirichletNeumann axis must be Dirichlet, not Neumann.
	badLow := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Neumann, Values: make([]float64, ny)},
	}
	if err := plan.SolveWithBC(got, rhs, badLow); err == nil {
		t.Fatal("expected error for Neumann data on the Dirichlet (low) face of a DirichletNeumann axis")
	}

	// High face must be Neumann, not Dirichlet.
	badHigh := poisson.BoundaryConditions{
		{Face: poisson.XHigh, Type: poisson.Dirichlet, Values: make([]float64, ny)},
	}
	if err := plan.SolveWithBC(got, rhs, badHigh); err == nil {
		t.Fatal("expected error for Dirichlet data on the Neumann (high) face of a DirichletNeumann axis")
	}

	// The correct pairing (Dirichlet low, Neumann high) is accepted.
	okBC := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: make([]float64, ny)},
		{Face: poisson.XHigh, Type: poisson.Neumann, Values: make([]float64, ny)},
	}
	if err := plan.SolveWithBC(got, rhs, okBC); err != nil {
		t.Fatalf("correct mixed face pairing rejected: %v", err)
	}

	// A rejected call must not have been a nil-buffer false positive.
	if errors.Is(plan.SolveWithBC(got, rhs, badLow), poisson.ErrNilBuffer) {
		t.Fatal("unexpected ErrNilBuffer")
	}
}
