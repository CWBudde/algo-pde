package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const neumann1dTol = 1e-10

func TestPlan1DNeumann_Solve_Mode1(t *testing.T) {
	n := 64
	h := 1.0 / float64(n)

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Neumann})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := (float64(i) + 0.5) * h
		u[i] = math.Cos(math.Pi * x)
	}

	checkNeumannDerivative(t, u, h)

	rhs := make([]float64, n)
	if err := fd.Apply1D(rhs, u, h, poisson.Neumann); err != nil {
		t.Fatal(err)
	}

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > neumann1dTol {
		t.Fatalf("max error %g exceeds tol %g", max, neumann1dTol)
	}
}

func TestPlan1DNeumann_Solve_Mode2(t *testing.T) {
	n := 96
	h := 1.0 / float64(n)

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Neumann})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := (float64(i) + 0.5) * h
		u[i] = math.Cos(2.0 * math.Pi * x)
	}

	checkNeumannDerivative(t, u, h)

	rhs := make([]float64, n)
	if err := fd.Apply1D(rhs, u, h, poisson.Neumann); err != nil {
		t.Fatal(err)
	}

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > neumann1dTol {
		t.Fatalf("max error %g exceeds tol %g", max, neumann1dTol)
	}
}

func TestPlan1DNeumann_NonZeroMean_Default(t *testing.T) {
	n := 32
	h := 1.0 / float64(n)

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Neumann})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, n)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, n)
	if err := plan.Solve(dst, rhs); !errors.Is(err, poisson.ErrNonZeroMean) {
		t.Fatalf("expected ErrNonZeroMean, got %v", err)
	}
}

func TestPlan1DNeumann_SubtractMean(t *testing.T) {
	n := 32
	h := 1.0 / float64(n)

	plan, err := poisson.NewPlan(
		1,
		[]int{n},
		[]float64{h},
		[]poisson.BCType{poisson.Neumann},
		poisson.WithSubtractMean(),
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, n)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, n)
	if err := plan.Solve(dst, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if mean := sliceMean(dst); math.Abs(mean) > neumann1dTol {
		t.Fatalf("mean %g exceeds tol %g", mean, neumann1dTol)
	}
}

func checkNeumannDerivative(t *testing.T, u []float64, h float64) {
	t.Helper()

	if len(u) < 3 {
		return
	}
	n := len(u)

	// The manufactured Neumann fields are cell-centered samples of cos(kπx),
	// whose physical derivative vanishes at both boundaries — that is exactly the
	// homogeneous Neumann data (∂u/∂x = 0) the solver enforces. Estimate that
	// boundary derivative with a genuine second-order one-sided finite difference
	// built from the three nearest cell-centered nodes and confirm it is
	// (numerically) zero. Unlike the previous (u[0]-u[0])/h tautology this
	// combines distinct sample values, so a corrupted field would be caught.
	//
	// The one-sided stencil (-2·u0 + 3·u1 - u2)/h approximates ∂u/∂x at the left
	// boundary x=0 (nodes sit at (i+½)h); its mirror does the right boundary. For
	// cos(kπx) with k up to 2 the residual is O(h²) ≈ 2e-3, far below the gate.
	const boundarySlopeTol = 1e-2

	leftDeriv := (-2*u[0] + 3*u[1] - u[2]) / h
	rightDeriv := (2*u[n-1] - 3*u[n-2] + u[n-3]) / h

	if math.Abs(leftDeriv) > boundarySlopeTol || math.Abs(rightDeriv) > boundarySlopeTol {
		t.Fatalf("expected ~zero Neumann boundary derivative, got left=%g right=%g (tol %g)",
			leftDeriv, rightDeriv, boundarySlopeTol)
	}
}
