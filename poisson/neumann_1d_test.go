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
	fd.Apply1D(rhs, u, h, poisson.Neumann)

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
	fd.Apply1D(rhs, u, h, poisson.Neumann)

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

	if len(u) == 0 {
		return
	}

	leftDeriv := (u[0] - u[0]) / h
	rightDeriv := (u[len(u)-1] - u[len(u)-1]) / h
	if math.Abs(leftDeriv) > neumann1dTol || math.Abs(rightDeriv) > neumann1dTol {
		t.Fatalf("expected zero boundary derivative, got %g %g", leftDeriv, rightDeriv)
	}
}
