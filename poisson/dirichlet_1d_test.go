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

	if math.Abs(math.Sin(0)) > dirichlet1dTol || math.Abs(math.Sin(math.Pi)) > dirichlet1dTol {
		t.Fatalf("expected Dirichlet boundary values to be zero")
	}

	rhs := make([]float64, n)
	fd.Apply1D(rhs, u, h, poisson.Dirichlet)

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > dirichlet1dTol {
		t.Fatalf("max error %g exceeds tol %g", max, dirichlet1dTol)
	}
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

	u0 := math.Sin(0) + 0.3*math.Sin(0)
	uL := math.Sin(math.Pi) + 0.3*math.Sin(2.0*math.Pi)
	if math.Abs(u0) > dirichlet1dTol || math.Abs(uL) > dirichlet1dTol {
		t.Fatalf("expected Dirichlet boundary values to be zero")
	}

	rhs := make([]float64, n)
	fd.Apply1D(rhs, u, h, poisson.Dirichlet)

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > dirichlet1dTol {
		t.Fatalf("max error %g exceeds tol %g", max, dirichlet1dTol)
	}
}
