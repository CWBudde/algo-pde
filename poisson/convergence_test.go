package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

const convergenceMinRate = 1.8

func TestConvergence1D_Dirichlet(t *testing.T) {
	sizes := []int{32, 64, 128}
	errors := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h
		hs[idx] = h

		k := math.Pi / L
		lambda := k * k

		u := make([]float64, n)
		for i := range n {
			x := float64(i+1) * h
			u[i] = math.Sin(math.Pi * x / L)
		}

		plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet})
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		rhs := make([]float64, n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		got := make([]float64, n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errors[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errors)
}

func TestConvergence2D_Dirichlet(t *testing.T) {
	sizes := []int{16, 32, 64}
	errors := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h
		hs[idx] = h

		k := math.Pi / L
		lambda := 2.0 * k * k

		u := make([]float64, n*n)
		for i := range n {
			x := float64(i+1) * h
			for j := range n {
				y := float64(j+1) * h
				u[i*n+j] = math.Sin(math.Pi*x/L) * math.Sin(math.Pi*y/L)
			}
		}

		plan, err := poisson.NewPlan(
			2,
			[]int{n, n},
			[]float64{h, h},
			[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
		)
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		got := make([]float64, n*n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errors[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errors)
}

func checkConvergenceRates(t *testing.T, hs, errors []float64) {
	t.Helper()

	for i := 0; i < len(errors)-1; i++ {
		if errors[i] == 0 || errors[i+1] == 0 {
			t.Fatalf("zero error encountered: %g -> %g", errors[i], errors[i+1])
		}
		rate := math.Log(errors[i+1]/errors[i]) / math.Log(hs[i+1]/hs[i])
		if rate < convergenceMinRate {
			t.Fatalf("convergence rate %.3f below threshold %.1f", rate, convergenceMinRate)
		}
	}
}
