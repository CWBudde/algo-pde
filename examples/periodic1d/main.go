package main

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func main() {
	// 1. Setup grid
	nx := 64
	hx := 1.0 / float64(nx)

	// 2. Create Plan
	// We want to solve -u_xx = f with periodic BCs
	plan, err := poisson.NewPlan1DPeriodic(nx, hx)
	if err != nil {
		panic(err)
	}

	// 3. Setup RHS (f) and Exact Solution (u_exact)
	// u_exact = sin(2*pi*x)
	// -u_xx = 4*pi^2 * sin(2*pi*x)
	rhs := make([]float64, nx)
	uExact := make([]float64, nx)
	
	for i := 0; i < nx; i++ {
		x := float64(i) * hx
		uExact[i] = math.Sin(2.0 * math.Pi * x)
		rhs[i] = 4.0 * math.Pi * math.Pi * math.Sin(2.0 * math.Pi * x)
	}

	// 4. Solve
	u := make([]float64, nx)

	// We use WithSubtractMean() to handle the zero mode (singular matrix for periodic Poisson)
	// automatically by subtracting the mean from RHS (which should be 0 anyway for sin)
	// and setting the mean of the solution to 0.
	plan, err = poisson.NewPlan1DPeriodic(nx, hx, poisson.WithSubtractMean())
	if err != nil {
		panic(err)
	}

	if err := plan.Solve(u, rhs); err != nil {
		panic(err)
	}

	// 5. Check error
	maxErr := 0.0
	for i := 0; i < nx; i++ {
		errVal := math.Abs(u[i] - uExact[i])
		if errVal > maxErr {
			maxErr = errVal
		}
	}

	fmt.Printf("1D Periodic Poisson Solver\n")
	fmt.Printf("Grid: %d, h: %.4f\n", nx, hx)
	fmt.Printf("Max Error: %.3e\n", maxErr)
}
