package main

import (
	"fmt"
	"math"

	"github.com/cwbudde/algo-pde/poisson"
)

func main() {
	// For Dirichlet, N is the number of internal grid points.
	// The domain length L = (N+1)*h.
	nx, ny := 64, 64
	// If nx=64 points, L = (64+1)*h = 1 => h = 1/65.
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)

	fmt.Printf("2D Dirichlet Poisson Solver\n")

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
	)
	if err != nil {
		panic(err)
	}

	// u_exact = sin(pi*x) * sin(pi*y)
	// -Lap u = 2*pi^2 * u
	// Grid points are at x_i = (i+1)*h for i=0..N-1
	rhs := make([]float64, nx*ny)
	uExact := make([]float64, nx*ny)

	for i := range nx {
		x := float64(i+1) * hx

		for j := range ny {
			y := float64(j+1) * hy
			val := math.Sin(math.Pi*x) * math.Sin(math.Pi*y)
			uExact[i*ny+j] = val
			rhs[i*ny+j] = 2.0 * math.Pi * math.Pi * val
		}
	}

	u := make([]float64, nx*ny)
	if err := plan.Solve(u, rhs); err != nil {
		panic(err)
	}

	maxErr := 0.0

	for i := range u {
		diff := math.Abs(u[i] - uExact[i])
		if diff > maxErr {
			maxErr = diff
		}
	}

	fmt.Printf("Max Error: %.3e\n", maxErr)
}
