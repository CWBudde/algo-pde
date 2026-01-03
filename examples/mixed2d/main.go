package main

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func main() {
	// Mixed BC: Periodic in X, Dirichlet in Y.
	// X: [0, 1), hx = 1/Nx
	// Y: [0, 1], hy = 1/(Ny+1), points at (j+1)*hy
	
	nx, ny := 64, 64
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny+1)

	fmt.Printf("2D Mixed Poisson Solver (X: Periodic, Y: Dirichlet)\n")

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Periodic, poisson.Dirichlet},
	)
	if err != nil {
		panic(err)
	}

	// u_exact = sin(2*pi*x) * sin(pi*y)
	// -Lap u = (4*pi^2 + pi^2) * u = 5*pi^2 * u
	
	rhs := make([]float64, nx*ny)
	uExact := make([]float64, nx*ny)

	for i := 0; i < nx; i++ {
		x := float64(i) * hx
		for j := 0; j < ny; j++ {
			y := float64(j+1) * hy
			val := math.Sin(2.0*math.Pi*x) * math.Sin(math.Pi*y)
			uExact[i*ny+j] = val
			rhs[i*ny+j] = 5.0 * math.Pi * math.Pi * val
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
