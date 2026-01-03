package main

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func main() {
	// For Neumann, algo-pde uses DCT-II, which corresponds to a staggered grid
	// (cell-centered points).
	// Grid points x_i = (i + 0.5) * h.
	// Domain [0, L] with L = N*h.
	
	nx, ny := 64, 64
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)

	fmt.Printf("2D Neumann Poisson Solver\n")

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Neumann, poisson.Neumann},
	)
	if err != nil {
		panic(err)
	}

	// u_exact = cos(pi*x) * cos(pi*y)
	// -Lap u = 2*pi^2 * u
	// Grid points x_i = (i + 0.5)*h
	rhs := make([]float64, nx*ny)
	uExact := make([]float64, nx*ny)

	for i := 0; i < nx; i++ {
		x := (float64(i) + 0.5) * hx
		for j := 0; j < ny; j++ {
			y := (float64(j) + 0.5) * hy
			val := math.Cos(math.Pi*x) * math.Cos(math.Pi*y)
			uExact[i*ny+j] = val
			rhs[i*ny+j] = 2.0 * math.Pi * math.Pi * val
		}
	}

	u := make([]float64, nx*ny)
	// Neumann has a nullspace (constant function).
	// We handle it by enabling automatic mean subtraction (NullspaceSubtractMean).
	// This ensures the solution has zero mean and the RHS is projected onto the range.
	
	plan, err = poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Neumann, poisson.Neumann},
		poisson.WithSubtractMean(),
	)
	if err != nil {
		panic(err)
	}

	if err := plan.Solve(u, rhs); err != nil {
		panic(err)
	}

	// Since u is determined up to a constant, we should compare (u - mean(u)) with (uExact - mean(uExact)).
	// Or just align them at one point.
	// Or subtract mean from both. 
	
	meanU := 0.0
	meanExact := 0.0
	for k := range u {
		meanU += u[k]
		meanExact += uExact[k]
	}
	meanU /= float64(len(u))
	meanExact /= float64(len(uExact))

	maxErr := 0.0
	for i := range u {
		diff := math.Abs((u[i] - meanU) - (uExact[i] - meanExact))
		if diff > maxErr {
			maxErr = diff
		}
	}
	fmt.Printf("Max Error (modulo constant): %.3e\n", maxErr)
}
