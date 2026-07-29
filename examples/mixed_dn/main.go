package main

import (
	"fmt"
	"math"

	"github.com/CWBudde/algo-pde/poisson"
)

func main() {
	// Per-face asymmetric (mixed) axes solved with quarter-wave transforms:
	//   X: DirichletNeumann — Dirichlet at the low face (x=0), Neumann at the
	//      high face (x=Lx). Diagonalised by DST-IV.
	//   Y: NeumannDirichlet — Neumann at the low face (y=0), Dirichlet at the
	//      high face (y=Ly). Diagonalised by DCT-IV.
	// Both axes use the cell-centered grid x_i = (i+½)h, L = n·h.
	nx, ny := 64, 64
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy

	fmt.Println("2D Mixed Poisson Solver (X: DirichletNeumann, Y: NeumannDirichlet)")

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.DirichletNeumann, poisson.NeumannDirichlet},
	)
	if err != nil {
		panic(err)
	}

	// Manufactured solution: the lowest quarter-wave mode on each axis.
	//   X: sin(πx/2Lx)  — zero at the Dirichlet low face, zero slope at the
	//                     Neumann high face.
	//   Y: cos(πy/2Ly)  — zero slope at the Neumann low face, zero at the
	//                     Dirichlet high face.
	// -Δu = ((π/2Lx)² + (π/2Ly)²) · u.
	kx := math.Pi / (2.0 * Lx)
	ky := math.Pi / (2.0 * Ly)
	lambda := kx*kx + ky*ky

	rhs := make([]float64, nx*ny)
	uExact := make([]float64, nx*ny)
	for i := range nx {
		x := (float64(i) + 0.5) * hx
		for j := range ny {
			y := (float64(j) + 0.5) * hy
			val := math.Sin(kx*x) * math.Cos(ky*y)
			uExact[i*ny+j] = val
			rhs[i*ny+j] = lambda * val
		}
	}

	u := make([]float64, nx*ny)
	if err := plan.Solve(u, rhs); err != nil {
		panic(err)
	}

	maxErr := 0.0
	for i := range u {
		if diff := math.Abs(u[i] - uExact[i]); diff > maxErr {
			maxErr = diff
		}
	}
	fmt.Printf("Max Error: %.3e\n", maxErr)
}
