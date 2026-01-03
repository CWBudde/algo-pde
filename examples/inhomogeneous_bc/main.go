package main

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

func main() {
	nx, ny := 24, 20
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
	)
	if err != nil {
		panic(err)
	}

	u := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		for j := 0; j < ny; j++ {
			y := float64(j+1) * hy
			u[i*ny+j] = x + y
		}
	}

	rhs := make([]float64, nx*ny)
	fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Dirichlet, poisson.Dirichlet,
	})

	xLow := make([]float64, ny)
	xHigh := make([]float64, ny)
	for j := 0; j < ny; j++ {
		y := float64(j+1) * hy
		xLow[j] = y
		xHigh[j] = Lx + y
	}

	yLow := make([]float64, nx)
	yHigh := make([]float64, nx)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		yLow[i] = x
		yHigh[i] = x + Ly
	}

	bc := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: xLow},
		{Face: poisson.XHigh, Type: poisson.Dirichlet, Values: xHigh},
		{Face: poisson.YLow, Type: poisson.Dirichlet, Values: yLow},
		{Face: poisson.YHigh, Type: poisson.Dirichlet, Values: yHigh},
	}

	got := make([]float64, nx*ny)
	if err := plan.SolveWithBC(got, rhs, bc); err != nil {
		panic(err)
	}

	fmt.Printf("max error: %.3e\n", maxAbsDiff(got, u))
}

func maxAbsDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	max := 0.0
	for i := range a {
		diff := math.Abs(a[i] - b[i])
		if diff > max {
			max = diff
		}
	}

	return max
}
