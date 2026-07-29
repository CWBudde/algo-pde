// Command projection demonstrates the Helmholtz–Hodge pressure projection:
// it takes a noisy, compressible 2D periodic velocity field and projects it
// onto its divergence-free part, the way an incompressible flow solver enforces
// ∇·u = 0 after an unconstrained velocity update.
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/CWBudde/algo-pde/poisson"
)

func main() {
	nx, ny := 128, 128
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)

	fmt.Printf("Pressure projection of a %dx%d periodic velocity field...\n", nx, ny)

	plan, err := poisson.NewProjectionPlan2D(nx, ny, hx, hy)
	if err != nil {
		panic(err)
	}

	// A smooth swirl (divergence-free) plus random compressible noise.
	rng := rand.New(rand.NewSource(42))
	u := make([]float64, nx*ny)
	v := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		for j := range ny {
			y := float64(j) * hy
			idx := i*ny + j
			u[idx] = math.Sin(2*math.Pi*x)*math.Cos(2*math.Pi*y) + 0.3*rng.NormFloat64()
			v[idx] = -math.Cos(2*math.Pi*x)*math.Sin(2*math.Pi*y) + 0.3*rng.NormFloat64()
		}
	}

	div := make([]float64, nx*ny)
	if err := plan.Divergence(div, u, v); err != nil {
		panic(err)
	}
	fmt.Printf("max|div| before projection: %.3e\n", maxAbs(div))

	if err := plan.Project(u, v); err != nil {
		panic(err)
	}

	if err := plan.Divergence(div, u, v); err != nil {
		panic(err)
	}
	fmt.Printf("max|div| after  projection: %.3e\n", maxAbs(div))
}

func maxAbs(s []float64) float64 {
	m := 0.0
	for _, v := range s {
		if a := math.Abs(v); a > m {
			m = a
		}
	}
	return m
}
