package main

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

// Implicit diffusion step: u^{n+1} - nu*dt*Δu^{n+1} = u^n
// => (alpha - Δ)u^{n+1} = u^n / (nu*dt), where alpha = 1/(nu*dt).
func main() {
	nx, ny := 64, 64
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	nu := 0.1
	dt := 0.05
	alpha := 1.0 / (nu * dt)

	plan, err := poisson.NewHelmholtzPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Periodic, poisson.Periodic},
		alpha,
	)
	if err != nil {
		panic(err)
	}

	u0 := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		x := float64(i) * hx
		for j := 0; j < ny; j++ {
			y := float64(j) * hy
			u0[i*ny+j] = math.Sin(2.0*math.Pi*x) * math.Cos(2.0*math.Pi*y)
		}
	}

	rhs := make([]float64, nx*ny)
	for i, v := range u0 {
		rhs[i] = v * alpha
	}

	u1 := make([]float64, nx*ny)
	if err := plan.Solve(u1, rhs); err != nil {
		panic(err)
	}

	fmt.Printf("max |u^{n+1}|: %.6f\n", maxAbs(u1))
}

func maxAbs(values []float64) float64 {
	max := 0.0
	for _, v := range values {
		if v < 0 {
			v = -v
		}
		if v > max {
			max = v
		}
	}
	return max
}
