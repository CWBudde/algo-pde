// Command varcoeff demonstrates the variable-coefficient elliptic solver:
// it solves −∇·(a(x)∇u) = f on a 2D Dirichlet grid for a smooth, high-contrast
// coefficient a(x) that the constant-coefficient spectral solver cannot invert
// directly. Preconditioned conjugate gradient, with the spectral solve of a
// nearby constant-coefficient operator as preconditioner, converges in a small,
// grid-independent number of iterations.
package main

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func main() {
	const nx, ny = 128, 128
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	size := nx * ny

	// A smooth coefficient with ~exp(4) ≈ 55x contrast.
	a := make([]float64, size)
	minA, maxA := math.Inf(1), math.Inf(-1)
	for i := range nx {
		x := float64(i) * hx
		for j := range ny {
			y := float64(j) * hy
			v := math.Exp(2 * math.Sin(2*math.Pi*x) * math.Cos(2*math.Pi*y))
			a[i*ny+j] = v
			minA, maxA = math.Min(minA, v), math.Max(maxA, v)
		}
	}

	fmt.Printf("Variable-coefficient Poisson on a %dx%d Dirichlet grid\n", nx, ny)
	fmt.Printf("coefficient contrast max/min: %.1f\n", maxA/minA)

	plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, a)
	if err != nil {
		panic(err)
	}

	// A smooth source term.
	rhs := make([]float64, size)
	for i := range nx {
		x := float64(i) * hx
		for j := range ny {
			y := float64(j) * hy
			rhs[i*ny+j] = math.Sin(math.Pi*x) * math.Sin(math.Pi*y)
		}
	}

	u := make([]float64, size)
	stats, err := plan.Solve(u, rhs)
	if err != nil {
		panic(err)
	}
	fmt.Printf("PCG converged in %d iterations (relative residual %.2e)\n",
		stats.Iterations, stats.Residual)

	// Verify: reapply the operator and measure the max residual against the RHS.
	back := make([]float64, size)
	if err := plan.ApplyOperator(back, u); err != nil {
		panic(err)
	}
	fmt.Printf("max|L_a·u − f|: %.3e\n", maxAbsDiff(back, rhs))
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}
