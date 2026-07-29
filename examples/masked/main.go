// Command masked demonstrates the immersed-boundary (masked-domain) solver: it
// solves −Δu = f on a disk inscribed in a square, with u = 0 on the disk's
// boundary and exterior. The disk is embedded in the enclosing rectangular grid
// and the fast spectral solver of that rectangle preconditions a conjugate-
// gradient iteration that enforces the u = 0 immersed boundary, so a non-
// rectangular domain is solved in a handful of grid-independent iterations.
package main

import (
	"fmt"
	"math"

	"github.com/cwbudde/algo-pde/poisson"
)

func main() {
	const nx, ny = 128, 128
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	size := nx * ny

	// A disk inscribed in the square: cells inside are active, the corners are
	// masked (solid), giving a non-rectangular domain with u = 0 on the boundary.
	cx, cy := float64(nx-1)/2, float64(ny-1)/2
	radius := 0.45 * math.Min(float64(nx), float64(ny))
	mask := make([]bool, size)
	active := 0
	for i := range nx {
		for j := range ny {
			dx, dy := float64(i)-cx, float64(j)-cy
			if dx*dx+dy*dy <= radius*radius {
				mask[i*ny+j] = true
				active++
			}
		}
	}

	fmt.Printf("Masked Poisson on a disk inscribed in a %dx%d grid\n", nx, ny)
	fmt.Printf("active (interior) cells: %d of %d\n", active, size)

	plan, err := poisson.NewMaskedPlan(2, []int{nx, ny}, []float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, mask)
	if err != nil {
		panic(err)
	}

	// A smooth source term (its value on the masked cells is ignored).
	rhs := make([]float64, size)
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := float64(j+1) * hy
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

	// Verify: reapply the operator and measure the max residual over the active
	// cells (on the masked cells both the operator output and the RHS are zero).
	back := make([]float64, size)
	if err := plan.ApplyOperator(back, u); err != nil {
		panic(err)
	}
	resid := 0.0
	for idx := range mask {
		if mask[idx] {
			if d := math.Abs(back[idx] - rhs[idx]); d > resid {
				resid = d
			}
		}
	}
	fmt.Printf("max|L·u − f| over active cells: %.3e\n", resid)
}
