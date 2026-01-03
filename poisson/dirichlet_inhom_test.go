package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const dirichletInhomTol = 1e-10

func TestApplyDirichletRHS1D_NonZero(t *testing.T) {
	n := 64
	h := 1.0 / float64(n+1)
	L := float64(n+1) * h

	u := make([]float64, n)
	for i := range n {
		x := float64(i+1) * h
		u[i] = math.Sin(math.Pi*x/L) + 0.2*x + 0.1
	}

	g0 := 0.1
	gL := 0.2*L + 0.1

	rhs := make([]float64, n)
	applyInhomDirichlet1D(rhs, u, h, g0, gL)

	err := poisson.ApplyDirichletRHS(rhs, grid.NewShape1D(n), [3]float64{h, 1, 1}, poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: []float64{g0}},
		{Face: poisson.XHigh, Type: poisson.Dirichlet, Values: []float64{gL}},
	})
	if err != nil {
		t.Fatalf("ApplyDirichletRHS failed: %v", err)
	}

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > dirichletInhomTol {
		t.Fatalf("max error %g exceeds tol %g", max, dirichletInhomTol)
	}
}

func TestApplyDirichletRHS2D_NonZero(t *testing.T) {
	nx := 48
	ny := 40
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy

	u := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		for j := 0; j < ny; j++ {
			y := float64(j+1) * hy
			u[i*ny+j] = math.Sin(math.Pi*x/Lx)*math.Sin(math.Pi*y/Ly) + 0.2*x + 0.3*y + 0.1
		}
	}

	xLow := make([]float64, ny)
	xHigh := make([]float64, ny)
	for j := 0; j < ny; j++ {
		y := float64(j+1) * hy
		xLow[j] = 0.3*y + 0.1
		xHigh[j] = 0.2*Lx + 0.3*y + 0.1
	}

	yLow := make([]float64, nx)
	yHigh := make([]float64, nx)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		yLow[i] = 0.2*x + 0.1
		yHigh[i] = 0.2*x + 0.3*Ly + 0.1
	}

	rhs := make([]float64, nx*ny)
	applyInhomDirichlet2D(rhs, u, grid.NewShape2D(nx, ny), hx, hy, xLow, xHigh, yLow, yHigh)

	err := poisson.ApplyDirichletRHS(rhs, grid.NewShape2D(nx, ny), [3]float64{hx, hy, 1}, poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: xLow},
		{Face: poisson.XHigh, Type: poisson.Dirichlet, Values: xHigh},
		{Face: poisson.YLow, Type: poisson.Dirichlet, Values: yLow},
		{Face: poisson.YHigh, Type: poisson.Dirichlet, Values: yHigh},
	})
	if err != nil {
		t.Fatalf("ApplyDirichletRHS failed: %v", err)
	}

	plan, err := poisson.NewPlan(2, []int{nx, ny}, []float64{hx, hy}, []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > dirichletInhomTol {
		t.Fatalf("max error %g exceeds tol %g", max, dirichletInhomTol)
	}
}

func applyInhomDirichlet1D(dst, src []float64, h, g0, gL float64) {
	n := len(src)
	invH2 := 1.0 / (h * h)
	for i := range n {
		left := g0
		if i > 0 {
			left = src[i-1]
		}
		right := gL
		if i+1 < n {
			right = src[i+1]
		}
		dst[i] = (2.0*src[i] - left - right) * invH2
	}
}

func applyInhomDirichlet2D(dst, src []float64, shape grid.Shape, hx, hy float64, xLow, xHigh, yLow, yHigh []float64) {
	nx := shape[0]
	ny := shape[1]
	invHx2 := 1.0 / (hx * hx)
	invHy2 := 1.0 / (hy * hy)

	for i := 0; i < nx; i++ {
		row := i * ny
		for j := 0; j < ny; j++ {
			idx := row + j
			u := src[idx]

			left := xLow[j]
			if i > 0 {
				left = src[(i-1)*ny+j]
			}

			right := xHigh[j]
			if i+1 < nx {
				right = src[(i+1)*ny+j]
			}

			down := yLow[i]
			if j > 0 {
				down = src[row+j-1]
			}

			up := yHigh[i]
			if j+1 < ny {
				up = src[row+j+1]
			}

			dst[idx] = (2.0*u-left-right)*invHx2 + (2.0*u-down-up)*invHy2
		}
	}
}
