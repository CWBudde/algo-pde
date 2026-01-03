package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const neumannInhomTol = 1e-9

func TestApplyNeumannRHS1D_NonZero(t *testing.T) {
	n := 96
	h := 1.0 / float64(n)

	u := make([]float64, n)
	for i := range n {
		x := (float64(i) + 0.5) * h
		u[i] = math.Sin(math.Pi*x) + 0.25*x*x
	}

	g0 := math.Pi
	gL := -math.Pi + 0.5

	mean := sliceMean(u)
	for i := range u {
		u[i] -= mean
	}

	rhs := make([]float64, n)
	applyInhomNeumann1D(rhs, u, h, g0, gL)

	err := poisson.ApplyNeumannRHS(rhs, grid.NewShape1D(n), [3]float64{h, 1, 1}, poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Neumann, Values: []float64{g0}},
		{Face: poisson.XHigh, Type: poisson.Neumann, Values: []float64{gL}},
	})
	if err != nil {
		t.Fatalf("ApplyNeumannRHS failed: %v", err)
	}

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Neumann})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > neumannInhomTol {
		t.Fatalf("max error %g exceeds tol %g", max, neumannInhomTol)
	}

	leftDeriv := (-3.0*got[0] + 4.0*got[1] - got[2]) / (2.0 * h)
	rightDeriv := (3.0*got[n-1] - 4.0*got[n-2] + got[n-3]) / (2.0 * h)
	if math.Abs(leftDeriv-g0) > 5e-3 || math.Abs(rightDeriv-gL) > 5e-3 {
		t.Fatalf("boundary derivatives %g/%g do not match %g/%g", leftDeriv, rightDeriv, g0, gL)
	}
}

func TestApplyNeumannRHS2D_NonZero(t *testing.T) {
	nx := 40
	ny := 36
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)

	u := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		x := (float64(i) + 0.5) * hx
		for j := 0; j < ny; j++ {
			y := (float64(j) + 0.5) * hy
			u[i*ny+j] = math.Sin(math.Pi*x)*math.Sin(math.Pi*y) + 0.2*x + 0.3*y
		}
	}

	mean := sliceMean(u)
	for i := range u {
		u[i] -= mean
	}

	xLow := make([]float64, ny)
	xHigh := make([]float64, ny)
	for j := 0; j < ny; j++ {
		y := (float64(j) + 0.5) * hy
		xLow[j] = math.Pi*math.Sin(math.Pi*y) + 0.2
		xHigh[j] = -math.Pi*math.Sin(math.Pi*y) + 0.2
	}

	yLow := make([]float64, nx)
	yHigh := make([]float64, nx)
	for i := 0; i < nx; i++ {
		x := (float64(i) + 0.5) * hx
		yLow[i] = math.Pi*math.Sin(math.Pi*x) + 0.3
		yHigh[i] = -math.Pi*math.Sin(math.Pi*x) + 0.3
	}

	rhs := make([]float64, nx*ny)
	applyInhomNeumann2D(rhs, u, grid.NewShape2D(nx, ny), hx, hy, xLow, xHigh, yLow, yHigh)

	err := poisson.ApplyNeumannRHS(rhs, grid.NewShape2D(nx, ny), [3]float64{hx, hy, 1}, poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Neumann, Values: xLow},
		{Face: poisson.XHigh, Type: poisson.Neumann, Values: xHigh},
		{Face: poisson.YLow, Type: poisson.Neumann, Values: yLow},
		{Face: poisson.YHigh, Type: poisson.Neumann, Values: yHigh},
	})
	if err != nil {
		t.Fatalf("ApplyNeumannRHS failed: %v", err)
	}

	plan, err := poisson.NewPlan(2, []int{nx, ny}, []float64{hx, hy}, []poisson.BCType{poisson.Neumann, poisson.Neumann})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > neumannInhomTol {
		t.Fatalf("max error %g exceeds tol %g", max, neumannInhomTol)
	}
}

func applyInhomNeumann1D(dst, src []float64, h, g0, gL float64) {
	n := len(src)
	invH2 := 1.0 / (h * h)
	for i := range n {
		left := src[i] - g0*h
		if i > 0 {
			left = src[i-1]
		}
		right := src[i] + gL*h
		if i+1 < n {
			right = src[i+1]
		}
		dst[i] = (2.0*src[i] - left - right) * invH2
	}
}

func applyInhomNeumann2D(dst, src []float64, shape grid.Shape, hx, hy float64, xLow, xHigh, yLow, yHigh []float64) {
	nx := shape[0]
	ny := shape[1]
	invHx2 := 1.0 / (hx * hx)
	invHy2 := 1.0 / (hy * hy)

	for i := 0; i < nx; i++ {
		row := i * ny
		for j := 0; j < ny; j++ {
			idx := row + j
			u := src[idx]

			left := src[idx] - xLow[j]*hx
			if i > 0 {
				left = src[(i-1)*ny+j]
			}

			right := src[idx] + xHigh[j]*hx
			if i+1 < nx {
				right = src[(i+1)*ny+j]
			}

			down := src[idx] - yLow[i]*hy
			if j > 0 {
				down = src[row+j-1]
			}

			up := src[idx] + yHigh[i]*hy
			if j+1 < ny {
				up = src[row+j+1]
			}

			dst[idx] = (2.0*u-left-right)*invHx2 + (2.0*u-down-up)*invHy2
		}
	}
}
