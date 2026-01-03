package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const inhomAPITol = 1e-9

func TestPlan2D_SolveWithBC_DirichletNeumann(t *testing.T) {
	nx, ny := 48, 36
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny)
	Lx := float64(nx+1) * hx
	Ly := float64(ny) * hy

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Neumann},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		for j := 0; j < ny; j++ {
			y := (float64(j) + 0.5) * hy
			u[i*ny+j] = math.Sin(math.Pi*x/Lx)*math.Cos(math.Pi*y/Ly) + 0.2*x + 0.3*y + 0.1
		}
	}

	xLow := make([]float64, ny)
	xHigh := make([]float64, ny)
	for j := 0; j < ny; j++ {
		y := (float64(j) + 0.5) * hy
		xLow[j] = 0.3*y + 0.1
		xHigh[j] = 0.2*Lx + 0.3*y + 0.1
	}

	yLow := make([]float64, nx)
	yHigh := make([]float64, nx)
	for i := 0; i < nx; i++ {
		yLow[i] = 0.3
		yHigh[i] = 0.3
	}

	rhs := make([]float64, nx*ny)
	applyInhomDirichletNeumann2D(rhs, u, grid.NewShape2D(nx, ny), hx, hy, xLow, xHigh, yLow, yHigh)

	bc := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: xLow},
		{Face: poisson.XHigh, Type: poisson.Dirichlet, Values: xHigh},
		{Face: poisson.YLow, Type: poisson.Neumann, Values: yLow},
		{Face: poisson.YHigh, Type: poisson.Neumann, Values: yHigh},
	}

	got := make([]float64, nx*ny)
	if err := plan.SolveWithBC(got, rhs, bc); err != nil {
		t.Fatalf("SolveWithBC failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > inhomAPITol {
		t.Fatalf("max error %g exceeds tol %g", max, inhomAPITol)
	}
}

func TestPlan3D_SolveWithBC_DirichletDirichletNeumann(t *testing.T) {
	nx, ny, nz := 24, 20, 18
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	hz := 1.0 / float64(nz)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy
	Lz := float64(nz) * hz

	plan, err := poisson.NewPlan(
		3,
		[]int{nx, ny, nz},
		[]float64{hx, hy, hz},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet, poisson.Neumann},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	shape := grid.NewShape3D(nx, ny, nz)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		for j := 0; j < ny; j++ {
			y := float64(j+1) * hy
			for k := 0; k < nz; k++ {
				z := (float64(k) + 0.5) * hz
				u[grid.Index3D(i, j, k, shape)] = math.Sin(math.Pi*x/Lx)*math.Sin(math.Pi*y/Ly)*math.Cos(math.Pi*z/Lz) + 0.2*x + 0.1 + 0.3*z
			}
		}
	}

	xLow := make([]float64, ny*nz)
	xHigh := make([]float64, ny*nz)
	for j := 0; j < ny; j++ {
		for k := 0; k < nz; k++ {
			z := (float64(k) + 0.5) * hz
			idx := j*nz + k
			xLow[idx] = 0.1 + 0.3*z
			xHigh[idx] = 0.2*Lx + 0.1 + 0.3*z
		}
	}

	yLow := make([]float64, nx*nz)
	yHigh := make([]float64, nx*nz)
	for i := 0; i < nx; i++ {
		x := float64(i+1) * hx
		for k := 0; k < nz; k++ {
			z := (float64(k) + 0.5) * hz
			idx := i*nz + k
			yLow[idx] = 0.2*x + 0.1 + 0.3*z
			yHigh[idx] = 0.2*x + 0.1 + 0.3*z
		}
	}

	zLow := make([]float64, nx*ny)
	zHigh := make([]float64, nx*ny)
	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			idx := i*ny + j
			zLow[idx] = 0.3
			zHigh[idx] = 0.3
		}
	}

	rhs := make([]float64, nx*ny*nz)
	applyInhomDirichletNeumann3D(rhs, u, shape, hx, hy, hz, xLow, xHigh, yLow, yHigh, zLow, zHigh)

	bc := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: xLow},
		{Face: poisson.XHigh, Type: poisson.Dirichlet, Values: xHigh},
		{Face: poisson.YLow, Type: poisson.Dirichlet, Values: yLow},
		{Face: poisson.YHigh, Type: poisson.Dirichlet, Values: yHigh},
		{Face: poisson.ZLow, Type: poisson.Neumann, Values: zLow},
		{Face: poisson.ZHigh, Type: poisson.Neumann, Values: zHigh},
	}

	got := make([]float64, nx*ny*nz)
	if err := plan.SolveWithBC(got, rhs, bc); err != nil {
		t.Fatalf("SolveWithBC failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > inhomAPITol {
		t.Fatalf("max error %g exceeds tol %g", max, inhomAPITol)
	}
}

func applyInhomDirichletNeumann2D(dst, src []float64, shape grid.Shape, hx, hy float64, xLow, xHigh, yLow, yHigh []float64) {
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

func applyInhomDirichletNeumann3D(dst, src []float64, shape grid.Shape, hx, hy, hz float64, xLow, xHigh, yLow, yHigh, zLow, zHigh []float64) {
	nx := shape[0]
	ny := shape[1]
	nz := shape[2]
	invHx2 := 1.0 / (hx * hx)
	invHy2 := 1.0 / (hy * hy)
	invHz2 := 1.0 / (hz * hz)
	plane := ny * nz

	for i := 0; i < nx; i++ {
		iPlane := i * plane
		for j := 0; j < ny; j++ {
			row := iPlane + j*nz
			for k := 0; k < nz; k++ {
				idx := row + k
				u := src[idx]

				left := xLow[j*nz+k]
				if i > 0 {
					left = src[idx-plane]
				}

				right := xHigh[j*nz+k]
				if i+1 < nx {
					right = src[idx+plane]
				}

				down := yLow[i*nz+k]
				if j > 0 {
					down = src[idx-nz]
				}

				up := yHigh[i*nz+k]
				if j+1 < ny {
					up = src[idx+nz]
				}

				back := src[idx] - zLow[i*ny+j]*hz
				if k > 0 {
					back = src[idx-1]
				}

				front := src[idx] + zHigh[i*ny+j]*hz
				if k+1 < nz {
					front = src[idx+1]
				}

				dst[idx] = (2.0*u-left-right)*invHx2 + (2.0*u-down-up)*invHy2 + (2.0*u-back-front)*invHz2
			}
		}
	}
}
