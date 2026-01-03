package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const (
	helmholtz1dTol = 1e-10
	helmholtz2dTol = 1e-9
	helmholtz3dTol = 1e-9
)

func TestHelmholtzPlan1D_PositiveAlpha(t *testing.T) {
	n := 64
	h := 1.0 / float64(n+1)
	L := float64(n+1) * h
	alpha := 2.75

	plan, err := poisson.NewHelmholtzPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet}, alpha)
	if err != nil {
		t.Fatalf("NewHelmholtzPlan failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := float64(i+1) * h
		u[i] = math.Sin(math.Pi*x/L) + 0.3*math.Sin(3.0*math.Pi*x/L)
	}

	lap := make([]float64, n)
	fd.Apply1D(lap, u, h, poisson.Dirichlet)

	rhs := make([]float64, n)
	for i := range n {
		rhs[i] = lap[i] + alpha*u[i]
	}

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > helmholtz1dTol {
		t.Fatalf("max error %g exceeds tol %g", max, helmholtz1dTol)
	}
}

func TestHelmholtzPlan1D_NegativeAlphaResonant(t *testing.T) {
	n := 32
	h := 1.0 / float64(n+1)
	alpha := -fd.EigenvaluesDirichlet(n, h)[0]

	plan, err := poisson.NewHelmholtzPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet}, alpha)
	if err != nil {
		t.Fatalf("NewHelmholtzPlan failed: %v", err)
	}

	rhs := make([]float64, n)
	dst := make([]float64, n)
	if err := plan.Solve(dst, rhs); !errors.Is(err, poisson.ErrResonant) {
		t.Fatalf("expected ErrResonant, got %v", err)
	}
}

func TestHelmholtzPlan2D_PositiveAlpha(t *testing.T) {
	nx, ny := 48, 36
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy
	alpha := 1.25

	plan, err := poisson.NewHelmholtzPlan(2, []int{nx, ny}, []float64{hx, hy}, []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, alpha)
	if err != nil {
		t.Fatalf("NewHelmholtzPlan failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := float64(j+1) * hy
			u[i*ny+j] = math.Sin(math.Pi*x/Lx)*math.Sin(2.0*math.Pi*y/Ly) +
				0.2*math.Sin(3.0*math.Pi*x/Lx)*math.Sin(math.Pi*y/Ly)
		}
	}

	lap := make([]float64, nx*ny)
	fd.Apply2D(lap, u, grid.Shape{nx, ny}, [2]float64{hx, hy}, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})

	rhs := make([]float64, nx*ny)
	for i := range rhs {
		rhs[i] = lap[i] + alpha*u[i]
	}

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > helmholtz2dTol {
		t.Fatalf("max error %g exceeds tol %g", max, helmholtz2dTol)
	}
}

func TestHelmholtzPlan3D_PositiveAlpha(t *testing.T) {
	nx, ny, nz := 24, 20, 16
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	hz := 1.0 / float64(nz+1)
	Lx := float64(nx+1) * hx
	Ly := float64(ny+1) * hy
	Lz := float64(nz+1) * hz
	alpha := 0.9

	plan, err := poisson.NewHelmholtzPlan(3, []int{nx, ny, nz}, []float64{hx, hy, hz}, []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet, poisson.Dirichlet}, alpha)
	if err != nil {
		t.Fatalf("NewHelmholtzPlan failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	plane := ny * nz
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := float64(j+1) * hy
			for k := range nz {
				z := float64(k+1) * hz
				u[i*plane+j*nz+k] = math.Sin(math.Pi*x/Lx)*math.Sin(math.Pi*y/Ly)*math.Sin(2.0*math.Pi*z/Lz) +
					0.1*math.Sin(2.0*math.Pi*x/Lx)*math.Sin(3.0*math.Pi*y/Ly)*math.Sin(math.Pi*z/Lz)
			}
		}
	}

	lap := make([]float64, nx*ny*nz)
	fd.Apply3D(lap, u, grid.Shape{nx, ny, nz}, [3]float64{hx, hy, hz}, [3]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet, poisson.Dirichlet})

	rhs := make([]float64, nx*ny*nz)
	for i := range rhs {
		rhs[i] = lap[i] + alpha*u[i]
	}

	got := make([]float64, nx*ny*nz)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > helmholtz3dTol {
		t.Fatalf("max error %g exceeds tol %g", max, helmholtz3dTol)
	}
}
