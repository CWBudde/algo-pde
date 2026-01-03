package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const mixed2dTol = 1e-10

func TestPlan2D_DirichletDirichlet(t *testing.T) {
	nx, ny := 48, 40
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := float64(j+1) * hy
			u[i*ny+j] = math.Sin(math.Pi*x) * math.Sin(2.0*math.Pi*y)
		}
	}

	rhs := make([]float64, nx*ny)
	fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Dirichlet, poisson.Dirichlet,
	})

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed2dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed2dTol)
	}
}

func TestPlan2D_NeumannNeumann(t *testing.T) {
	nx, ny := 56, 44
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Neumann, poisson.Neumann},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := (float64(i) + 0.5) * hx
		for j := range ny {
			y := (float64(j) + 0.5) * hy
			u[i*ny+j] = math.Cos(math.Pi*x) * math.Cos(2.0*math.Pi*y)
		}
	}

	rhs := make([]float64, nx*ny)
	fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Neumann, poisson.Neumann,
	})

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed2dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed2dTol)
	}
}

func TestPlan2D_PeriodicDirichlet(t *testing.T) {
	nx, ny := 64, 36
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny+1)

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{poisson.Periodic, poisson.Dirichlet},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		for j := range ny {
			y := float64(j+1) * hy
			u[i*ny+j] = math.Sin(2.0*math.Pi*x) * math.Sin(math.Pi*y)
		}
	}

	rhs := make([]float64, nx*ny)
	fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Periodic, poisson.Dirichlet,
	})

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed2dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed2dTol)
	}
}

func TestPlan2D_DirichletNeumann(t *testing.T) {
	nx, ny := 52, 40
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny)

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
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := (float64(j) + 0.5) * hy
			u[i*ny+j] = math.Sin(math.Pi*x) * math.Cos(math.Pi*y)
		}
	}

	rhs := make([]float64, nx*ny)
	fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Dirichlet, poisson.Neumann,
	})

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed2dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed2dTol)
	}
}
