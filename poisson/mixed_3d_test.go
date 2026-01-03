package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const mixed3dTol = 1e-10

func TestPlan3D_AllBCCombinations(t *testing.T) {
	nx, ny, nz := 8, 7, 6

	bcs := []poisson.BCType{poisson.Periodic, poisson.Dirichlet, poisson.Neumann}
	for _, bcx := range bcs {
		for _, bcy := range bcs {
			for _, bcz := range bcs {
				hx, fx := axisBasis(bcx, nx)
				hy, fy := axisBasis(bcy, ny)
				hz, fz := axisBasis(bcz, nz)

				plan, err := poisson.NewPlan(
					3,
					[]int{nx, ny, nz},
					[]float64{hx, hy, hz},
					[]poisson.BCType{bcx, bcy, bcz},
				)
				if err != nil {
					t.Fatalf("NewPlan failed for %v/%v/%v: %v", bcx, bcy, bcz, err)
				}

				u := make([]float64, nx*ny*nz)
				for i := range nx {
					for j := range ny {
						for k := range nz {
							u[(i*ny+j)*nz+k] = fx[i] * fy[j] * fz[k]
						}
					}
				}

				rhs := make([]float64, len(u))
				fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
					bcx, bcy, bcz,
				})

				got := make([]float64, len(u))
				if err := plan.Solve(got, rhs); err != nil {
					t.Fatalf("Solve failed for %v/%v/%v: %v", bcx, bcy, bcz, err)
				}

				if max := maxAbsDiff(got, u); max > mixed3dTol {
					t.Fatalf("max error %g exceeds tol %g for %v/%v/%v", max, mixed3dTol, bcx, bcy, bcz)
				}
			}
		}
	}
}

func TestPlan3D_DirichletDirichletDirichlet(t *testing.T) {
	nx, ny, nz := 24, 18, 16
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny+1)
	hz := 1.0 / float64(nz+1)

	plan, err := poisson.NewPlan(
		3,
		[]int{nx, ny, nz},
		[]float64{hx, hy, hz},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet, poisson.Dirichlet},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := float64(j+1) * hy
			for k := range nz {
				z := float64(k+1) * hz
				u[(i*ny+j)*nz+k] = math.Sin(math.Pi*x) * math.Sin(2.0*math.Pi*y) * math.Sin(3.0*math.Pi*z)
			}
		}
	}

	rhs := make([]float64, len(u))
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Dirichlet, poisson.Dirichlet, poisson.Dirichlet,
	})

	got := make([]float64, len(u))
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed3dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed3dTol)
	}
}

func TestPlan3D_NeumannNeumannNeumann(t *testing.T) {
	nx, ny, nz := 22, 16, 14
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	hz := 1.0 / float64(nz)

	plan, err := poisson.NewPlan(
		3,
		[]int{nx, ny, nz},
		[]float64{hx, hy, hz},
		[]poisson.BCType{poisson.Neumann, poisson.Neumann, poisson.Neumann},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := (float64(i) + 0.5) * hx
		for j := range ny {
			y := (float64(j) + 0.5) * hy
			for k := range nz {
				z := (float64(k) + 0.5) * hz
				u[(i*ny+j)*nz+k] = math.Cos(math.Pi*x) * math.Cos(2.0*math.Pi*y) * math.Cos(math.Pi*z)
			}
		}
	}

	rhs := make([]float64, len(u))
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Neumann, poisson.Neumann, poisson.Neumann,
	})

	got := make([]float64, len(u))
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed3dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed3dTol)
	}
}

func TestPlan3D_PeriodicDirichletNeumann(t *testing.T) {
	nx, ny, nz := 20, 14, 18
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny+1)
	hz := 1.0 / float64(nz)

	plan, err := poisson.NewPlan(
		3,
		[]int{nx, ny, nz},
		[]float64{hx, hy, hz},
		[]poisson.BCType{poisson.Periodic, poisson.Dirichlet, poisson.Neumann},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := float64(i) * hx
		for j := range ny {
			y := float64(j+1) * hy
			for k := range nz {
				z := (float64(k) + 0.5) * hz
				u[(i*ny+j)*nz+k] = math.Sin(2.0*math.Pi*x) * math.Sin(math.Pi*y) * math.Cos(2.0*math.Pi*z)
			}
		}
	}

	rhs := make([]float64, len(u))
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Periodic, poisson.Dirichlet, poisson.Neumann,
	})

	got := make([]float64, len(u))
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed3dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed3dTol)
	}
}

func TestPlan3D_DirichletPeriodicNeumann(t *testing.T) {
	nx, ny, nz := 18, 22, 16
	hx := 1.0 / float64(nx+1)
	hy := 1.0 / float64(ny)
	hz := 1.0 / float64(nz)

	plan, err := poisson.NewPlan(
		3,
		[]int{nx, ny, nz},
		[]float64{hx, hy, hz},
		[]poisson.BCType{poisson.Dirichlet, poisson.Periodic, poisson.Neumann},
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := float64(i+1) * hx
		for j := range ny {
			y := float64(j) * hy
			for k := range nz {
				z := (float64(k) + 0.5) * hz
				u[(i*ny+j)*nz+k] = math.Sin(math.Pi*x) * math.Sin(2.0*math.Pi*y) * math.Cos(math.Pi*z)
			}
		}
	}

	rhs := make([]float64, len(u))
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Dirichlet, poisson.Periodic, poisson.Neumann,
	})

	got := make([]float64, len(u))
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > mixed3dTol {
		t.Fatalf("max error %g exceeds tol %g", max, mixed3dTol)
	}
}

func axisBasis(bc poisson.BCType, n int) (float64, []float64) {
	values := make([]float64, n)
	switch bc {
	case poisson.Dirichlet:
		h := 1.0 / float64(n+1)
		for i := range n {
			x := float64(i+1) * h
			values[i] = math.Sin(math.Pi * x)
		}
		return h, values
	case poisson.Neumann:
		h := 1.0 / float64(n)
		for i := range n {
			x := (float64(i) + 0.5) * h
			values[i] = math.Cos(math.Pi * x)
		}
		return h, values
	default:
		h := 1.0 / float64(n)
		for i := range n {
			x := float64(i) * h
			values[i] = math.Sin(2.0 * math.Pi * x)
		}
		return h, values
	}
}
