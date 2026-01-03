package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const (
	manufactured1DTol = 1e-10
	manufactured2DTol = 1e-9
	manufactured3DTol = 1e-8
)

func TestManufactured1D(t *testing.T) {
	t.Run("Periodic", func(t *testing.T) {
		n := 64
		h := 1.0 / float64(n)
		L := float64(n) * h

		u := make([]float64, n)
		for i := range n {
			x := float64(i) * h
			u[i] = math.Sin(2.0 * math.Pi * x / L)
		}

		meanU := sliceMean(u)
		solveAndCompare1D(
			t,
			n,
			h,
			poisson.Periodic,
			u,
			manufactured1DTol,
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
	})

	t.Run("Dirichlet", func(t *testing.T) {
		n := 64
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h

		u := make([]float64, n)
		for i := range n {
			x := float64(i+1) * h
			u[i] = math.Sin(math.Pi * x / L)
		}

		solveAndCompare1D(t, n, h, poisson.Dirichlet, u, manufactured1DTol)
	})

	t.Run("Neumann", func(t *testing.T) {
		n := 64
		h := 1.0 / float64(n)
		L := float64(n) * h

		u := make([]float64, n)
		for i := range n {
			x := (float64(i) + 0.5) * h
			u[i] = math.Cos(math.Pi*x/L) + x
		}

		meanU := sliceMean(u)
		solveAndCompare1D(
			t,
			n,
			h,
			poisson.Neumann,
			u,
			manufactured1DTol,
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
	})
}

func TestManufactured2D(t *testing.T) {
	t.Run("Periodic", func(t *testing.T) {
		nx, ny := 48, 40
		hx := 1.0 / float64(nx)
		hy := 1.0 / float64(ny)
		Lx := float64(nx) * hx
		Ly := float64(ny) * hy

		u := make([]float64, nx*ny)
		for i := range nx {
			x := float64(i) * hx
			for j := range ny {
				y := float64(j) * hy
				u[i*ny+j] = math.Sin(2.0*math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly)
			}
		}

		meanU := sliceMean(u)
		solveAndCompare2D(
			t,
			nx,
			ny,
			hx,
			hy,
			[2]poisson.BCType{poisson.Periodic, poisson.Periodic},
			u,
			manufactured2DTol,
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
	})

	t.Run("Dirichlet", func(t *testing.T) {
		nx, ny := 32, 36
		hx := 1.0 / float64(nx+1)
		hy := 1.0 / float64(ny+1)
		Lx := float64(nx+1) * hx
		Ly := float64(ny+1) * hy

		u := make([]float64, nx*ny)
		for i := range nx {
			x := float64(i+1) * hx
			for j := range ny {
				y := float64(j+1) * hy
				u[i*ny+j] = math.Sin(math.Pi*x/Lx) * math.Sin(math.Pi*y/Ly)
			}
		}

		solveAndCompare2D(
			t,
			nx,
			ny,
			hx,
			hy,
			[2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
			u,
			manufactured2DTol,
		)
	})

	t.Run("Neumann", func(t *testing.T) {
		nx, ny := 32, 36
		hx := 1.0 / float64(nx)
		hy := 1.0 / float64(ny)
		Lx := float64(nx) * hx
		Ly := float64(ny) * hy

		u := make([]float64, nx*ny)
		for i := range nx {
			x := (float64(i) + 0.5) * hx
			for j := range ny {
				y := (float64(j) + 0.5) * hy
				u[i*ny+j] = math.Cos(math.Pi*x/Lx) * math.Cos(math.Pi*y/Ly)
			}
		}

		meanU := sliceMean(u)
		solveAndCompare2D(
			t,
			nx,
			ny,
			hx,
			hy,
			[2]poisson.BCType{poisson.Neumann, poisson.Neumann},
			u,
			manufactured2DTol,
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
	})

	t.Run("MixedPeriodicNeumann", func(t *testing.T) {
		nx, ny := 36, 30
		hx := 1.0 / float64(nx)
		hy := 1.0 / float64(ny)
		Lx := float64(nx) * hx
		Ly := float64(ny) * hy

		u := make([]float64, nx*ny)
		for i := range nx {
			x := float64(i) * hx
			for j := range ny {
				y := (float64(j) + 0.5) * hy
				u[i*ny+j] = math.Sin(2.0*math.Pi*x/Lx) * math.Cos(math.Pi*y/Ly)
			}
		}

		meanU := sliceMean(u)
		solveAndCompare2D(
			t,
			nx,
			ny,
			hx,
			hy,
			[2]poisson.BCType{poisson.Periodic, poisson.Neumann},
			u,
			manufactured2DTol,
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
	})
}

func TestManufactured3D(t *testing.T) {
	t.Run("Periodic", func(t *testing.T) {
		n := 24
		h := 1.0 / float64(n)
		L := float64(n) * h

		u := make([]float64, n*n*n)
		for i := range n {
			x := float64(i) * h
			for j := range n {
				y := float64(j) * h
				for k := range n {
					z := float64(k) * h
					u[(i*n+j)*n+k] = math.Sin(2.0*math.Pi*x/L) *
						math.Sin(2.0*math.Pi*y/L) *
						math.Sin(2.0*math.Pi*z/L)
				}
			}
		}

		meanU := sliceMean(u)
		solveAndCompare3D(
			t,
			n,
			n,
			n,
			h,
			h,
			h,
			[3]poisson.BCType{poisson.Periodic, poisson.Periodic, poisson.Periodic},
			u,
			manufactured3DTol,
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
	})

	t.Run("Dirichlet", func(t *testing.T) {
		n := 20
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h

		u := make([]float64, n*n*n)
		for i := range n {
			x := float64(i+1) * h
			for j := range n {
				y := float64(j+1) * h
				for k := range n {
					z := float64(k+1) * h
					u[(i*n+j)*n+k] = math.Sin(math.Pi*x/L) *
						math.Sin(math.Pi*y/L) *
						math.Sin(math.Pi*z/L)
				}
			}
		}

		solveAndCompare3D(
			t,
			n,
			n,
			n,
			h,
			h,
			h,
			[3]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet, poisson.Dirichlet},
			u,
			manufactured3DTol,
		)
	})

	t.Run("Neumann", func(t *testing.T) {
		n := 20
		h := 1.0 / float64(n)
		L := float64(n) * h

		u := make([]float64, n*n*n)
		for i := range n {
			x := (float64(i) + 0.5) * h
			for j := range n {
				y := (float64(j) + 0.5) * h
				for k := range n {
					z := (float64(k) + 0.5) * h
					u[(i*n+j)*n+k] = math.Cos(math.Pi*x/L) *
						math.Cos(math.Pi*y/L) *
						math.Cos(math.Pi*z/L)
				}
			}
		}

		meanU := sliceMean(u)
		solveAndCompare3D(
			t,
			n,
			n,
			n,
			h,
			h,
			h,
			[3]poisson.BCType{poisson.Neumann, poisson.Neumann, poisson.Neumann},
			u,
			manufactured3DTol,
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
	})

	t.Run("MixedPeriodicDirichletNeumann", func(t *testing.T) {
		nx, ny, nz := 24, 20, 18
		hx := 1.0 / float64(nx)
		hy := 1.0 / float64(ny+1)
		hz := 1.0 / float64(nz)
		Lx := float64(nx) * hx
		Ly := float64(ny+1) * hy
		Lz := float64(nz) * hz

		u := make([]float64, nx*ny*nz)
		for i := range nx {
			x := float64(i) * hx
			for j := range ny {
				y := float64(j+1) * hy
				for k := range nz {
					z := (float64(k) + 0.5) * hz
					u[(i*ny+j)*nz+k] = math.Sin(2.0*math.Pi*x/Lx) *
						math.Sin(math.Pi*y/Ly) *
						math.Cos(math.Pi*z/Lz)
				}
			}
		}

		solveAndCompare3D(
			t,
			nx,
			ny,
			nz,
			hx,
			hy,
			hz,
			[3]poisson.BCType{poisson.Periodic, poisson.Dirichlet, poisson.Neumann},
			u,
			manufactured3DTol,
		)
	})
}

func solveAndCompare1D(
	t *testing.T,
	n int,
	h float64,
	bc poisson.BCType,
	u []float64,
	tol float64,
	opts ...poisson.Option,
) {
	t.Helper()

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{bc}, opts...)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, n)
	fd.Apply1D(rhs, u, h, bc)

	got := make([]float64, n)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > tol {
		t.Fatalf("max error %g exceeds tol %g", max, tol)
	}
}

func solveAndCompare2D(
	t *testing.T,
	nx int,
	ny int,
	hx float64,
	hy float64,
	bc [2]poisson.BCType,
	u []float64,
	tol float64,
	opts ...poisson.Option,
) {
	t.Helper()

	plan, err := poisson.NewPlan(
		2,
		[]int{nx, ny},
		[]float64{hx, hy},
		[]poisson.BCType{bc[0], bc[1]},
		opts...,
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, nx*ny)
	fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, bc)

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > tol {
		t.Fatalf("max error %g exceeds tol %g", max, tol)
	}
}

func solveAndCompare3D(
	t *testing.T,
	nx int,
	ny int,
	nz int,
	hx float64,
	hy float64,
	hz float64,
	bc [3]poisson.BCType,
	u []float64,
	tol float64,
	opts ...poisson.Option,
) {
	t.Helper()

	plan, err := poisson.NewPlan(
		3,
		[]int{nx, ny, nz},
		[]float64{hx, hy, hz},
		[]poisson.BCType{bc[0], bc[1], bc[2]},
		opts...,
	)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, nx*ny*nz)
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, bc)

	got := make([]float64, nx*ny*nz)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > tol {
		t.Fatalf("max error %g exceeds tol %g", max, tol)
	}
}
