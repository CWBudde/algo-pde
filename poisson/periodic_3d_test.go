package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/cwbudde/algo-pde/fd"
	"github.com/cwbudde/algo-pde/grid"
	"github.com/cwbudde/algo-pde/poisson"
)

const (
	periodic3dTol     = 1e-10
	periodic3dRealTol = 1e-6
)

func TestNewPlan3DPeriodic_InvalidInputs(t *testing.T) {
	t.Parallel()

	_, err := poisson.NewPlan3DPeriodic(0, 4, 4, 1.0, 1.0, 1.0)
	if !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	_, err = poisson.NewPlan3DPeriodic(4, 0, 4, 1.0, 1.0, 1.0)
	if !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	_, err = poisson.NewPlan3DPeriodic(4, 4, 0, 1.0, 1.0, 1.0)
	if !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	_, err = poisson.NewPlan3DPeriodic(4, 4, 4, 0, 1.0, 1.0)
	if !errors.Is(err, poisson.ErrInvalidSpacing) {
		t.Fatalf("expected ErrInvalidSpacing, got %v", err)
	}
}

func TestPlan3DPeriodic_Solve_Manufactured(t *testing.T) {
	t.Parallel()

	nx, ny, nz := 24, 20, 18
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	hz := 1.0 / float64(nz)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy
	Lz := float64(nz) * hz

	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz, hx, hy, hz)
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := float64(i) * hx

		baseXY := i * ny * nz
		for j := range ny {
			y := float64(j) * hy

			base := baseXY + j*nz
			for k := range nz {
				z := float64(k) * hz
				u[base+k] = math.Sin(2.0*math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly) * math.Cos(4.0*math.Pi*z/Lz)
			}
		}
	}

	rhs := make([]float64, nx*ny*nz)
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Periodic, poisson.Periodic, poisson.Periodic,
	})

	got := make([]float64, nx*ny*nz)

	err = plan.Solve(got, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > periodic3dTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodic3dTol)
	}
}

func TestPlan3DPeriodic_Solve_RealFFT(t *testing.T) {
	t.Parallel()

	nx, ny, nz := 8, 8, 8
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	hz := 1.0 / float64(nz)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy
	Lz := float64(nz) * hz

	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz, hx, hy, hz, poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := float64(i) * hx

		baseXY := i * ny * nz
		for j := range ny {
			y := float64(j) * hy

			base := baseXY + j*nz
			for k := range nz {
				z := float64(k) * hz
				u[base+k] = math.Sin(2.0*math.Pi*x/Lx) * math.Cos(2.0*math.Pi*y/Ly) * math.Sin(2.0*math.Pi*z/Lz)
			}
		}
	}

	rhs := make([]float64, nx*ny*nz)
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Periodic, poisson.Periodic, poisson.Periodic,
	})

	got := make([]float64, nx*ny*nz)

	err = plan.Solve(got, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > periodic3dRealTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodic3dRealTol)
	}
}

func TestPlan3DPeriodic_NonZeroMean_Default(t *testing.T) {
	t.Parallel()

	nx, ny, nz := 6, 6, 6
	hx, hy, hz := 1.0, 1.0, 1.0

	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz, hx, hy, hz)
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	rhs := make([]float64, nx*ny*nz)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, nx*ny*nz)

	err = plan.Solve(dst, rhs)
	if !errors.Is(err, poisson.ErrNonZeroMean) {
		t.Fatalf("expected ErrNonZeroMean, got %v", err)
	}
}

func TestPlan3DPeriodic_SubtractMean(t *testing.T) {
	t.Parallel()

	nx, ny, nz := 6, 6, 6
	hx, hy, hz := 1.0, 1.0, 1.0

	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz, hx, hy, hz, poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	rhs := make([]float64, nx*ny*nz)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, nx*ny*nz)

	err = plan.Solve(dst, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	for i, v := range dst {
		if math.Abs(v) > periodic3dTol {
			t.Fatalf("dst[%d]=%g exceeds tol %g", i, v, periodic3dTol)
		}
	}
}

func TestPlan3DPeriodic_SetSolutionMean(t *testing.T) {
	t.Parallel()

	nx, ny, nz := 10, 8, 6
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	hz := 1.0 / float64(nz)
	targetMean := -0.75

	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz, hx, hy, hz, poisson.WithSolutionMean(targetMean))
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := float64(i) * hx

		baseXY := i * ny * nz
		for j := range ny {
			y := float64(j) * hy

			base := baseXY + j*nz
			for k := range nz {
				z := float64(k) * hz
				u[base+k] = math.Sin(2.0*math.Pi*x) * math.Cos(2.0*math.Pi*y) * math.Sin(2.0*math.Pi*z)
			}
		}
	}

	rhs := make([]float64, nx*ny*nz)
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Periodic, poisson.Periodic, poisson.Periodic,
	})

	dst := make([]float64, nx*ny*nz)

	err = plan.Solve(dst, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if mean := sliceMean(dst); math.Abs(mean-targetMean) > periodic3dTol {
		t.Fatalf("mean %g differs from target %g", mean, targetMean)
	}
}

func TestPlan3DPeriodic_SolveInPlace_MatchesSolve(t *testing.T) {
	t.Parallel()

	nx, ny, nz := 12, 10, 8
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	hz := 1.0 / float64(nz)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy
	Lz := float64(nz) * hz

	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz, hx, hy, hz)
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny*nz)
	for i := range nx {
		x := float64(i) * hx

		baseXY := i * ny * nz
		for j := range ny {
			y := float64(j) * hy

			base := baseXY + j*nz
			for k := range nz {
				z := float64(k) * hz
				u[base+k] = math.Sin(2.0*math.Pi*x/Lx) * math.Cos(2.0*math.Pi*y/Ly) * math.Sin(4.0*math.Pi*z/Lz)
			}
		}
	}

	rhs := make([]float64, nx*ny*nz)
	fd.Apply3D(rhs, u, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz}, [3]poisson.BCType{
		poisson.Periodic, poisson.Periodic, poisson.Periodic,
	})

	want := make([]float64, nx*ny*nz)
	err = plan.Solve(want, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	got := append([]float64(nil), rhs...)
	err = plan.SolveInPlace(got)
	if err != nil {
		t.Fatalf("SolveInPlace failed: %v", err)
	}

	if max := maxAbsDiff(got, want); max > periodic3dTol {
		t.Fatalf("max diff %g exceeds tol %g", max, periodic3dTol)
	}
}

func BenchmarkPlan3DPeriodic_Solve_32(b *testing.B)  { benchmarkPlan3DPeriodicSolve(b, 32) }
func BenchmarkPlan3DPeriodic_Solve_64(b *testing.B)  { benchmarkPlan3DPeriodicSolve(b, 64) }
func BenchmarkPlan3DPeriodic_Solve_128(b *testing.B) { benchmarkPlan3DPeriodicSolve(b, 128) }

func benchmarkPlan3DPeriodicSolve(b *testing.B, n int) {
	b.Helper()

	h := 1.0 / float64(n)

	plan, err := poisson.NewPlan3DPeriodic(n, n, n, h, h, h)
	if err != nil {
		b.Fatalf("NewPlan3DPeriodic failed: %v", err)
	}

	u := make([]float64, n*n*n)
	rhs := make([]float64, n*n*n)
	fd.Apply3D(rhs, u, grid.NewShape3D(n, n, n), [3]float64{h, h, h}, [3]poisson.BCType{
		poisson.Periodic, poisson.Periodic, poisson.Periodic,
	})

	dst := make([]float64, n*n*n)

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		err := plan.Solve(dst, rhs)
		if err != nil {
			b.Fatalf("Solve failed: %v", err)
		}
	}
}
