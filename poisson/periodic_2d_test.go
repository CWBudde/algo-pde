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
	periodic2dTol     = 1e-10
	periodic2dRealTol = 1e-6
)

func TestNewPlan2DPeriodic_InvalidInputs(t *testing.T) {
	if _, err := poisson.NewPlan2DPeriodic(0, 4, 1.0, 1.0); !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	if _, err := poisson.NewPlan2DPeriodic(4, 0, 1.0, 1.0); !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	if _, err := poisson.NewPlan2DPeriodic(4, 4, 0, 1.0); !errors.Is(err, poisson.ErrInvalidSpacing) {
		t.Fatalf("expected ErrInvalidSpacing, got %v", err)
	}
}

func TestPlan2DPeriodic_Solve_Manufactured_SineSine(t *testing.T) {
	nx, ny := 64, 48
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		row := i * ny
		for j := range ny {
			y := float64(j) * hy
			u[row+j] = math.Sin(2.0*math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly)
		}
	}

	rhs := make([]float64, nx*ny)
	if err := fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{poisson.Periodic, poisson.Periodic}); err != nil {
		t.Fatal(err)
	}

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > periodic2dTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodic2dTol)
	}
}

func TestPlan2DPeriodic_Solve_Manufactured_CosCos(t *testing.T) {
	nx, ny := 48, 64
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		row := i * ny
		for j := range ny {
			y := float64(j) * hy
			u[row+j] = math.Cos(2.0*math.Pi*x/Lx) * math.Cos(4.0*math.Pi*y/Ly)
		}
	}

	rhs := make([]float64, nx*ny)
	if err := fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{poisson.Periodic, poisson.Periodic}); err != nil {
		t.Fatal(err)
	}

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > periodic2dTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodic2dTol)
	}
}

func TestPlan2DPeriodic_Solve_RealFFT(t *testing.T) {
	nx, ny := 32, 32
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy, poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		row := i * ny
		for j := range ny {
			y := float64(j) * hy
			u[row+j] = math.Sin(2.0*math.Pi*x/Lx) * math.Cos(2.0*math.Pi*y/Ly)
		}
	}

	rhs := make([]float64, nx*ny)
	if err := fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Periodic, poisson.Periodic,
	}); err != nil {
		t.Fatal(err)
	}

	got := make([]float64, nx*ny)
	if err := plan.Solve(got, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > periodic2dRealTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodic2dRealTol)
	}
}

func TestPlan2DPeriodic_Convergence(t *testing.T) {
	sizes := []int{16, 32, 64}
	errs := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n)
		hs[idx] = h
		plan, err := poisson.NewPlan2DPeriodic(n, n, h, h)
		if err != nil {
			t.Fatalf("NewPlan2DPeriodic failed: %v", err)
		}

		u := make([]float64, n*n)
		rhs := make([]float64, n*n)
		continuousRHSPeriodic2D(u, rhs, n, n, h, h)

		got := make([]float64, n*n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errs[idx] = maxAbsDiff(got, u)
	}

	// Strict second-order log-log rate check, matching convergence_test.go.
	checkConvergenceRates(t, hs, errs)
}

// TestPlan2DPeriodic_RealFFT_NyquistBin pins the Hermitian/Nyquist bin in the
// real-FFT (float32) periodic path. The RHS is dominated by the pure Nyquist
// mode (-1)^j along the reduced (y) axis, whose coefficient lives in the
// self-conjugate Nyquist bin (index ny/2) of the real transform. If that bin
// were mishandled the alternating mode would come back wrong, so we require the
// real solve to match the complex solve well inside the float32 contract.
func TestPlan2DPeriodic_RealFFT_NyquistBin(t *testing.T) {
	const nx, ny = 32, 32 // power-of-two, even ny: qualifies for the real path
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		row := i * ny
		for j := range ny {
			sign := 1.0
			if j%2 == 1 {
				sign = -1.0
			}
			u[row+j] = math.Cos(2.0*math.Pi*x/Lx) * sign
		}
	}

	rhs := make([]float64, nx*ny)
	if err := fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{
		poisson.Periodic, poisson.Periodic,
	}); err != nil {
		t.Fatal(err)
	}

	complexPlan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("complex plan: %v", err)
	}
	realPlan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy, poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("real plan: %v", err)
	}
	if !realPlan.UsedRealFFT() {
		t.Fatalf("real plan did not take the real-FFT path at qualifying sizes")
	}

	solComplex := make([]float64, nx*ny)
	solReal := make([]float64, nx*ny)
	if err := complexPlan.Solve(solComplex, rhs); err != nil {
		t.Fatalf("complex solve: %v", err)
	}
	if err := realPlan.Solve(solReal, rhs); err != nil {
		t.Fatalf("real solve: %v", err)
	}

	// The complex path must recover the manufactured Nyquist-dominated field.
	if e := maxAbsDiff(solComplex, u); e > periodic2dTol {
		t.Fatalf("complex solve failed to recover Nyquist mode: max error %g", e)
	}

	scale := 0.0
	for _, v := range solComplex {
		if a := math.Abs(v); a > scale {
			scale = a
		}
	}
	if rel := maxAbsDiff(solReal, solComplex) / scale; rel > 1e-4 {
		t.Fatalf("real vs complex Nyquist-bin mismatch: rel error %g exceeds 1e-4", rel)
	}
}

func TestPlan2DPeriodic_NonZeroMean_Default(t *testing.T) {
	nx, ny := 8, 8
	hx := 1.0
	hy := 1.0

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	rhs := make([]float64, nx*ny)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, nx*ny)
	if err := plan.Solve(dst, rhs); !errors.Is(err, poisson.ErrNonZeroMean) {
		t.Fatalf("expected ErrNonZeroMean, got %v", err)
	}
}

func TestPlan2DPeriodic_SubtractMean(t *testing.T) {
	nx, ny := 8, 8
	hx := 1.0
	hy := 1.0

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy, poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	rhs := make([]float64, nx*ny)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, nx*ny)
	if err := plan.Solve(dst, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	for i, v := range dst {
		if math.Abs(v) > periodic2dTol {
			t.Fatalf("dst[%d]=%g exceeds tol %g", i, v, periodic2dTol)
		}
	}
}

func TestPlan2DPeriodic_SetSolutionMean(t *testing.T) {
	nx, ny := 32, 32
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	targetMean := 1.25

	plan, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy, poisson.WithSolutionMean(targetMean))
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	u := make([]float64, nx*ny)
	rhs := make([]float64, nx*ny)
	continuousRHSPeriodic2D(u, rhs, nx, ny, hx, hy)

	dst := make([]float64, nx*ny)
	if err := plan.Solve(dst, rhs); err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if mean := sliceMean(dst); math.Abs(mean-targetMean) > periodic2dTol {
		t.Fatalf("mean %g differs from target %g", mean, targetMean)
	}
}

func BenchmarkPlan2DPeriodic_Solve_64(b *testing.B)  { benchmarkPlan2DPeriodicSolve(b, 64) }
func BenchmarkPlan2DPeriodic_Solve_128(b *testing.B) { benchmarkPlan2DPeriodicSolve(b, 128) }
func BenchmarkPlan2DPeriodic_Solve_256(b *testing.B) { benchmarkPlan2DPeriodicSolve(b, 256) }
func BenchmarkPlan2DPeriodic_Solve_512(b *testing.B) { benchmarkPlan2DPeriodicSolve(b, 512) }
func BenchmarkPlan2DPeriodic_Solve_1024(b *testing.B) {
	benchmarkPlan2DPeriodicSolve(b, 1024)
}

func benchmarkPlan2DPeriodicSolve(b *testing.B, n int) {
	b.Helper()
	h := 1.0 / float64(n)
	plan, err := poisson.NewPlan2DPeriodic(n, n, h, h)
	if err != nil {
		b.Fatalf("NewPlan2DPeriodic failed: %v", err)
	}

	rhs := make([]float64, n*n)
	u := make([]float64, n*n)
	if err := fd.Apply2D(rhs, u, grid.NewShape2D(n, n), [2]float64{h, h}, [2]poisson.BCType{poisson.Periodic, poisson.Periodic}); err != nil {
		b.Fatal(err)
	}

	dst := make([]float64, n*n)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := plan.Solve(dst, rhs); err != nil {
			b.Fatalf("Solve failed: %v", err)
		}
	}
}

func continuousRHSPeriodic2D(u, rhs []float64, nx, ny int, hx, hy float64) {
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy
	kx := 2.0 * math.Pi / Lx
	ky := 2.0 * math.Pi / Ly
	factor := kx*kx + ky*ky

	for i := range nx {
		x := float64(i) * hx
		row := i * ny
		for j := range ny {
			y := float64(j) * hy
			uval := math.Sin(kx*x) * math.Sin(ky*y)
			u[row+j] = uval
			rhs[row+j] = factor * uval
		}
	}
}
