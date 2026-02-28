package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/cwbudde/algo-pde/fd"
	"github.com/cwbudde/algo-pde/poisson"
)

const periodic1dTol = 1e-10

func TestNewPlan1DPeriodic_InvalidInputs(t *testing.T) {
	t.Parallel()

	_, err := poisson.NewPlan1DPeriodic(0, 1.0)
	if !errors.Is(err, poisson.ErrInvalidSize) {
		t.Fatalf("expected ErrInvalidSize, got %v", err)
	}

	_, err = poisson.NewPlan1DPeriodic(4, 0)
	if !errors.Is(err, poisson.ErrInvalidSpacing) {
		t.Fatalf("expected ErrInvalidSpacing, got %v", err)
	}
}

func TestPlan1DPeriodic_Solve_Manufactured(t *testing.T) {
	t.Parallel()

	n := 64
	h := 1.0 / float64(n)
	L := float64(n) * h

	plan, err := poisson.NewPlan1DPeriodic(n, h)
	if err != nil {
		t.Fatalf("NewPlan1DPeriodic failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := float64(i) * h
		u[i] = math.Sin(2.0*math.Pi*x/L) + 0.25*math.Cos(4.0*math.Pi*x/L)
	}

	rhs := make([]float64, n)
	fd.Apply1D(rhs, u, h, poisson.Periodic)

	got := make([]float64, n)

	err = plan.Solve(got, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if max := maxAbsDiff(got, u); max > periodic1dTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodic1dTol)
	}
}

func TestPlan1DPeriodic_SolveInPlace(t *testing.T) {
	t.Parallel()

	n := 32
	h := 1.0 / float64(n)
	L := float64(n) * h

	plan, err := poisson.NewPlan1DPeriodic(n, h)
	if err != nil {
		t.Fatalf("NewPlan1DPeriodic failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := float64(i) * h
		u[i] = math.Sin(2.0*math.Pi*x/L) - 0.125*math.Cos(6.0*math.Pi*x/L)
	}

	buf := make([]float64, n)
	fd.Apply1D(buf, u, h, poisson.Periodic)

	err = plan.SolveInPlace(buf)
	if err != nil {
		t.Fatalf("SolveInPlace failed: %v", err)
	}

	if max := maxAbsDiff(buf, u); max > periodic1dTol {
		t.Fatalf("max error %g exceeds tol %g", max, periodic1dTol)
	}
}

func TestPlan1DPeriodic_NonZeroMean_Default(t *testing.T) {
	t.Parallel()

	n := 16
	h := 1.0

	plan, err := poisson.NewPlan1DPeriodic(n, h)
	if err != nil {
		t.Fatalf("NewPlan1DPeriodic failed: %v", err)
	}

	rhs := make([]float64, n)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, n)

	err = plan.Solve(dst, rhs)
	if !errors.Is(err, poisson.ErrNonZeroMean) {
		t.Fatalf("expected ErrNonZeroMean, got %v", err)
	}
}

func TestPlan1DPeriodic_SubtractMean(t *testing.T) {
	t.Parallel()

	n := 16
	h := 1.0

	plan, err := poisson.NewPlan1DPeriodic(n, h, poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan1DPeriodic failed: %v", err)
	}

	rhs := make([]float64, n)
	for i := range rhs {
		rhs[i] = 1.0
	}

	dst := make([]float64, n)

	err = plan.Solve(dst, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	for i, v := range dst {
		if math.Abs(v) > periodic1dTol {
			t.Fatalf("dst[%d]=%g exceeds tol %g", i, v, periodic1dTol)
		}
	}
}

func TestPlan1DPeriodic_SetSolutionMean(t *testing.T) {
	t.Parallel()

	n := 64
	h := 1.0 / float64(n)
	L := float64(n) * h
	targetMean := 2.5

	plan, err := poisson.NewPlan1DPeriodic(n, h, poisson.WithSolutionMean(targetMean))
	if err != nil {
		t.Fatalf("NewPlan1DPeriodic failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := float64(i) * h
		u[i] = math.Sin(2.0 * math.Pi * x / L)
	}

	rhs := make([]float64, n)
	fd.Apply1D(rhs, u, h, poisson.Periodic)

	dst := make([]float64, n)

	err = plan.Solve(dst, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	if mean := sliceMean(dst); math.Abs(mean-targetMean) > periodic1dTol {
		t.Fatalf("mean %g differs from target %g", mean, targetMean)
	}
}

func BenchmarkPlan1DPeriodic_Solve(b *testing.B) {
	n := 4096
	h := 1.0 / float64(n)
	L := float64(n) * h

	plan, err := poisson.NewPlan1DPeriodic(n, h)
	if err != nil {
		b.Fatalf("NewPlan1DPeriodic failed: %v", err)
	}

	rhs := make([]float64, n)
	for i := range n {
		x := float64(i) * h
		rhs[i] = math.Sin(2.0*math.Pi*x/L) + 0.5*math.Cos(6.0*math.Pi*x/L)
	}

	dst := make([]float64, n)

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		err := plan.Solve(dst, rhs)
		if err != nil {
			b.Fatalf("Solve failed: %v", err)
		}
	}
}

func maxAbsDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	max := 0.0

	for i := range a {
		diff := math.Abs(a[i] - b[i])
		if diff > max {
			max = diff
		}
	}

	return max
}

func sliceMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}

	return sum / float64(len(values))
}
