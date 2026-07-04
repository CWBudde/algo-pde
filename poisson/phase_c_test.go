package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

// --- Item 4: WithWorkers wired through PlanNDPeriodic ---

func TestPlanNDPeriodic_WithWorkersMatchesSerial(t *testing.T) {
	shape := poisson.Shape{8, 6, 10}
	h := []float64{1.0 / 8, 1.0 / 6, 1.0 / 10}

	size := shape[0] * shape[1] * shape[2]
	rhs := make([]float64, size)
	// Build a zero-mean RHS deterministically.
	for i := range rhs {
		rhs[i] = math.Sin(0.3*float64(i)) + math.Cos(0.17*float64(i))
	}
	mean := 0.0
	for _, v := range rhs {
		mean += v
	}
	mean /= float64(size)
	for i := range rhs {
		rhs[i] -= mean
	}

	serial, err := poisson.NewPlanNDPeriodic(shape, h, poisson.WithWorkers(1))
	if err != nil {
		t.Fatalf("serial NewPlanNDPeriodic: %v", err)
	}
	parallel, err := poisson.NewPlanNDPeriodic(shape, h, poisson.WithWorkers(4))
	if err != nil {
		t.Fatalf("parallel NewPlanNDPeriodic: %v", err)
	}

	gotSerial := make([]float64, size)
	if err := serial.Solve(gotSerial, rhs); err != nil {
		t.Fatalf("serial Solve: %v", err)
	}
	gotParallel := make([]float64, size)
	if err := parallel.Solve(gotParallel, rhs); err != nil {
		t.Fatalf("parallel Solve: %v", err)
	}

	for i := range gotSerial {
		if math.Abs(gotSerial[i]-gotParallel[i]) > 1e-12 {
			t.Fatalf("worker mismatch at %d: serial=%v parallel=%v", i, gotSerial[i], gotParallel[i])
		}
	}
}

// --- Item 4: WithSolutionMean requires a nullspace ---

func TestWithSolutionMean_NoNullspaceErrors(t *testing.T) {
	// Dirichlet has no nullspace, so WithSolutionMean must be rejected.
	_, err := poisson.NewPlan(
		1, []int{16}, []float64{1.0 / 17},
		[]poisson.BCType{poisson.Dirichlet},
		poisson.WithSolutionMean(0.5),
	)
	if !errors.Is(err, poisson.ErrSolutionMeanRequiresNullspace) {
		t.Fatalf("got %v, want ErrSolutionMeanRequiresNullspace", err)
	}

	// A nullspace plan (all-Neumann) accepts it.
	if _, err := poisson.NewPlan(
		1, []int{16}, []float64{1.0 / 16},
		[]poisson.BCType{poisson.Neumann},
		poisson.WithSolutionMean(0.5),
	); err != nil {
		t.Fatalf("Neumann plan with WithSolutionMean: unexpected error %v", err)
	}
}

// --- Item 4: WithRealFFT is unsupported on the general Plan ---

func TestWithRealFFT_GeneralPlanErrors(t *testing.T) {
	// Non-periodic BCs.
	_, err := poisson.NewPlan(
		2, []int{16, 16}, []float64{1.0 / 17, 1.0 / 17},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
		poisson.WithRealFFT(true),
	)
	if !errors.Is(err, poisson.ErrRealFFTUnsupported) {
		t.Fatalf("got %v, want ErrRealFFTUnsupported", err)
	}

	// WithFloat32 is an alias and must be rejected the same way.
	_, err = poisson.NewPlan(
		2, []int{16, 16}, []float64{1.0 / 17, 1.0 / 16},
		[]poisson.BCType{poisson.Dirichlet, poisson.Periodic},
		poisson.WithFloat32(true),
	)
	if !errors.Is(err, poisson.ErrRealFFTUnsupported) {
		t.Fatalf("got %v, want ErrRealFFTUnsupported", err)
	}

	// All-periodic BCs must ALSO be rejected: the general Plan never runs the
	// real-FFT path, so the option would be a silent no-op (regression test for
	// the Codex review on #7). Use NewPlan2DPeriodic to actually get real-FFT.
	_, err = poisson.NewPlan(
		2, []int{16, 16}, []float64{1.0 / 16, 1.0 / 16},
		[]poisson.BCType{poisson.Periodic, poisson.Periodic},
		poisson.WithRealFFT(true),
	)
	if !errors.Is(err, poisson.ErrRealFFTUnsupported) {
		t.Fatalf("all-periodic general Plan: got %v, want ErrRealFFTUnsupported", err)
	}
}

// --- Item 4: WithInPlace on periodic plans works ---

func TestPeriodicPlans_InPlaceSolve(t *testing.T) {
	n := 16
	h := 1.0 / float64(n)

	rhs := make([]float64, n)
	for i := range rhs {
		rhs[i] = math.Sin(2*math.Pi*float64(i)/float64(n)) - 0.0
	}

	plan, err := poisson.NewPlan1DPeriodic(n, h, poisson.WithInPlace(true))
	if err != nil {
		t.Fatalf("NewPlan1DPeriodic: %v", err)
	}

	// Reference via distinct buffers.
	want := make([]float64, n)
	if err := plan.Solve(want, rhs); err != nil {
		t.Fatalf("Solve: %v", err)
	}

	// In-place: dst aliases rhs.
	buf := make([]float64, n)
	copy(buf, rhs)
	if err := plan.SolveInPlace(buf); err != nil {
		t.Fatalf("SolveInPlace: %v", err)
	}

	for i := range want {
		if math.Abs(want[i]-buf[i]) > 1e-12 {
			t.Fatalf("in-place mismatch at %d: got %v want %v", i, buf[i], want[i])
		}
	}
}

// --- Item 5: duplicate faces rejected, bad face leaves rhs untouched ---

func TestSolveWithBC_DuplicateFaceRejected(t *testing.T) {
	nx, ny := 8, 8
	plan, err := poisson.NewPlan(
		2, []int{nx, ny}, []float64{1.0 / float64(nx+1), 1.0 / float64(ny+1)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
	)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}

	face := make([]float64, ny)
	bc := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: face},
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: face},
	}

	dst := make([]float64, nx*ny)
	rhs := make([]float64, nx*ny)
	if err := plan.SolveWithBC(dst, rhs, bc); !errors.Is(err, poisson.ErrDuplicateFace) {
		t.Fatalf("got %v, want ErrDuplicateFace", err)
	}
}

func TestSolveWithBC_InPlaceBadFaceLeavesRHSUnchanged(t *testing.T) {
	nx, ny := 8, 8
	plan, err := poisson.NewPlan(
		2, []int{nx, ny}, []float64{1.0 / float64(nx+1), 1.0 / float64(ny+1)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
		poisson.WithInPlace(true),
	)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}

	rhs := make([]float64, nx*ny)
	for i := range rhs {
		rhs[i] = float64(i) + 1
	}
	orig := make([]float64, nx*ny)
	copy(orig, rhs)

	// A face-values slice of the wrong length must be rejected before any
	// mutation of the caller's rhs.
	bc := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: make([]float64, ny+3)},
	}

	dst := make([]float64, nx*ny)
	if err := plan.SolveWithBC(dst, rhs, bc); err == nil {
		t.Fatal("expected error for bad face-values size, got nil")
	}

	for i := range rhs {
		if rhs[i] != orig[i] {
			t.Fatalf("rhs mutated at %d: got %v want %v", i, rhs[i], orig[i])
		}
	}
}

// --- Item 7: UsedRealFFT reflects the real-FFT fallback ---

func TestUsedRealFFT_FallbackReported(t *testing.T) {
	// Non-power-of-two sizes do not qualify for the real-FFT path: the plan must
	// fall back to the complex path and report UsedRealFFT()==false, without any
	// log output.
	fallback, err := poisson.NewPlan2DPeriodic(6, 6, 1.0/6, 1.0/6, poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic fallback: %v", err)
	}
	if fallback.UsedRealFFT() {
		t.Fatal("expected UsedRealFFT()==false for non-qualifying sizes")
	}

	// Qualifying power-of-two sizes take the real path.
	real2d, err := poisson.NewPlan2DPeriodic(32, 64, 1.0/32, 1.0/64, poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic real: %v", err)
	}
	if !real2d.UsedRealFFT() {
		t.Fatal("expected UsedRealFFT()==true for qualifying sizes")
	}

	// ND never uses the real path.
	nd, err := poisson.NewPlanNDPeriodic(poisson.Shape{8, 8}, []float64{1.0 / 8, 1.0 / 8}, poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("NewPlanNDPeriodic: %v", err)
	}
	if nd.UsedRealFFT() {
		t.Fatal("expected ND UsedRealFFT()==false")
	}
}
