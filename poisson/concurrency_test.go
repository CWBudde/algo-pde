package poisson_test

import (
	"math"
	"sync"
	"testing"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// These tests pin the doc.go contract: plans are safe for concurrent Solve()
// calls. Every plan type runs N goroutines sharing ONE plan and compares each
// result against a serial reference computed from the same plan. Run with
// `go test -race` (just test-race) to catch unsynchronized workspace sharing.

const (
	concGoroutines = 8
	concIters      = 20
	concTol        = 1e-12
)

// concRHS returns a deterministic pseudo-random RHS in [-1, 1), distinct per seed.
func concRHS(size int, seed uint64) []float64 {
	rhs := make([]float64, size)
	s := seed*2654435761 + 12345
	for i := range rhs {
		s = s*6364136223846793005 + 1442695040888963407
		rhs[i] = float64(int64(s))/float64(math.MaxInt64) - 0.5
	}
	return rhs
}

type solveFunc func(dst, rhs []float64) error

// runConcurrentSolves computes serial references with solve, then hammers the
// same solve from concGoroutines goroutines and checks results still match.
func runConcurrentSolves(t *testing.T, size int, solve solveFunc) {
	t.Helper()

	rhss := make([][]float64, concGoroutines)
	refs := make([][]float64, concGoroutines)
	for gid := range rhss {
		rhss[gid] = concRHS(size, uint64(gid+1))
		refs[gid] = make([]float64, size)
		if err := solve(refs[gid], rhss[gid]); err != nil {
			t.Fatalf("serial reference solve %d: %v", gid, err)
		}
	}

	var waitGroup sync.WaitGroup
	for gid := range concGoroutines {
		waitGroup.Add(1)
		go func(gid int) {
			defer waitGroup.Done()
			dst := make([]float64, size)
			for it := range concIters {
				if err := solve(dst, rhss[gid]); err != nil {
					t.Errorf("goroutine %d iter %d: %v", gid, it, err)
					return
				}
				for i := range dst {
					if diff := math.Abs(dst[i] - refs[gid][i]); diff > concTol || math.IsNaN(dst[i]) {
						t.Errorf("goroutine %d iter %d: mismatch at index %d: got %gid, want %gid",
							gid, it, i, dst[i], refs[gid][i])
						return
					}
				}
			}
		}(gid)
	}
	waitGroup.Wait()
}

func TestConcurrentSolve_Periodic1D(t *testing.T) {
	n := 64
	plan, err := poisson.NewPlan1DPeriodic(n, 1.0/float64(n), poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan1DPeriodic: %v", err)
	}
	runConcurrentSolves(t, n, plan.Solve)
}

func TestConcurrentSolve_Periodic2D_Complex(t *testing.T) {
	nx, ny := 32, 48 // non-power-of-two ny exercises the out-of-place FFT path
	plan, err := poisson.NewPlan2DPeriodic(nx, ny, 1.0/float64(nx), 1.0/float64(ny),
		poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic: %v", err)
	}
	runConcurrentSolves(t, nx*ny, plan.Solve)
}

func TestConcurrentSolve_Periodic2D_RealFFT(t *testing.T) {
	nx, ny := 32, 64
	plan, err := poisson.NewPlan2DPeriodic(nx, ny, 1.0/float64(nx), 1.0/float64(ny),
		poisson.WithSubtractMean(), poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic: %v", err)
	}
	runConcurrentSolves(t, nx*ny, plan.Solve)
}

func TestConcurrentSolve_Periodic2D_Workers(t *testing.T) {
	nx, ny := 32, 32
	plan, err := poisson.NewPlan2DPeriodic(nx, ny, 1.0/float64(nx), 1.0/float64(ny),
		poisson.WithSubtractMean(), poisson.WithWorkers(4))
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic: %v", err)
	}
	runConcurrentSolves(t, nx*ny, plan.Solve)
}

func TestConcurrentSolve_Periodic2D_InPlace(t *testing.T) {
	nx, ny := 32, 32
	plan, err := poisson.NewPlan2DPeriodic(nx, ny, 1.0/float64(nx), 1.0/float64(ny),
		poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan2DPeriodic: %v", err)
	}
	runConcurrentSolves(t, nx*ny, func(dst, rhs []float64) error {
		copy(dst, rhs)
		return plan.SolveInPlace(dst)
	})
}

func TestConcurrentSolve_Periodic3D_Complex(t *testing.T) {
	nx, ny, nz := 16, 12, 10
	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz,
		1.0/float64(nx), 1.0/float64(ny), 1.0/float64(nz),
		poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic: %v", err)
	}
	runConcurrentSolves(t, nx*ny*nz, plan.Solve)
}

func TestConcurrentSolve_Periodic3D_RealFFT(t *testing.T) {
	nx, ny, nz := 16, 8, 16
	plan, err := poisson.NewPlan3DPeriodic(nx, ny, nz,
		1.0/float64(nx), 1.0/float64(ny), 1.0/float64(nz),
		poisson.WithSubtractMean(), poisson.WithRealFFT(true))
	if err != nil {
		t.Fatalf("NewPlan3DPeriodic: %v", err)
	}
	runConcurrentSolves(t, nx*ny*nz, plan.Solve)
}

func TestConcurrentSolve_PeriodicND(t *testing.T) {
	shape := grid.NewShapeND(4, 6, 3, 2) // non-power-of-two axes exercise scratch paths
	h := []float64{0.25, 1.0 / 6.0, 1.0 / 3.0, 0.5}
	plan, err := poisson.NewPlanNDPeriodic(shape, h, poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlanNDPeriodic: %v", err)
	}
	runConcurrentSolves(t, shape.Size(), plan.Solve)
}

func TestConcurrentSolve_Plan_Dirichlet2D(t *testing.T) {
	nx, ny := 32, 24
	plan, err := poisson.NewPlan(2,
		[]int{nx, ny},
		[]float64{1.0 / float64(nx+1), 1.0 / float64(ny+1)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	runConcurrentSolves(t, nx*ny, plan.Solve)
}

func TestConcurrentSolve_Plan_Neumann2D(t *testing.T) {
	nx, ny := 24, 32
	plan, err := poisson.NewPlan(2,
		[]int{nx, ny},
		[]float64{1.0 / float64(nx), 1.0 / float64(ny)},
		[]poisson.BCType{poisson.Neumann, poisson.Neumann},
		poisson.WithSubtractMean())
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	runConcurrentSolves(t, nx*ny, plan.Solve)
}

func TestConcurrentSolve_Plan_Mixed3D(t *testing.T) {
	nx, ny, nz := 12, 10, 8
	plan, err := poisson.NewPlan(3,
		[]int{nx, ny, nz},
		[]float64{1.0 / float64(nx+1), 1.0 / float64(ny), 1.0 / float64(nz)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Neumann, poisson.Periodic})
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	runConcurrentSolves(t, nx*ny*nz, plan.Solve)
}

func TestConcurrentSolve_Plan_QuarterWave2D(t *testing.T) {
	// Mixed DirichletNeumann × NeumannDirichlet axes (quarter-wave DST-IV/DCT-IV).
	// Extents avoid a factor of 5 so the 4N FFT is sound.
	nx, ny := 16, 24
	plan, err := poisson.NewPlan(2,
		[]int{nx, ny},
		[]float64{1.0 / float64(nx), 0.8 / float64(ny)},
		[]poisson.BCType{poisson.DirichletNeumann, poisson.NeumannDirichlet})
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	runConcurrentSolves(t, nx*ny, plan.Solve)
}

func TestConcurrentSolve_Plan_Helmholtz2D(t *testing.T) {
	nx, ny := 24, 24
	plan, err := poisson.NewHelmholtzPlan(2,
		[]int{nx, ny},
		[]float64{1.0 / float64(nx+1), 1.0 / float64(ny+1)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
		1.5)
	if err != nil {
		t.Fatalf("NewHelmholtzPlan: %v", err)
	}
	runConcurrentSolves(t, nx*ny, plan.Solve)
}

func TestConcurrentSolve_Plan_SolveWithBC(t *testing.T) {
	nx, ny := 24, 20
	plan, err := poisson.NewPlan(2,
		[]int{nx, ny},
		[]float64{1.0 / float64(nx+1), 1.0 / float64(ny+1)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}

	xLow := make([]float64, ny)
	xHigh := make([]float64, ny)
	for j := range xLow {
		xLow[j] = 0.25
		xHigh[j] = -0.5
	}
	bc := poisson.BoundaryConditions{
		{Face: poisson.XLow, Type: poisson.Dirichlet, Values: xLow},
		{Face: poisson.XHigh, Type: poisson.Dirichlet, Values: xHigh},
	}

	runConcurrentSolves(t, nx*ny, func(dst, rhs []float64) error {
		return plan.SolveWithBC(dst, rhs, bc)
	})
}
