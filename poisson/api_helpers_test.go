package poisson_test

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-pde/fd"
	"github.com/cwbudde/algo-pde/grid"
	"github.com/cwbudde/algo-pde/poisson"
)

func TestPlan_SolveInPlace_MatchesSolve(t *testing.T) {
	t.Parallel()

	nx, ny := 24, 20
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy

	plan, err := poisson.NewPlan(2, []int{nx, ny}, []float64{hx, hy}, []poisson.BCType{poisson.Periodic, poisson.Periodic})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	u := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		base := i * ny

		for j := range ny {
			y := float64(j) * hy
			u[base+j] = math.Sin(2.0*math.Pi*x/Lx) * math.Cos(2.0*math.Pi*y/Ly)
		}
	}

	rhs := make([]float64, nx*ny)
	fd.Apply2D(rhs, u, grid.NewShape2D(nx, ny), [2]float64{hx, hy}, [2]poisson.BCType{poisson.Periodic, poisson.Periodic})

	want := make([]float64, nx*ny)
	err = plan.Solve(want, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	got := append([]float64(nil), rhs...)
	err = plan.SolveInPlace(got)
	if err != nil {
		t.Fatalf("SolveInPlace failed: %v", err)
	}

	if max := maxAbsDiffFloats(got, want); max > 1e-10 {
		t.Fatalf("max diff %g exceeds tolerance", max)
	}
}

func TestPlan_WorkBytes_RespectsInPlaceOption(t *testing.T) {
	t.Parallel()

	nx, ny := 4, 5
	size := nx * ny

	defaultPlan, err := poisson.NewPlan(2, []int{nx, ny}, []float64{1, 1}, []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan default failed: %v", err)
	}

	inPlacePlan, err := poisson.NewPlan(2, []int{nx, ny}, []float64{1, 1}, []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, poisson.WithInPlace(true))
	if err != nil {
		t.Fatalf("NewPlan in-place failed: %v", err)
	}

	if got, want := defaultPlan.WorkBytes(), size*24; got != want {
		t.Fatalf("default WorkBytes = %d, want %d", got, want)
	}

	if got, want := inPlacePlan.WorkBytes(), size*16; got != want {
		t.Fatalf("in-place WorkBytes = %d, want %d", got, want)
	}
}

func TestOptions_ConstructorsAndApply(t *testing.T) {
	t.Parallel()

	opts := poisson.ApplyOptions(poisson.DefaultOptions(), []poisson.Option{
		poisson.WithNullspace(poisson.NullspaceError),
		poisson.WithWorkers(3),
		poisson.WithInPlace(true),
	})

	if opts.Nullspace != poisson.NullspaceError {
		t.Fatalf("Nullspace = %v, want %v", opts.Nullspace, poisson.NullspaceError)
	}

	if opts.Workers != 3 {
		t.Fatalf("Workers = %d, want 3", opts.Workers)
	}

	if !opts.InPlace {
		t.Fatalf("InPlace = false, want true")
	}
}

func TestShape_DimAndN(t *testing.T) {
	t.Parallel()

	s := poisson.Shape{3, 4, 5}
	if got := s.Dim(); got != 3 {
		t.Fatalf("Dim = %d, want 3", got)
	}

	if got := s.N(1); got != 4 {
		t.Fatalf("N(1) = %d, want 4", got)
	}

	if got := s.N(-1); got != 0 {
		t.Fatalf("N(-1) = %d, want 0", got)
	}

	if got := s.N(3); got != 0 {
		t.Fatalf("N(3) = %d, want 0", got)
	}
}

func TestWorkspace_Bytes(t *testing.T) {
	t.Parallel()

	w := poisson.NewWorkspace(7, 11)
	if got, want := (&w).Bytes(), 7*8+11*16; got != want {
		t.Fatalf("Bytes = %d, want %d", got, want)
	}
}

func TestFFTPlan_Len(t *testing.T) {
	t.Parallel()

	plan, err := poisson.NewFFTPlan(9)
	if err != nil {
		t.Fatalf("NewFFTPlan failed: %v", err)
	}

	if got := plan.Len(); got != 9 {
		t.Fatalf("Len = %d, want 9", got)
	}
}

func maxAbsDiffFloats(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	maxDiff := 0.0
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if d > maxDiff {
			maxDiff = d
		}
	}

	return maxDiff
}
