package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// --- A.3.1 Helmholtz near-resonance -----------------------------------------

// A one-ulp perturbation off exact resonance still amplifies catastrophically,
// so the relative-tolerance gate must flag it rather than return a ~1e16 field.
func TestHelmholtz_NearResonantReturnsErrResonant(t *testing.T) {
	n := 64
	h := 1.0 / float64(n+1)
	eig0 := fd.EigenvaluesDirichlet(n, h)[0]

	// One ulp above exact resonance: denom at mode 0 is ~eig0*2^-52, far below
	// the term magnitudes, so the divide is hopelessly ill-conditioned.
	alpha := math.Nextafter(-eig0, math.Inf(1))

	plan, err := poisson.NewHelmholtzPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet}, alpha)
	if err != nil {
		t.Fatalf("NewHelmholtzPlan failed: %v", err)
	}

	rhs := make([]float64, n)
	for i := range rhs {
		rhs[i] = 1.0
	}
	dst := make([]float64, n)
	if err := plan.Solve(dst, rhs); !errors.Is(err, poisson.ErrResonant) {
		t.Fatalf("expected ErrResonant for near-resonant alpha, got %v", err)
	}
}

// A negative alpha that is comfortably clear of any eigenvalue is well
// conditioned and must solve, not be over-eagerly rejected as resonant.
func TestHelmholtz_NegativeNonResonantSolves(t *testing.T) {
	n := 64
	h := 1.0 / float64(n+1)
	L := float64(n+1) * h
	eig0 := fd.EigenvaluesDirichlet(n, h)[0]
	alpha := -0.5 * eig0 // between -eig0 and 0: shifts, never cancels

	plan, err := poisson.NewHelmholtzPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet}, alpha)
	if err != nil {
		t.Fatalf("NewHelmholtzPlan failed: %v", err)
	}

	u := make([]float64, n)
	for i := range n {
		x := float64(i+1) * h
		u[i] = math.Sin(math.Pi*x/L) + 0.25*math.Sin(4.0*math.Pi*x/L)
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
	if maxErr := maxAbsDiff(got, u); maxErr > 1e-10 {
		t.Fatalf("max error %g exceeds tol 1e-10", maxErr)
	}
}

// --- A.3.2 non-finite parameter validation ----------------------------------

func TestNewHelmholtzPlan_RejectsNonFiniteAlpha(t *testing.T) {
	for _, alpha := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err := poisson.NewHelmholtzPlan(1, []int{16}, []float64{0.1}, []poisson.BCType{poisson.Dirichlet}, alpha)
		if !errors.Is(err, poisson.ErrInvalidAlpha) {
			t.Fatalf("alpha=%v: expected ErrInvalidAlpha, got %v", alpha, err)
		}
	}
}

func TestConstructors_RejectNonFiniteSpacing(t *testing.T) {
	for _, h := range []float64{math.NaN(), math.Inf(1)} {
		constructors := map[string]error{
			"NewPlan":           errOf(poisson.NewPlan(1, []int{16}, []float64{h}, []poisson.BCType{poisson.Dirichlet})),
			"NewPlan1DPeriodic": errOf(poisson.NewPlan1DPeriodic(16, h)),
			"NewPlan2DPeriodic": errOf(poisson.NewPlan2DPeriodic(16, 16, h, 0.1)),
			"NewPlan3DPeriodic": errOf(poisson.NewPlan3DPeriodic(8, 8, 8, 0.1, h, 0.1)),
			"NewPlanNDPeriodic": errOf(poisson.NewPlanNDPeriodic(poisson.Shape{8, 8}, []float64{0.1, h})),
		}
		for name, err := range constructors {
			if !errors.Is(err, poisson.ErrInvalidSpacing) {
				t.Fatalf("%s h=%v: expected ErrInvalidSpacing, got %v", name, h, err)
			}
		}
	}
}

// --- A.3.3 nullspace mean gate ----------------------------------------------

// The default (NullspaceZeroMode) path must accept an analytically compatible
// inhomogeneous Neumann RHS whose discrete mean carries only O(h^2) quadrature
// error. Previously the fixed 1e-12 gate rejected it, forcing every caller onto
// WithSubtractMean.
func TestNeumann_DefaultZeroModeAcceptsQuadratureMean(t *testing.T) {
	n := 64
	h := 1.0 / float64(n)

	// A quadratic, analytically zero-mean RHS on [0,1]: integral of
	// (x-1/2)^2 - 1/12 is exactly zero, but the midpoint rule leaves an O(h^2)
	// discrete mean (unlike grid-resolvable trig, which integrates exactly).
	f := make([]float64, n)
	for i := range n {
		x := (float64(i) + 0.5) * h
		d := x - 0.5
		f[i] = d*d - 1.0/12.0
	}

	// Confirm the setup really has a non-trivial mean so the gate is exercised.
	if m := math.Abs(sliceMean(f)); m < 1e-8 {
		t.Fatalf("test setup: RHS mean %g too small to exercise the gate", m)
	}

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Neumann})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	got := make([]float64, n)
	if err := plan.Solve(got, f); err != nil {
		t.Fatalf("default Neumann solve rejected a compatible RHS: %v", err)
	}

	// Correctness independent of the (removed) constant mode: applying the
	// discrete operator to the solution reproduces f up to a constant vector,
	// so residual := A*got - f must be (numerically) constant.
	residual := make([]float64, n)
	fd.Apply1D(residual, got, h, poisson.Neumann)
	for i := range residual {
		residual[i] -= f[i]
	}
	rMean := sliceMean(residual)
	scale := 0.0
	for _, v := range f {
		if a := math.Abs(v); a > scale {
			scale = a
		}
	}
	for _, v := range residual {
		if dev := math.Abs(v - rMean); dev > 1e-9*scale {
			t.Fatalf("residual not constant: deviation %g exceeds tol %g", dev, 1e-9*scale)
		}
	}
}

// A genuinely inconsistent RHS (O(1) mean) must still be rejected.
func TestNeumann_DefaultZeroModeRejectsInconsistentMean(t *testing.T) {
	n := 64
	h := 1.0 / float64(n)
	L := float64(n) * h

	f := make([]float64, n)
	for i := range n {
		x := (float64(i) + 0.5) * h
		f[i] = (math.Pi/L)*(math.Pi/L)*math.Cos(math.Pi*x/L) + 0.5 // large DC offset
	}

	plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Neumann})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	dst := make([]float64, n)
	if err := plan.Solve(dst, f); !errors.Is(err, poisson.ErrNonZeroMean) {
		t.Fatalf("expected ErrNonZeroMean for O(1) mean, got %v", err)
	}
}

// --- A.3.4 float32 real-FFT contract ----------------------------------------

func TestWithFloat32_SinglePrecisionContract(t *testing.T) {
	const nx, ny = 64, 64 // power-of-two, even: qualifies for the real path
	hx := 1.0 / float64(nx)
	hy := 1.0 / float64(ny)
	Lx := float64(nx) * hx
	Ly := float64(ny) * hy

	rhs := make([]float64, nx*ny)
	for i := range nx {
		x := float64(i) * hx
		for j := range ny {
			y := float64(j) * hy
			rhs[i*ny+j] = math.Sin(2.0*math.Pi*x/Lx) * math.Sin(2.0*math.Pi*y/Ly)
		}
	}

	p64, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("float64 plan: %v", err)
	}
	p32, err := poisson.NewPlan2DPeriodic(nx, ny, hx, hy, poisson.WithFloat32(true))
	if err != nil {
		t.Fatalf("float32 plan: %v", err)
	}

	if p64.UsedRealFFT() {
		t.Fatalf("default plan unexpectedly took the float32 real-FFT path")
	}
	if !p32.UsedRealFFT() {
		t.Fatalf("WithFloat32 plan did not take the real-FFT path at qualifying sizes")
	}

	sol64 := make([]float64, nx*ny)
	sol32 := make([]float64, nx*ny)
	if err := p64.Solve(sol64, rhs); err != nil {
		t.Fatalf("float64 solve: %v", err)
	}
	if err := p32.Solve(sol32, rhs); err != nil {
		t.Fatalf("float32 solve: %v", err)
	}

	scale := 0.0
	for _, v := range sol64 {
		if a := math.Abs(v); a > scale {
			scale = a
		}
	}
	relDiff := maxAbsDiff(sol32, sol64) / scale

	// The float32 path must stay within its ~1e-6 contract...
	if relDiff > 1e-4 {
		t.Fatalf("float32 relative error %g exceeds the ~1e-6 contract (>1e-4)", relDiff)
	}
	// ...but must visibly differ from float64, proving it really is the lower
	// precision path and not silently identical.
	if relDiff < 1e-9 {
		t.Fatalf("float32 result matches float64 to %g; real path not exercised", relDiff)
	}
}

// --- A.3.5 NullspaceError fails at construction ------------------------------

func TestNullspaceError_RejectedAtConstruction(t *testing.T) {
	errOpt := poisson.WithNullspace(poisson.NullspaceError)

	// Plans that always carry the constant nullspace must reject NullspaceError
	// up front rather than failing on every Solve.
	rejected := map[string]error{
		"NewPlan1DPeriodic": errOf(poisson.NewPlan1DPeriodic(16, 0.1, errOpt)),
		"NewPlan2DPeriodic": errOf(poisson.NewPlan2DPeriodic(16, 16, 0.1, 0.1, errOpt)),
		"NewPlan3DPeriodic": errOf(poisson.NewPlan3DPeriodic(8, 8, 8, 0.1, 0.1, 0.1, errOpt)),
		"NewPlanNDPeriodic": errOf(poisson.NewPlanNDPeriodic(poisson.Shape{8, 8}, []float64{0.1, 0.1}, errOpt)),
		// Core Plan with a pure-Neumann axis also carries the nullspace.
		"NewPlan(Neumann)": errOf(poisson.NewPlan(1, []int{16}, []float64{0.1}, []poisson.BCType{poisson.Neumann}, errOpt)),
	}
	for name, err := range rejected {
		if !errors.Is(err, poisson.ErrNullspace) {
			t.Fatalf("%s: expected ErrNullspace at construction, got %v", name, err)
		}
	}

	// A Dirichlet problem has a unique solution, so NullspaceError is a no-op
	// and construction must succeed.
	dirichlet := []poisson.BCType{poisson.Dirichlet}
	if _, err := poisson.NewPlan(1, []int{16}, []float64{0.1}, dirichlet, errOpt); err != nil {
		t.Fatalf("NewPlan Dirichlet with NullspaceError should construct, got %v", err)
	}
}

// errOf discards a constructor's first return value, keeping only its error so
// several constructors with different result types can be checked uniformly.
func errOf[T any](_ T, err error) error {
	return err
}
