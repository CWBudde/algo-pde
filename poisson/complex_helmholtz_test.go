package poisson_test

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Tech/algo-pde/bc"
	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// TestSolveComplex_Residual2D solves (alpha - Δ)u = f for a complex alpha across
// every boundary condition and verifies the complex residual is ~0. Because
// fd.Apply2D applies the NEGATIVE Laplacian (-Δ), the residual is
// alpha*u + lap - f, evaluated on the real and imaginary parts of u separately.
func TestSolveComplex_Residual2D(t *testing.T) {
	const (
		nx, ny = 12, 10
		hx, hy = 1.0 / 13, 1.0 / 11
	)
	alpha := complex(-30, 5)

	cases := [][2]poisson.BCType{
		{poisson.Neumann, poisson.Neumann},
		{poisson.Periodic, poisson.Periodic},
		{poisson.Dirichlet, poisson.Dirichlet},
		{poisson.Dirichlet, poisson.Neumann},
	}

	seed := int64(1)
	for _, bcPair := range cases {
		name := bcPair[0].String() + "_" + bcPair[1].String()
		t.Run(name, func(t *testing.T) {
			n := nx * ny
			f := randomField(seed, n)
			seed++

			plan, err := poisson.NewComplexHelmholtzPlan(
				2, []int{nx, ny}, []float64{hx, hy},
				[]poisson.BCType{bcPair[0], bcPair[1]}, alpha,
			)
			if err != nil {
				t.Fatalf("plan: %v", err)
			}

			u := make([]complex128, n)
			if err := plan.SolveComplex(u, f); err != nil {
				t.Fatalf("SolveComplex: %v", err)
			}

			re := make([]float64, n)
			im := make([]float64, n)
			for i, v := range u {
				re[i] = real(v)
				im[i] = imag(v)
			}

			lapRe := make([]float64, n)
			lapIm := make([]float64, n)
			shape := grid.NewShape2D(nx, ny)
			if err := fd.Apply2D(lapRe, re, shape, [2]float64{hx, hy}, bcPair); err != nil {
				t.Fatalf("Apply2D re: %v", err)
			}
			if err := fd.Apply2D(lapIm, im, shape, [2]float64{hx, hy}, bcPair); err != nil {
				t.Fatalf("Apply2D im: %v", err)
			}

			var maxRes, maxF float64
			for i := range u {
				res := alpha*u[i] + complex(lapRe[i], lapIm[i]) - complex(f[i], 0)
				if a := cmplx.Abs(res); a > maxRes {
					maxRes = a
				}
				if a := math.Abs(f[i]); a > maxF {
					maxF = a
				}
			}
			if rel := maxRes / maxF; rel > 1e-9 {
				t.Fatalf("%s: residual rel error %g exceeds 1e-9", name, rel)
			}
		})
	}
}

// TestSolveComplex_Residual1D is the 1D counterpart of the residual check.
func TestSolveComplex_Residual1D(t *testing.T) {
	const (
		n = 24
		h = 1.0 / 25
	)
	alpha := complex(-12, 3)
	bcType := poisson.Neumann

	f := randomField(99, n)
	plan, err := poisson.NewComplexHelmholtzPlan(1, []int{n}, []float64{h}, []poisson.BCType{bcType}, alpha)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}

	u := make([]complex128, n)
	if err := plan.SolveComplex(u, f); err != nil {
		t.Fatalf("SolveComplex: %v", err)
	}

	re := make([]float64, n)
	im := make([]float64, n)
	for i, v := range u {
		re[i] = real(v)
		im[i] = imag(v)
	}
	lapRe := make([]float64, n)
	lapIm := make([]float64, n)
	if err := fd.Apply1D(lapRe, re, h, bcType); err != nil {
		t.Fatalf("Apply1D re: %v", err)
	}
	if err := fd.Apply1D(lapIm, im, h, bcType); err != nil {
		t.Fatalf("Apply1D im: %v", err)
	}

	var maxRes, maxF float64
	for i := range u {
		res := alpha*u[i] + complex(lapRe[i], lapIm[i]) - complex(f[i], 0)
		if a := cmplx.Abs(res); a > maxRes {
			maxRes = a
		}
		if a := math.Abs(f[i]); a > maxF {
			maxF = a
		}
	}
	if rel := maxRes / maxF; rel > 1e-9 {
		t.Fatalf("residual rel error %g exceeds 1e-9", rel)
	}
}

// TestSolveComplex_MatchesRealPath checks that a real (imag=0) alpha through the
// complex API reproduces the real Helmholtz solver: matching real part, ~0
// imaginary part.
func TestSolveComplex_MatchesRealPath(t *testing.T) {
	const (
		nx, ny = 16, 16
		hx, hy = 1.0 / 17, 1.0 / 17
		alpha  = 2.0
	)
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}
	f := randomField(5, nx*ny)

	realPlan, err := poisson.NewHelmholtzPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, alpha)
	if err != nil {
		t.Fatalf("real plan: %v", err)
	}
	realU := make([]float64, nx*ny)
	if err := realPlan.Solve(realU, f); err != nil {
		t.Fatalf("real solve: %v", err)
	}

	cplxPlan, err := poisson.NewComplexHelmholtzPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, complex(alpha, 0))
	if err != nil {
		t.Fatalf("complex plan: %v", err)
	}
	cU := make([]complex128, nx*ny)
	if err := cplxPlan.SolveComplex(cU, f); err != nil {
		t.Fatalf("complex solve: %v", err)
	}

	cRe := make([]float64, nx*ny)
	var maxImag float64
	for i, v := range cU {
		cRe[i] = real(v)
		if a := math.Abs(imag(v)); a > maxImag {
			maxImag = a
		}
	}
	if e := maxAbsDiff(cRe, realU); e > 1e-12 {
		t.Fatalf("real part differs from real solver: %g", e)
	}
	if maxImag > 1e-12 {
		t.Fatalf("imaginary part should be ~0 for real alpha, got max %g", maxImag)
	}
}

// TestSolveComplex_DampingAvoidsResonance drives the operator exactly at a
// resonance. The real solver must return ErrResonant; a small imaginary shift
// (damping) must instead yield a finite field.
func TestSolveComplex_DampingAvoidsResonance(t *testing.T) {
	const (
		n = 16
		h = 0.1
	)
	eig := bc.EigenvaluesDirichlet(n, h)
	resAlpha := -eig[0] // alpha + lambda_0 == 0 -> exact resonance
	f := randomField(3, n)
	bcs := []poisson.BCType{poisson.Dirichlet}

	realPlan, err := poisson.NewHelmholtzPlan(1, []int{n}, []float64{h}, bcs, resAlpha)
	if err != nil {
		t.Fatalf("real plan: %v", err)
	}
	if err := realPlan.Solve(make([]float64, n), f); !errors.Is(err, poisson.ErrResonant) {
		t.Fatalf("real solver at resonance: got %v, want ErrResonant", err)
	}

	dampedPlan, err := poisson.NewComplexHelmholtzPlan(1, []int{n}, []float64{h}, bcs, complex(resAlpha, 0.1))
	if err != nil {
		t.Fatalf("damped plan: %v", err)
	}
	u := make([]complex128, n)
	if err := dampedPlan.SolveComplex(u, f); err != nil {
		t.Fatalf("damped solve at resonance should succeed, got %v", err)
	}
	for i, v := range u {
		if cmplx.IsNaN(v) || cmplx.IsInf(v) {
			t.Fatalf("non-finite damped solution at %d: %v", i, v)
		}
	}
}

// TestComplexPlan_RealPathRejected verifies the real solve paths refuse a
// complex-alpha plan (which would otherwise silently use only Re(alpha)) and
// point the caller at SolveComplex.
func TestComplexPlan_RealPathRejected(t *testing.T) {
	plan, err := poisson.NewComplexHelmholtzPlan(
		1, []int{8}, []float64{0.1},
		[]poisson.BCType{poisson.Dirichlet}, complex(-5, 2),
	)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	buf := make([]float64, 8)

	if err := plan.Solve(buf, buf); !errors.Is(err, poisson.ErrComplexPlan) {
		t.Fatalf("Solve on complex plan: got %v, want ErrComplexPlan", err)
	}
	if err := plan.SolveInPlace(buf); !errors.Is(err, poisson.ErrComplexPlan) {
		t.Fatalf("SolveInPlace on complex plan: got %v, want ErrComplexPlan", err)
	}
	if err := plan.SolveWithBC(buf, buf, nil); !errors.Is(err, poisson.ErrComplexPlan) {
		t.Fatalf("SolveWithBC on complex plan: got %v, want ErrComplexPlan", err)
	}

	// A real-alpha plan through the complex constructor still takes the real path.
	realish, err := poisson.NewComplexHelmholtzPlan(1, []int{8}, []float64{0.1}, []poisson.BCType{poisson.Dirichlet}, complex(2, 0))
	if err != nil {
		t.Fatalf("real-alpha complex plan: %v", err)
	}
	if err := realish.Solve(buf, buf); err != nil {
		t.Fatalf("Solve on real-alpha plan should work, got %v", err)
	}
}

func TestSolveComplex_Validation(t *testing.T) {
	bcs := []poisson.BCType{poisson.Dirichlet}

	if _, err := poisson.NewComplexHelmholtzPlan(1, []int{8}, []float64{0.1}, bcs, complex(math.Inf(1), 0)); !errors.Is(err, poisson.ErrInvalidAlpha) {
		t.Fatalf("Inf real alpha: got %v, want ErrInvalidAlpha", err)
	}
	if _, err := poisson.NewComplexHelmholtzPlan(1, []int{8}, []float64{0.1}, bcs, complex(0, math.NaN())); !errors.Is(err, poisson.ErrInvalidAlpha) {
		t.Fatalf("NaN imag alpha: got %v, want ErrInvalidAlpha", err)
	}

	plan, err := poisson.NewComplexHelmholtzPlan(1, []int{8}, []float64{0.1}, bcs, complex(1, 1))
	if err != nil {
		t.Fatalf("valid plan: %v", err)
	}
	f := make([]float64, 8)
	if err := plan.SolveComplex(nil, f); !errors.Is(err, poisson.ErrNilBuffer) {
		t.Fatalf("nil dst: got %v, want ErrNilBuffer", err)
	}
	if err := plan.SolveComplex(make([]complex128, 7), f); !errors.Is(err, poisson.ErrSizeMismatch) {
		t.Fatalf("short dst: got %v, want ErrSizeMismatch", err)
	}
}

// TestSolveComplex_AllocParity confirms SolveComplex adds no per-call
// allocation over the real Solve on the same plan configuration. (A few
// allocations remain inherent to the DCT transform even serially; the point is
// that carrying the complex path does not make it worse.) WithWorkers(1) runs
// the pipeline inline so the parallel worker-scheduling overhead that both
// paths share does not dominate the measurement.
func TestSolveComplex_AllocParity(t *testing.T) {
	const (
		nx, ny = 32, 32
		hx, hy = 1.0 / 33, 1.0 / 33
		alpha  = 2.0
	)
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}
	f := randomField(7, nx*ny)

	realPlan, err := poisson.NewHelmholtzPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, alpha, poisson.WithWorkers(1))
	if err != nil {
		t.Fatalf("real plan: %v", err)
	}
	realDst := make([]float64, nx*ny)
	if err := realPlan.Solve(realDst, f); err != nil {
		t.Fatalf("warm real solve: %v", err)
	}
	realAllocs := testing.AllocsPerRun(50, func() { _ = realPlan.Solve(realDst, f) })

	cplxPlan, err := poisson.NewComplexHelmholtzPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, complex(alpha, 0), poisson.WithWorkers(1))
	if err != nil {
		t.Fatalf("complex plan: %v", err)
	}
	dst := make([]complex128, nx*ny)
	if err := cplxPlan.SolveComplex(dst, f); err != nil {
		t.Fatalf("warm complex solve: %v", err)
	}
	cplxAllocs := testing.AllocsPerRun(50, func() { _ = cplxPlan.SolveComplex(dst, f) })

	if cplxAllocs > realAllocs {
		t.Fatalf("SolveComplex allocates %v/op vs real Solve %v/op — complex path must not allocate more", cplxAllocs, realAllocs)
	}
}
