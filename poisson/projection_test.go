package poisson

import (
	"errors"
	"math"
	"math/rand"
	"sync"
	"testing"
)

func maxAbsSlice(s []float64) float64 {
	m := 0.0
	for _, v := range s {
		if a := math.Abs(v); a > m {
			m = a
		}
	}
	return m
}

func l2Norm(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func randomField(rng *rand.Rand, n int) []float64 {
	f := make([]float64, n)
	for i := range f {
		f[i] = rng.NormFloat64()
	}
	return f
}

func TestProjection2D_MakesDivergenceFree(t *testing.T) {
	const nx, ny = 24, 32
	hx, hy := 0.1, 0.07
	plan, err := NewProjectionPlan2D(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewProjectionPlan2D: %v", err)
	}

	rng := rand.New(rand.NewSource(1))
	u := randomField(rng, nx*ny)
	v := randomField(rng, nx*ny)

	div := make([]float64, nx*ny)
	_ = plan.Divergence(div, u, v)
	before := maxAbsSlice(div)

	if err := plan.Project(u, v); err != nil {
		t.Fatalf("Project: %v", err)
	}

	if err := plan.Divergence(div, u, v); err != nil {
		t.Fatalf("Divergence: %v", err)
	}
	after := maxAbsSlice(div)

	if after > 1e-10 {
		t.Fatalf("projected field not divergence-free: max|div| before=%.3e after=%.3e", before, after)
	}
	if before < 1e-3 {
		t.Fatalf("test field was already nearly divergence-free (before=%.3e); test is vacuous", before)
	}
}

// A pure gradient field u* = ∇φ has no divergence-free part, so Project must
// remove essentially all of it.
func TestProjection2D_PureGradientProjectsToZero(t *testing.T) {
	const nx, ny = 20, 28
	hx, hy := 0.13, 0.09
	plan, err := NewProjectionPlan2D(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewProjectionPlan2D: %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	phi := randomField(rng, nx*ny)

	// Build (u, v) = forward-difference gradient of phi — the same gradient
	// Project subtracts, so the whole field should be removed.
	u := make([]float64, nx*ny)
	v := make([]float64, nx*ny)
	for i := range nx {
		ip1 := (i + 1) % nx
		for j := range ny {
			jp1 := (j + 1) % ny
			idx := i*ny + j
			u[idx] = (phi[ip1*ny+j] - phi[idx]) / hx
			v[idx] = (phi[i*ny+jp1] - phi[idx]) / hy
		}
	}
	mag := math.Max(l2Norm(u), l2Norm(v))

	if err := plan.Project(u, v); err != nil {
		t.Fatalf("Project: %v", err)
	}

	residual := math.Max(l2Norm(u), l2Norm(v))
	if residual > 1e-9*mag {
		t.Fatalf("pure gradient not removed: residual=%.3e original=%.3e", residual, mag)
	}
}

// Projecting an already divergence-free field must leave it (nearly) unchanged.
func TestProjection2D_Idempotent(t *testing.T) {
	const nx, ny = 30, 30
	hx, hy := 0.1, 0.1
	plan, err := NewProjectionPlan2D(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewProjectionPlan2D: %v", err)
	}

	rng := rand.New(rand.NewSource(3))
	u := randomField(rng, nx*ny)
	v := randomField(rng, nx*ny)

	if err := plan.Project(u, v); err != nil {
		t.Fatalf("first Project: %v", err)
	}
	u1 := append([]float64(nil), u...)
	v1 := append([]float64(nil), v...)

	if err := plan.Project(u, v); err != nil {
		t.Fatalf("second Project: %v", err)
	}

	du, dv := 0.0, 0.0
	for i := range u {
		du = math.Max(du, math.Abs(u[i]-u1[i]))
		dv = math.Max(dv, math.Abs(v[i]-v1[i]))
	}
	change := math.Max(du, dv)
	scale := math.Max(maxAbsSlice(u1), maxAbsSlice(v1))
	if change > 1e-10*scale {
		t.Fatalf("projection not idempotent: change=%.3e scale=%.3e", change, scale)
	}
}

func TestProjection3D_MakesDivergenceFree(t *testing.T) {
	const nx, ny, nz = 12, 16, 10
	hx, hy, hz := 0.1, 0.08, 0.12
	plan, err := NewProjectionPlan3D(nx, ny, nz, hx, hy, hz)
	if err != nil {
		t.Fatalf("NewProjectionPlan3D: %v", err)
	}

	n := nx * ny * nz
	rng := rand.New(rand.NewSource(11))
	u := randomField(rng, n)
	v := randomField(rng, n)
	w := randomField(rng, n)

	div := make([]float64, n)
	_ = plan.Divergence(div, u, v, w)
	before := maxAbsSlice(div)

	if err := plan.Project(u, v, w); err != nil {
		t.Fatalf("Project: %v", err)
	}

	if err := plan.Divergence(div, u, v, w); err != nil {
		t.Fatalf("Divergence: %v", err)
	}
	after := maxAbsSlice(div)

	if after > 1e-10 {
		t.Fatalf("projected 3D field not divergence-free: before=%.3e after=%.3e", before, after)
	}
	if before < 1e-3 {
		t.Fatalf("test field already nearly divergence-free (before=%.3e); vacuous", before)
	}
}

func TestProjection_Validation(t *testing.T) {
	p2, err := NewProjectionPlan2D(8, 8, 0.1, 0.1)
	if err != nil {
		t.Fatalf("NewProjectionPlan2D: %v", err)
	}
	if err := p2.Project(nil, make([]float64, 64)); !errors.Is(err, ErrNilBuffer) {
		t.Fatalf("nil u: got %v want ErrNilBuffer", err)
	}
	if err := p2.Project(make([]float64, 10), make([]float64, 64)); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("bad size: got %v want ErrSizeMismatch", err)
	}
	if err := p2.Divergence(make([]float64, 64), make([]float64, 64), nil); !errors.Is(err, ErrNilBuffer) {
		t.Fatalf("nil v in Divergence: got %v want ErrNilBuffer", err)
	}

	if _, err := NewProjectionPlan2D(0, 8, 0.1, 0.1); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("bad nx: got %v want ErrInvalidSize", err)
	}
	if _, err := NewProjectionPlan3D(8, 8, 8, 0.1, math.NaN(), 0.1); !errors.Is(err, ErrInvalidSpacing) {
		t.Fatalf("NaN spacing: got %v want ErrInvalidSpacing", err)
	}
}

// Concurrent Project calls on one shared plan must match serial results,
// exercising the per-call scratch pool. Run under -race.
func TestProjection2D_Concurrent(t *testing.T) {
	const nx, ny = 16, 16
	hx, hy := 0.1, 0.1
	plan, err := NewProjectionPlan2D(nx, ny, hx, hy)
	if err != nil {
		t.Fatalf("NewProjectionPlan2D: %v", err)
	}

	const goroutines = 8
	fields := make([][2][]float64, goroutines)
	serial := make([][2][]float64, goroutines)
	for g := range goroutines {
		rng := rand.New(rand.NewSource(int64(g) + 100))
		u := randomField(rng, nx*ny)
		v := randomField(rng, nx*ny)
		fields[g] = [2][]float64{append([]float64(nil), u...), append([]float64(nil), v...)}
		su := append([]float64(nil), u...)
		sv := append([]float64(nil), v...)
		if err := plan.Project(su, sv); err != nil {
			t.Fatalf("serial Project: %v", err)
		}
		serial[g] = [2][]float64{su, sv}
	}

	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			_ = plan.Project(fields[g][0], fields[g][1])
		}(g)
	}
	wg.Wait()

	for g := range goroutines {
		for i := range fields[g][0] {
			if math.Abs(fields[g][0][i]-serial[g][0][i]) > 1e-12 ||
				math.Abs(fields[g][1][i]-serial[g][1][i]) > 1e-12 {
				t.Fatalf("goroutine %d diverged from serial at %d", g, i)
			}
		}
	}
}
