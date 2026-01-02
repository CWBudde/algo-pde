package fd

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func TestApply1DPeriodicModes(t *testing.T) {
	n := 16
	h := 1.0 / float64(n)
	dst := make([]float64, n)
	src := make([]float64, n)

	k := 1
	for i := range n {
		src[i] = math.Cos(2.0 * math.Pi * float64(k) * float64(i) / float64(n))
	}

	Apply1D(dst, src, h, poisson.Periodic)
	eig := EigenvaluesPeriodic(n, h)

	for i := range n {
		want := eig[k] * src[i]
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("periodic mode k=%d i=%d: got %v want %v", k, i, dst[i], want)
		}
	}

	for i := range n {
		src[i] = 1.0
	}
	Apply1D(dst, src, h, poisson.Periodic)
	for i := range n {
		if math.Abs(dst[i]) > 1e-12 {
			t.Fatalf("periodic constant mode i=%d: got %v want 0", i, dst[i])
		}
	}
}

func TestApply1DDirichletModes(t *testing.T) {
	n := 12
	h := 1.0 / float64(n+1)
	dst := make([]float64, n)
	src := make([]float64, n)

	m := 2
	for i := range n {
		x := float64(i+1) / float64(n+1)
		src[i] = math.Sin(math.Pi * float64(m) * x)
	}

	Apply1D(dst, src, h, poisson.Dirichlet)
	eig := EigenvaluesDirichlet(n, h)

	for i := range n {
		want := eig[m-1] * src[i]
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("dirichlet mode m=%d i=%d: got %v want %v", m, i, dst[i], want)
		}
	}
}

func TestApply1DNeumannModes(t *testing.T) {
	n := 10
	h := 1.0
	dst := make([]float64, n)
	src := make([]float64, n)

	m := 1
	for i := range n {
		x := (float64(i) + 0.5) / float64(n)
		src[i] = math.Cos(math.Pi * float64(m) * x)
	}

	Apply1D(dst, src, h, poisson.Neumann)
	eig := EigenvaluesNeumann(n, h)

	for i := range n {
		want := eig[m] * src[i]
		if math.Abs(dst[i]-want) > 1e-12 {
			t.Fatalf("neumann mode m=%d i=%d: got %v want %v", m, i, dst[i], want)
		}
	}
}

func TestApply1DInPlace(t *testing.T) {
	n := 8
	h := 1.0
	src := make([]float64, n)

	for i := range n {
		src[i] = math.Sin(2.0 * math.Pi * float64(i) / float64(n))
	}

	want := make([]float64, n)
	Apply1D(want, src, h, poisson.Periodic)
	Apply1D(src, src, h, poisson.Periodic)

	for i := range n {
		if math.Abs(src[i]-want[i]) > 1e-12 {
			t.Fatalf("in-place i=%d: got %v want %v", i, src[i], want[i])
		}
	}
}
