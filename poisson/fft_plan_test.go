package poisson

import (
	"errors"
	"math"
	"testing"

	algofft "github.com/MeKo-Christian/algo-fft"
	"github.com/MeKo-Tech/algo-pde/grid"
)

const fftTol = 1e-10

func TestNewFFTPlan_InvalidSize(t *testing.T) {
	if _, err := NewFFTPlan(0); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("NewFFTPlan(0) err = %v, want ErrInvalidSize", err)
	}
}

func TestFFTPlan_TransformLines_Errors(t *testing.T) {
	p, err := NewFFTPlan(4)
	if err != nil {
		t.Fatalf("NewFFTPlan failed: %v", err)
	}

	shape := grid.NewShape2D(4, 3)

	if err := p.TransformLines(nil, shape, 0, false); !errors.Is(err, ErrNilBuffer) {
		t.Fatalf("nil data err = %v, want ErrNilBuffer", err)
	}

	badLen := make([]complex128, shape.Size()-1)
	if err := p.TransformLines(badLen, shape, 0, false); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("len mismatch err = %v, want ErrSizeMismatch", err)
	}

	wrongAxisLenShape := grid.NewShape2D(5, 3)
	data := make([]complex128, wrongAxisLenShape.Size())
	if err := p.TransformLines(data, wrongAxisLenShape, 0, false); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("axis length mismatch err = %v, want ErrSizeMismatch", err)
	}
}

func TestFFTPlan_TransformLines_MatchesReference_Axis0_NonContiguous(t *testing.T) {
	// Axis 0 in row-major 2D has stride = ny (usually != 1).
	nx, ny := 8, 5
	shape := grid.NewShape2D(nx, ny)

	p, err := NewFFTPlan(nx)
	if err != nil {
		t.Fatalf("NewFFTPlan failed: %v", err)
	}

	refPlan, err := algofft.NewPlan64(nx)
	if err != nil {
		t.Fatalf("algofft.NewPlan64 failed: %v", err)
	}

	data := make([]complex128, shape.Size())
	for i := range data {
		data[i] = complex(float64(i+1), float64(1000+i))
	}
	got := append([]complex128(nil), data...)
	want := append([]complex128(nil), data...)

	if err := p.TransformLines(got, shape, 0, false); err != nil {
		t.Fatalf("TransformLines forward failed: %v", err)
	}

	it := grid.NewLineIterator(shape, 0)
	stride := it.LineStride()
	start := it.StartIndex()
	if err := refPlan.TransformStrided(want[start:], want[start:], stride, false); err != nil {
		t.Fatalf("reference TransformStrided failed: %v", err)
	}
	for it.Next() {
		start = it.StartIndex()
		if err := refPlan.TransformStrided(want[start:], want[start:], stride, false); err != nil {
			t.Fatalf("reference TransformStrided failed: %v", err)
		}
	}

	for i := range got {
		if cmplxAbs(got[i]-want[i]) > fftTol {
			t.Fatalf("mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestFFTPlan_TransformLines_RoundTrip_2D_Axis0And1(t *testing.T) {
	nx, ny := 8, 4
	shape := grid.NewShape2D(nx, ny)

	planX, err := NewFFTPlan(nx)
	if err != nil {
		t.Fatalf("NewFFTPlan(nx) failed: %v", err)
	}
	planY, err := NewFFTPlan(ny)
	if err != nil {
		t.Fatalf("NewFFTPlan(ny) failed: %v", err)
	}

	data := make([]complex128, shape.Size())
	for i := range data {
		data[i] = complex(math.Sin(float64(i+1)*0.1), math.Cos(float64(i+1)*0.07))
	}
	orig := append([]complex128(nil), data...)

	// Forward along both axes.
	if err := planX.TransformLines(data, shape, 0, false); err != nil {
		t.Fatalf("forward axis 0 failed: %v", err)
	}
	if err := planY.TransformLines(data, shape, 1, false); err != nil {
		t.Fatalf("forward axis 1 failed: %v", err)
	}

	// Inverse back.
	if err := planY.TransformLines(data, shape, 1, true); err != nil {
		t.Fatalf("inverse axis 1 failed: %v", err)
	}
	if err := planX.TransformLines(data, shape, 0, true); err != nil {
		t.Fatalf("inverse axis 0 failed: %v", err)
	}

	for i := range data {
		if cmplxAbs(data[i]-orig[i]) > fftTol {
			t.Fatalf("round-trip mismatch at %d: got %v, want %v", i, data[i], orig[i])
		}
	}
}

func TestFFTPlan_TransformLines_ParallelMatchesSerial(t *testing.T) {
	nx, ny := 6, 5
	shape := grid.NewShape2D(nx, ny)

	serialPlan, err := NewFFTPlan(nx)
	if err != nil {
		t.Fatalf("NewFFTPlan(nx) failed: %v", err)
	}
	parallelPlan, err := NewFFTPlanWithWorkers(nx, 3)
	if err != nil {
		t.Fatalf("NewFFTPlanWithWorkers(nx, 3) failed: %v", err)
	}

	data := make([]complex128, shape.Size())
	for i := range data {
		data[i] = complex(float64(i%7+1), float64(100+i))
	}
	got := append([]complex128(nil), data...)
	want := append([]complex128(nil), data...)

	if err := parallelPlan.TransformLines(got, shape, 0, false); err != nil {
		t.Fatalf("parallel TransformLines failed: %v", err)
	}
	if err := serialPlan.TransformLines(want, shape, 0, false); err != nil {
		t.Fatalf("serial TransformLines failed: %v", err)
	}

	for i := range got {
		if cmplxAbs(got[i]-want[i]) > fftTol {
			t.Fatalf("parallel mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func cmplxAbs(z complex128) float64 {
	return math.Hypot(real(z), imag(z))
}
