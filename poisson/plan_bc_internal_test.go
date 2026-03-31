package poisson

import (
	"errors"
	"testing"
)

func TestSolveWithBC_InputValidation(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan(1, []int{8}, []float64{1}, []BCType{Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := make([]float64, 8)
	dst := make([]float64, 8)

	err = plan.SolveWithBC(nil, rhs, nil)
	if !errors.Is(err, ErrNilBuffer) {
		t.Fatalf("nil dst err = %v, want ErrNilBuffer", err)
	}

	err = plan.SolveWithBC(dst, nil, nil)
	if !errors.Is(err, ErrNilBuffer) {
		t.Fatalf("nil rhs err = %v, want ErrNilBuffer", err)
	}

	err = plan.SolveWithBC(dst[:7], rhs, nil)
	if !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch err = %v, want ErrSizeMismatch", err)
	}
}

func TestSolveWithBC_EmptyBoundaryMatchesSolve(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan(1, []int{8}, []float64{0.1}, []BCType{Dirichlet})
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	rhs := []float64{0, 1, -0.5, 0.25, 2, -1, 0.75, -0.125}
	want := make([]float64, len(rhs))
	got := make([]float64, len(rhs))

	err = plan.Solve(want, rhs)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	err = plan.SolveWithBC(got, rhs, nil)
	if err != nil {
		t.Fatalf("SolveWithBC failed: %v", err)
	}

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("mismatch at %d: got %g, want %g", i, got[i], want[i])
		}
	}
}

func TestValidateBoundaryConditions_Errors(t *testing.T) {
	t.Parallel()

	t.Run("face not valid for dimension", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlan(1, []int{8}, []float64{1}, []BCType{Dirichlet})
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		err = plan.validateBoundaryConditions(BoundaryConditions{{Face: YLow, Type: Dirichlet, Values: make([]float64, 8)}})
		var vErr *ValidationError
		if !errors.As(err, &vErr) {
			t.Fatalf("expected ValidationError, got %v", err)
		}

		if vErr.Field != "Face" {
			t.Fatalf("Field = %q, want %q", vErr.Field, "Face")
		}
	})

	t.Run("boundary data not allowed for periodic", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlan(1, []int{8}, []float64{1}, []BCType{Periodic})
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		err = plan.validateBoundaryConditions(BoundaryConditions{{Face: XLow, Type: Periodic, Values: make([]float64, 8)}})
		var vErr *ValidationError
		if !errors.As(err, &vErr) {
			t.Fatalf("expected ValidationError, got %v", err)
		}

		if vErr.Field != "Face" {
			t.Fatalf("Field = %q, want %q", vErr.Field, "Face")
		}
	})

	t.Run("type mismatch", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlan(1, []int{8}, []float64{1}, []BCType{Dirichlet})
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		err = plan.validateBoundaryConditions(BoundaryConditions{{Face: XLow, Type: Neumann, Values: make([]float64, 8)}})
		var vErr *ValidationError
		if !errors.As(err, &vErr) {
			t.Fatalf("expected ValidationError, got %v", err)
		}

		if vErr.Field != "Type" {
			t.Fatalf("Field = %q, want %q", vErr.Field, "Type")
		}
	})
}

func TestFaceAxis(t *testing.T) {
	t.Parallel()

	tests := []struct {
		face BoundaryFace
		axis int
		ok   bool
	}{
		{face: XLow, axis: 0, ok: true},
		{face: XHigh, axis: 0, ok: true},
		{face: YLow, axis: 1, ok: true},
		{face: YHigh, axis: 1, ok: true},
		{face: ZLow, axis: 2, ok: true},
		{face: ZHigh, axis: 2, ok: true},
		{face: BoundaryFace(-1), axis: 0, ok: false},
	}

	for _, tc := range tests {
		axis, ok := faceAxis(tc.face)
		if axis != tc.axis || ok != tc.ok {
			t.Fatalf("faceAxis(%v) = (%d, %v), want (%d, %v)", tc.face, axis, ok, tc.axis, tc.ok)
		}
	}
}

func TestAxisTransforms_LengthAndNormalization(t *testing.T) {
	t.Parallel()

	fftTr, err := newFFTAxisTransform(8, 1)
	if err != nil {
		t.Fatalf("newFFTAxisTransform failed: %v", err)
	}

	if got := fftTr.Length(); got != 8 {
		t.Fatalf("fft Length = %d, want 8", got)
	}

	if got := fftTr.NormalizationFactor(); got != 1 {
		t.Fatalf("fft NormalizationFactor = %g, want 1", got)
	}

	dstTr, err := newDSTAxisTransform(8, 1)
	if err != nil {
		t.Fatalf("newDSTAxisTransform failed: %v", err)
	}

	if got := dstTr.Length(); got != 8 {
		t.Fatalf("dst Length = %d, want 8", got)
	}

	if got := dstTr.NormalizationFactor(); got <= 0 {
		t.Fatalf("dst NormalizationFactor = %g, want > 0", got)
	}

	dctTr, err := newDCTAxisTransform(8, 1)
	if err != nil {
		t.Fatalf("newDCTAxisTransform failed: %v", err)
	}

	if got := dctTr.Length(); got != 8 {
		t.Fatalf("dct Length = %d, want 8", got)
	}

	if got := dctTr.NormalizationFactor(); got <= 0 {
		t.Fatalf("dct NormalizationFactor = %g, want > 0", got)
	}
}
