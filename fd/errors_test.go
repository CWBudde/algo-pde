package fd

import (
	"errors"
	"testing"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// bogusBC is a BCType value outside the supported set.
const bogusBC = poisson.BCType(99)

func TestApply1DErrors(t *testing.T) {
	if err := Apply1D(make([]float64, 3), make([]float64, 4), 1.0, poisson.Periodic); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: got %v, want ErrSizeMismatch", err)
	}

	if err := Apply1D(make([]float64, 4), make([]float64, 4), 1.0, bogusBC); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}

	if err := Apply1D(make([]float64, 4), make([]float64, 4), 1.0, poisson.Periodic); err != nil {
		t.Fatalf("valid call: unexpected error %v", err)
	}
}

func TestApply2DErrors(t *testing.T) {
	shape := grid.NewShape2D(2, 3)
	bc := [2]poisson.BCType{poisson.Periodic, poisson.Periodic}

	if err := Apply2D(make([]float64, 5), make([]float64, 6), shape, [2]float64{1, 1}, bc); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: got %v, want ErrSizeMismatch", err)
	}

	badBC := [2]poisson.BCType{poisson.Periodic, bogusBC}
	if err := Apply2D(make([]float64, 6), make([]float64, 6), shape, [2]float64{1, 1}, badBC); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}

	if err := Apply2D(make([]float64, 6), make([]float64, 6), shape, [2]float64{1, 1}, bc); err != nil {
		t.Fatalf("valid call: unexpected error %v", err)
	}
}

func TestApply3DErrors(t *testing.T) {
	shape := grid.NewShape3D(2, 2, 2)
	bc := [3]poisson.BCType{poisson.Periodic, poisson.Periodic, poisson.Periodic}

	if err := Apply3D(make([]float64, 7), make([]float64, 8), shape, [3]float64{1, 1, 1}, bc); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: got %v, want ErrSizeMismatch", err)
	}

	badBC := [3]poisson.BCType{poisson.Periodic, poisson.Periodic, bogusBC}
	if err := Apply3D(make([]float64, 8), make([]float64, 8), shape, [3]float64{1, 1, 1}, badBC); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}

	if err := Apply3D(make([]float64, 8), make([]float64, 8), shape, [3]float64{1, 1, 1}, bc); err != nil {
		t.Fatalf("valid call: unexpected error %v", err)
	}
}

func TestEigenvaluesErrors(t *testing.T) {
	if _, err := Eigenvalues(4, 1.0, bogusBC); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}

	// A negative n must return an error, not panic in make (no-panic contract).
	if _, err := Eigenvalues(-1, 1.0, poisson.Periodic); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("negative n: got %v, want ErrSizeMismatch", err)
	}

	eig, err := Eigenvalues(4, 1.0, poisson.Periodic)
	if err != nil {
		t.Fatalf("valid call: unexpected error %v", err)
	}
	if len(eig) != 4 {
		t.Fatalf("expected 4 eigenvalues, got %d", len(eig))
	}
}
