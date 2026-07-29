package fd

import (
	"errors"
	"testing"

	"github.com/CWBudde/algo-pde/bc"
	"github.com/CWBudde/algo-pde/grid"
)

// bogusBC is a BCType value outside the supported set.
const bogusBC = bc.BCType(99)

func TestApply1DErrors(t *testing.T) {
	if err := Apply1D(make([]float64, 3), make([]float64, 4), 1.0, bc.Periodic); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: got %v, want ErrSizeMismatch", err)
	}

	if err := Apply1D(make([]float64, 4), make([]float64, 4), 1.0, bogusBC); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}

	if err := Apply1D(make([]float64, 4), make([]float64, 4), 1.0, bc.Periodic); err != nil {
		t.Fatalf("valid call: unexpected error %v", err)
	}
}

func TestApply2DErrors(t *testing.T) {
	shape := grid.NewShape2D(2, 3)
	bcs := [2]bc.BCType{bc.Periodic, bc.Periodic}

	if err := Apply2D(make([]float64, 5), make([]float64, 6), shape, [2]float64{1, 1}, bcs); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: got %v, want ErrSizeMismatch", err)
	}

	badBC := [2]bc.BCType{bc.Periodic, bogusBC}
	if err := Apply2D(make([]float64, 6), make([]float64, 6), shape, [2]float64{1, 1}, badBC); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}

	if err := Apply2D(make([]float64, 6), make([]float64, 6), shape, [2]float64{1, 1}, bcs); err != nil {
		t.Fatalf("valid call: unexpected error %v", err)
	}
}

func TestApply3DErrors(t *testing.T) {
	shape := grid.NewShape3D(2, 2, 2)
	bcs := [3]bc.BCType{bc.Periodic, bc.Periodic, bc.Periodic}

	if err := Apply3D(make([]float64, 7), make([]float64, 8), shape, [3]float64{1, 1, 1}, bcs); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: got %v, want ErrSizeMismatch", err)
	}

	badBC := [3]bc.BCType{bc.Periodic, bc.Periodic, bogusBC}
	if err := Apply3D(make([]float64, 8), make([]float64, 8), shape, [3]float64{1, 1, 1}, badBC); !errors.Is(err, ErrInvalidBC) {
		t.Fatalf("bogus BC: got %v, want ErrInvalidBC", err)
	}

	if err := Apply3D(make([]float64, 8), make([]float64, 8), shape, [3]float64{1, 1, 1}, bcs); err != nil {
		t.Fatalf("valid call: unexpected error %v", err)
	}
}
