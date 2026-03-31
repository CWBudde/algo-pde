package poisson_test

import (
	"testing"

	"github.com/cwbudde/algo-pde/poisson"
)

func TestBCType_String(t *testing.T) {
	t.Parallel()

	if got := poisson.Periodic.String(); got != "Periodic" {
		t.Fatalf("Periodic.String() = %q", got)
	}

	if got := poisson.Dirichlet.String(); got != "Dirichlet" {
		t.Fatalf("Dirichlet.String() = %q", got)
	}

	if got := poisson.Neumann.String(); got != "Neumann" {
		t.Fatalf("Neumann.String() = %q", got)
	}

	if got := poisson.BCType(999).String(); got != "Unknown" {
		t.Fatalf("unknown BCType String() = %q, want Unknown", got)
	}
}

func TestNewAxisBC(t *testing.T) {
	t.Parallel()

	axis := poisson.NewAxisBC(poisson.Neumann)
	if axis.Type != poisson.Neumann {
		t.Fatalf("axis.Type = %v, want %v", axis.Type, poisson.Neumann)
	}
}

func TestErrorTypes_ErrorMessages(t *testing.T) {
	t.Parallel()

	sizeErr := &poisson.SizeError{Expected: 16, Got: 8, Context: "test"}
	if got, want := sizeErr.Error(), "size mismatch in test: expected 16, got 8"; got != want {
		t.Fatalf("SizeError.Error() = %q, want %q", got, want)
	}

	validationErr := &poisson.ValidationError{Field: "bc", Message: "invalid"}
	if got, want := validationErr.Error(), "validation error for bc: invalid"; got != want {
		t.Fatalf("ValidationError.Error() = %q, want %q", got, want)
	}
}
