package poisson

import (
	"errors"
	"fmt"
)

var (
	// ErrInvalidSize is returned when grid dimensions are invalid.
	ErrInvalidSize = errors.New("invalid grid size: dimensions must be positive")

	// ErrInvalidSpacing is returned when grid spacing is invalid.
	ErrInvalidSpacing = errors.New("invalid grid spacing: must be positive")

	// ErrSizeMismatch is returned when buffer sizes don't match the plan.
	ErrSizeMismatch = errors.New("buffer size does not match plan dimensions")

	// ErrNullspace is returned when a problem has a nullspace but
	// NullspaceError handling is configured.
	ErrNullspace = errors.New("problem has nullspace (zero eigenvalue): " +
		"periodic or Neumann BC without unique solution")

	// ErrNonZeroMean is returned when the RHS does not have mean zero
	// but the problem requires it (for nullspace consistency).
	ErrNonZeroMean = errors.New("RHS does not have mean zero: " +
		"problem is inconsistent for periodic/Neumann BC")

	// ErrNilBuffer is returned when a required buffer is nil.
	ErrNilBuffer = errors.New("buffer is nil")

	// ErrResonant is returned when the Helmholtz operator is singular.
	ErrResonant = errors.New("helmholtz operator is singular: alpha cancels eigenvalue")
)

// SizeError provides details about a size mismatch.
type SizeError struct {
	Expected int
	Got      int
	Context  string
}

func (e *SizeError) Error() string {
	return fmt.Sprintf("size mismatch in %s: expected %d, got %d",
		e.Context, e.Expected, e.Got)
}

// ValidationError wraps validation failures with context.
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation error for %s: %s", e.Field, e.Message)
}
