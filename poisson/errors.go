package poisson

import (
	"errors"
	"fmt"
)

var (
	// ErrInvalidSize is returned when grid dimensions are invalid.
	ErrInvalidSize = errors.New("invalid grid size: dimensions must be positive")

	// ErrInvalidSpacing is returned when grid spacing is invalid.
	ErrInvalidSpacing = errors.New("invalid grid spacing: must be positive and finite")

	// ErrInvalidAlpha is returned when a Helmholtz alpha is not finite.
	ErrInvalidAlpha = errors.New("invalid Helmholtz alpha: must be finite")

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

	// ErrComplexPlan is returned by the real solve paths (Solve, SolveInPlace,
	// SolveWithBC) when the plan was built with a complex alpha (imag != 0). Such
	// a plan represents a complex-valued problem; solve it with SolveComplex.
	ErrComplexPlan = errors.New("plan has a complex alpha: use SolveComplex")

	// ErrSolutionMeanRequiresNullspace is returned when WithSolutionMean is set
	// on a plan whose operator has no nullspace, where the requested mean would
	// otherwise be silently ignored.
	ErrSolutionMeanRequiresNullspace = errors.New(
		"WithSolutionMean requires a plan with a nullspace (all-periodic or all-Neumann, alpha==0)",
	)

	// ErrRealFFTUnsupported is returned when WithRealFFT/WithFloat32 is set on a
	// plan type that never runs the real-FFT path. Only the dedicated all-periodic
	// plans (NewPlan2DPeriodic / NewPlan3DPeriodic) honor it; the general Plan —
	// including the SolveWithBC path, even with all-periodic BCs — always uses the
	// complex pipeline, so the option would otherwise be a silent no-op there.
	ErrRealFFTUnsupported = errors.New(
		"WithRealFFT/WithFloat32 is only supported by NewPlan2DPeriodic/NewPlan3DPeriodic",
	)

	// ErrDuplicateFace is returned when SolveWithBC receives more than one
	// boundary entry for the same face, which would double that contribution.
	ErrDuplicateFace = errors.New("duplicate boundary face in boundary conditions")

	// ErrInvalidCoefficient is returned by NewVariableCoeffPlan when the
	// coefficient field a(x) is the wrong length or contains a non-finite or
	// non-positive value. Strict positivity is required for ellipticity (the
	// operator is only SPD when a > 0) and for the harmonic face average.
	ErrInvalidCoefficient = errors.New(
		"invalid variable coefficient: length must match the grid and every value must be finite and positive",
	)

	// ErrNotConverged is returned by VariableCoeffPlan.Solve and MaskedPlan.Solve
	// when the preconditioned conjugate gradient iteration does not reach the
	// requested tolerance within the maximum number of iterations. The returned
	// SolveStats still reports how far it got.
	ErrNotConverged = errors.New("iterative solve did not converge within the iteration limit")

	// ErrInvalidMask is returned by NewMaskedPlan when the mask is the wrong
	// length, has no active cells, or leaves the operator singular (an all-active
	// mask with all-nullspace bounding-box BCs — an unmasked periodic/Neumann
	// problem that NewPlan with WithSubtractMean already solves).
	ErrInvalidMask = errors.New(
		"invalid mask: length must match the grid, at least one cell must be active, " +
			"and an all-active mask requires a non-nullspace boundary condition",
	)
)

// Field and message strings reused across validation errors.
const (
	fieldType          = "Type"
	fieldFace          = "Face"
	msgLenMustMatchDim = "length must match dim"
	msgUnknownFace     = "unknown boundary face"
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
