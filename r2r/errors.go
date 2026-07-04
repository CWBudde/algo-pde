package r2r

import "errors"

var (
	// ErrInvalidSize is returned when the transform size is invalid.
	ErrInvalidSize = errors.New("invalid transform size")

	// ErrSizeMismatch is returned when buffer sizes don't match the plan.
	ErrSizeMismatch = errors.New("buffer size mismatch")

	// ErrInvalidAxis is returned when a transform axis is out of range for the
	// given shape.
	ErrInvalidAxis = errors.New("invalid transform axis")
)
