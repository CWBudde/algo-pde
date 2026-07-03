package fd

import "errors"

var (
	// ErrSizeMismatch is returned when the destination or source buffer length
	// does not match the size expected for the given shape.
	ErrSizeMismatch = errors.New("fd: buffer size does not match expected grid size")

	// ErrInvalidBC is returned when an unknown boundary condition type is passed.
	ErrInvalidBC = errors.New("fd: invalid boundary condition type")
)
