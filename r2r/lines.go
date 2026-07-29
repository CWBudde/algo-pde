package r2r

import "github.com/cwbudde/algo-pde/grid"

// transformFunc is a function that transforms a line of data in-place.
type transformFunc func(dst, src []float64) error

// validateLineArgs checks the arguments common to every ForwardLines/InverseLines
// method before any data is touched. It rejects out-of-range axes, zero-extent
// shapes, plan/axis size mismatches, and buffers that are too short, so that no
// bad input reaches the (panicking) iterator or transform kernels.
func validateLineArgs(data []float64, shape grid.Shape, axis, n int) error {
	// The line transforms use grid's row-major line iteration, which supports at
	// most 3 dimensions. Reject higher-dimensional shapes here so the API returns
	// an error rather than panicking in the iterator.
	if shape.Dim() > 3 {
		return ErrInvalidAxis
	}

	if axis < 0 || axis >= shape.Dim() {
		return ErrInvalidAxis
	}

	if shape.Size() == 0 {
		return ErrInvalidSize
	}

	if shape.N(axis) != n {
		return ErrSizeMismatch
	}

	if len(data) < shape.Size() {
		return ErrSizeMismatch
	}

	return nil
}

// ForwardLines applies the forward DST-I transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
//
// For a 2D array with shape [nx, ny] and axis=0, this transforms each of
// the ny columns (lines along x). For axis=1, it transforms each of the
// nx rows (lines along y).
//
// The operation is performed in-place on the data slice.
func (p *DSTPlan) ForwardLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Forward)
}

// InverseLines applies the inverse DST-I transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
func (p *DSTPlan) InverseLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Inverse)
}

// ForwardLines applies the forward DCT-I transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
//
// For a 2D array with shape [nx, ny] and axis=0, this transforms each of
// the ny columns (lines along x). For axis=1, it transforms each of the
// nx rows (lines along y).
//
// The operation is performed in-place on the data slice.
func (p *DCTPlan) ForwardLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Forward)
}

// InverseLines applies the inverse DCT-I transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
func (p *DCTPlan) InverseLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Inverse)
}

// ForwardLines applies the forward DST-II transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
// The operation is performed in-place on the data slice.
func (p *DST2Plan) ForwardLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Forward)
}

// InverseLines applies the inverse DST-II transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
func (p *DST2Plan) InverseLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Inverse)
}

// ForwardLines applies the forward DCT-II transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
// The operation is performed in-place on the data slice.
func (p *DCT2Plan) ForwardLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Forward)
}

// InverseLines applies the inverse DCT-II transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
func (p *DCT2Plan) InverseLines(data []float64, shape grid.Shape, axis int) error {
	if err := validateLineArgs(data, shape, axis, p.n); err != nil {
		return err
	}

	return transformAllLines(data, shape, axis, p.Inverse)
}

// transformAllLines applies a transform function to all lines along an axis.
func transformAllLines(
	data []float64, shape grid.Shape, axis int, transform transformFunc,
) error {
	it := grid.NewLineIterator(shape, axis)
	if it.NumLines() == 0 {
		return nil
	}

	lineLen := it.LineLength()
	lineStride := it.LineStride()

	// Allocate temporary buffer for non-contiguous lines
	var buf []float64
	if lineStride != 1 {
		buf = make([]float64, lineLen)
	}

	// Process first line (iterator starts at position 0)
	if err := processOneLine(data, it.StartIndex(), lineLen, lineStride, buf, transform); err != nil {
		return err
	}

	// Process remaining lines
	for it.Next() {
		if err := processOneLine(data, it.StartIndex(), lineLen, lineStride, buf, transform); err != nil {
			return err
		}
	}

	return nil
}

// processOneLine transforms a single line in the data array.
func processOneLine(
	data []float64, start, length, stride int, buf []float64, transform transformFunc,
) error {
	if stride == 1 {
		// Contiguous line: transform in place
		return transform(data[start:start+length], data[start:start+length])
	}

	// Non-contiguous line: copy to buffer, transform, copy back
	for i := range length {
		buf[i] = data[start+i*stride]
	}

	if err := transform(buf, buf); err != nil {
		return err
	}

	for i := range length {
		data[start+i*stride] = buf[i]
	}

	return nil
}
