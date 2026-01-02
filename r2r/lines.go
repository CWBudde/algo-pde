package r2r

import "github.com/MeKo-Tech/algo-pde/grid"

// transformFunc is a function that transforms a line of data in-place.
type transformFunc func(dst, src []float64) error

// ForwardLines applies the forward DST-I transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
//
// For a 2D array with shape [nx, ny] and axis=0, this transforms each of
// the ny columns (lines along x). For axis=1, it transforms each of the
// nx rows (lines along y).
//
// The operation is performed in-place on the data slice.
func (p *DSTPlan) ForwardLines(data []float64, shape grid.Shape, axis int) error {
	if shape.N(axis) != p.n {
		return ErrSizeMismatch
	}

	return transformAllLines(data, shape, axis, p.Forward)
}

// InverseLines applies the inverse DST-I transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
func (p *DSTPlan) InverseLines(data []float64, shape grid.Shape, axis int) error {
	if shape.N(axis) != p.n {
		return ErrSizeMismatch
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
	if shape.N(axis) != p.n {
		return ErrSizeMismatch
	}

	return transformAllLines(data, shape, axis, p.Forward)
}

// InverseLines applies the inverse DCT-I transform along all lines of data
// parallel to the given axis. The plan's size must match shape[axis].
func (p *DCTPlan) InverseLines(data []float64, shape grid.Shape, axis int) error {
	if shape.N(axis) != p.n {
		return ErrSizeMismatch
	}

	return transformAllLines(data, shape, axis, p.Inverse)
}

// transformAllLines applies a transform function to all lines along an axis.
func transformAllLines(
	data []float64, shape grid.Shape, axis int, transform transformFunc,
) error {
	it := grid.NewLineIterator(shape, axis)
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
