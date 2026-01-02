// Package grid provides shape, stride, and indexing utilities for N-dimensional grids.
package grid

// Shape represents the dimensions of an N-dimensional grid.
// For 1D: [nx, 1, 1], 2D: [nx, ny, 1], 3D: [nx, ny, nz].
type Shape [3]int

// NewShape1D creates a 1D shape.
func NewShape1D(nx int) Shape {
	return Shape{nx, 1, 1}
}

// NewShape2D creates a 2D shape.
func NewShape2D(nx, ny int) Shape {
	return Shape{nx, ny, 1}
}

// NewShape3D creates a 3D shape.
func NewShape3D(nx, ny, nz int) Shape {
	return Shape{nx, ny, nz}
}

// Dim returns the dimensionality (1, 2, or 3).
func (s Shape) Dim() int {
	if s[2] > 1 {
		return 3
	}

	if s[1] > 1 {
		return 2
	}

	return 1
}

// Size returns the total number of elements.
func (s Shape) Size() int {
	return s[0] * s[1] * s[2]
}

// N returns the size along the given axis (0=x, 1=y, 2=z).
func (s Shape) N(axis int) int {
	return s[axis]
}

// Stride represents the memory strides for an N-dimensional grid.
// stride[i] is the number of elements to skip to advance one step along axis i.
type Stride [3]int

// RowMajorStride computes row-major (C-order) strides for a shape.
// For shape [nx, ny, nz], strides are [ny*nz, nz, 1].
func RowMajorStride(s Shape) Stride {
	return Stride{s[1] * s[2], s[2], 1}
}

// Index1D returns the linear index for a 1D coordinate.
func Index1D(i int) int {
	return i
}

// Index2D returns the linear index for a 2D coordinate (row-major).
func Index2D(i, j, ny int) int {
	return i*ny + j
}

// Index3D returns the linear index for a 3D coordinate (row-major).
func Index3D(i, j, k int, s Shape) int {
	return i*s[1]*s[2] + j*s[2] + k
}

// Index returns the linear index for coordinates using strides.
func Index(i, j, k int, stride Stride) int {
	return i*stride[0] + j*stride[1] + k*stride[2]
}

// FromIndex1D converts a linear index to 1D coordinate.
func FromIndex1D(idx int) int {
	return idx
}

// FromIndex2D converts a linear index to 2D coordinates (row-major).
func FromIndex2D(idx, ny int) (i, j int) {
	return idx / ny, idx % ny
}

// FromIndex3D converts a linear index to 3D coordinates (row-major).
func FromIndex3D(idx int, s Shape) (i, j, k int) {
	i = idx / (s[1] * s[2])
	rem := idx % (s[1] * s[2])
	j = rem / s[2]
	k = rem % s[2]

	return i, j, k
}

// LineIterator iterates over lines along a given axis.
type LineIterator struct {
	shape  Shape
	stride Stride
	axis   int

	// Current position in the "other" dimensions
	pos   [2]int // positions in the two non-axis dimensions
	max   [2]int // max values for those dimensions
	other [2]int // which axes are the "other" ones

	done bool
}

// NewLineIterator creates an iterator over lines along the given axis.
// For axis=0 in a 2D grid, it iterates over all rows (varying j).
// For axis=1 in a 2D grid, it iterates over all columns (varying i).
func NewLineIterator(shape Shape, axis int) *LineIterator {
	stride := RowMajorStride(shape)
	it := &LineIterator{
		shape:  shape,
		stride: stride,
		axis:   axis,
	}

	// Determine which dimensions are "other" (not the axis we're iterating along)
	idx := 0

	for d := range 3 {
		if d != axis {
			it.other[idx] = d
			it.max[idx] = shape[d]

			idx++
			if idx >= 2 {
				break
			}
		}
	}

	// Handle lower dimensions
	if shape.Dim() < 3 && axis != 2 {
		it.max[1] = 1 // Only iterate once in the "z" dimension for 2D
	}

	if shape.Dim() < 2 && axis != 1 {
		it.max[0] = 1 // Only iterate once for 1D
	}

	return it
}

// Next advances to the next line. Returns false when done.
func (it *LineIterator) Next() bool {
	if it.done {
		return false
	}

	it.pos[0]++
	if it.pos[0] >= it.max[0] {
		it.pos[0] = 0

		it.pos[1]++
		if it.pos[1] >= it.max[1] {
			it.done = true
			return false
		}
	}

	return true
}

// Reset resets the iterator to the beginning.
func (it *LineIterator) Reset() {
	it.pos = [2]int{}
	it.done = false
}

// StartIndex returns the starting linear index for the current line.
func (it *LineIterator) StartIndex() int {
	var coords [3]int

	coords[it.other[0]] = it.pos[0]
	coords[it.other[1]] = it.pos[1]
	coords[it.axis] = 0

	return Index(coords[0], coords[1], coords[2], it.stride)
}

// LineStride returns the stride to advance along the line.
func (it *LineIterator) LineStride() int {
	return it.stride[it.axis]
}

// LineLength returns the number of elements in each line.
func (it *LineIterator) LineLength() int {
	return it.shape[it.axis]
}

// NumLines returns the total number of lines.
func (it *LineIterator) NumLines() int {
	total := 1

	for d := range 3 {
		if d != it.axis && it.shape[d] > 0 {
			total *= it.shape[d]
		}
	}

	return total
}
