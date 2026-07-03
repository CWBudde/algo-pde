// Package grid provides shape, stride, and indexing utilities for N-dimensional grids.
package grid

// Shape represents the dimensions of an N-dimensional grid (1, 2, or 3D).
//
// The extents are stored in a fixed array (trailing axes are 1) alongside the
// DECLARED dimensionality, so a grid whose last extent happens to be 1 (e.g.
// 64x64x1) still reports Dim() == 3. Storing the declared dimension avoids the
// ambiguity of inferring it from the trailing extents.
//
// Shape has value semantics: it is comparable and never heap-allocates, which
// keeps it cheap to construct once per Solve on the solver hot path.
type Shape struct {
	dims [3]int
	ndim int
}

// mustNonNegative panics on a negative extent. The NewShapeND constructors take
// no error return and are called inline throughout the codebase, so a negative
// extent is treated as a programming error (like a slice bounds violation)
// rather than a recoverable condition. Zero extents are permitted: they denote
// an empty grid, which the iterators handle explicitly.
func mustNonNegative(extents ...int) {
	for _, n := range extents {
		if n < 0 {
			panic("grid: negative shape extent")
		}
	}
}

// NewShape1D creates a 1D shape. It panics if nx is negative.
func NewShape1D(nx int) Shape {
	mustNonNegative(nx)
	return Shape{dims: [3]int{nx, 1, 1}, ndim: 1}
}

// NewShape2D creates a 2D shape. It panics if any extent is negative.
func NewShape2D(nx, ny int) Shape {
	mustNonNegative(nx, ny)
	return Shape{dims: [3]int{nx, ny, 1}, ndim: 2}
}

// NewShape3D creates a 3D shape. It panics if any extent is negative.
func NewShape3D(nx, ny, nz int) Shape {
	mustNonNegative(nx, ny, nz)
	return Shape{dims: [3]int{nx, ny, nz}, ndim: 3}
}

// Dim returns the declared dimensionality (1, 2, or 3).
func (s Shape) Dim() int {
	return s.ndim
}

// Size returns the total number of elements.
func (s Shape) Size() int {
	return s.dims[0] * s.dims[1] * s.dims[2]
}

// N returns the size along the given axis (0=x, 1=y, 2=z).
func (s Shape) N(axis int) int {
	return s.dims[axis]
}

// hasZeroExtent reports whether any axis has a zero extent, in which case the
// grid is empty (no lines or planes to iterate).
func (s Shape) hasZeroExtent() bool {
	return s.dims[0] == 0 || s.dims[1] == 0 || s.dims[2] == 0
}

// Stride represents the memory strides for an N-dimensional grid.
// stride[i] is the number of elements to skip to advance one step along axis i.
type Stride [3]int

// RowMajorStride computes row-major (C-order) strides for a shape.
// For shape [nx, ny, nz], strides are [ny*nz, nz, 1].
func RowMajorStride(s Shape) Stride {
	return Stride{s.dims[1] * s.dims[2], s.dims[2], 1}
}

// Index2D returns the linear index for a 2D coordinate (row-major).
func Index2D(i, j, ny int) int {
	return i*ny + j
}

// Index3D returns the linear index for a 3D coordinate (row-major).
func Index3D(i, j, k int, s Shape) int {
	return i*s.dims[1]*s.dims[2] + j*s.dims[2] + k
}

// Index returns the linear index for coordinates using strides.
func Index(i, j, k int, stride Stride) int {
	return i*stride[0] + j*stride[1] + k*stride[2]
}

// FromIndex2D converts a linear index to 2D coordinates (row-major).
func FromIndex2D(idx, ny int) (i, j int) {
	return idx / ny, idx % ny
}

// FromIndex3D converts a linear index to 3D coordinates (row-major).
func FromIndex3D(idx int, s Shape) (i, j, k int) {
	i = idx / (s.dims[1] * s.dims[2])
	rem := idx % (s.dims[1] * s.dims[2])
	j = rem / s.dims[2]
	k = rem % s.dims[2]

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

	empty bool
	done  bool
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
			it.max[idx] = shape.dims[d]

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

	// A zero-extent shape has no lines at all: mark the iterator empty and done
	// so it yields nothing (rather than a phantom line for the collapsed dim).
	if shape.hasZeroExtent() {
		it.empty = true
		it.done = true
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
	it.done = it.empty
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
	return it.shape.dims[it.axis]
}

// NumLines returns the total number of lines. A shape with any zero extent has
// no lines and returns 0.
func (it *LineIterator) NumLines() int {
	if it.empty {
		return 0
	}

	total := 1

	for d := range 3 {
		if d != it.axis && it.shape.dims[d] > 0 {
			total *= it.shape.dims[d]
		}
	}

	return total
}

// CopyStrided copies n elements from src to dst using the given strides.
// dstStride and srcStride are element strides (not bytes).
func CopyStrided(dst []float64, dstStride int, src []float64, srcStride int, n int) {
	di := 0
	si := 0

	for range n {
		dst[di] = src[si]
		di += dstStride
		si += srcStride
	}
}

// CopyStridedToContiguous copies a strided source into a contiguous slice.
func CopyStridedToContiguous(dst []float64, src []float64, srcStride int) {
	CopyStrided(dst, 1, src, srcStride, len(dst))
}

// CopyContiguousToStrided copies a contiguous source into a strided destination.
func CopyContiguousToStrided(dst []float64, dstStride int, src []float64) {
	CopyStrided(dst, dstStride, src, 1, len(src))
}

// PlaneIterator iterates over planes orthogonal to a given axis.
// A plane is defined by fixing one coordinate along the axis and varying
// the other two coordinates.
type PlaneIterator struct {
	shape  Shape
	stride Stride
	axis   int

	pos   int
	max   int
	other [2]int

	empty bool
	done  bool
}

// NewPlaneIterator creates an iterator over planes orthogonal to the given axis.
// For axis=0 in a 3D grid, it iterates over all YZ planes (varying i).
func NewPlaneIterator(shape Shape, axis int) *PlaneIterator {
	stride := RowMajorStride(shape)
	it := &PlaneIterator{
		shape:  shape,
		stride: stride,
		axis:   axis,
		max:    shape.dims[axis],
	}

	idx := 0
	for d := range 3 {
		if d != axis {
			it.other[idx] = d
			idx++
			if idx >= 2 {
				break
			}
		}
	}

	// A zero-extent shape has no planes at all.
	if shape.hasZeroExtent() {
		it.empty = true
		it.done = true
	}

	return it
}

// Next advances to the next plane. Returns false when done.
func (it *PlaneIterator) Next() bool {
	if it.done {
		return false
	}

	it.pos++
	if it.pos >= it.max {
		it.done = true
		return false
	}

	return true
}

// Reset resets the iterator to the beginning.
func (it *PlaneIterator) Reset() {
	it.pos = 0
	it.done = it.empty
}

// StartIndex returns the starting linear index for the current plane.
func (it *PlaneIterator) StartIndex() int {
	var coords [3]int

	coords[it.axis] = it.pos
	coords[it.other[0]] = 0
	coords[it.other[1]] = 0

	return Index(coords[0], coords[1], coords[2], it.stride)
}

// PlaneStride0 returns the stride along the first plane axis.
func (it *PlaneIterator) PlaneStride0() int {
	return it.stride[it.other[0]]
}

// PlaneStride1 returns the stride along the second plane axis.
func (it *PlaneIterator) PlaneStride1() int {
	return it.stride[it.other[1]]
}

// PlaneSize0 returns the size along the first plane axis.
func (it *PlaneIterator) PlaneSize0() int {
	return it.shape.dims[it.other[0]]
}

// PlaneSize1 returns the size along the second plane axis.
func (it *PlaneIterator) PlaneSize1() int {
	return it.shape.dims[it.other[1]]
}

// NumPlanes returns the total number of planes. A shape with any zero extent has
// no planes and returns 0.
func (it *PlaneIterator) NumPlanes() int {
	if it.empty || it.max < 1 {
		return 0
	}

	return it.max
}
