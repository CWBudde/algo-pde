package poisson

// Shape represents the dimensions of an N-dimensional grid in row-major order.
type Shape []int

// Dim returns the number of dimensions.
func (s Shape) Dim() int {
	return len(s)
}

// Size returns the total number of elements.
func (s Shape) Size() int {
	if len(s) == 0 {
		return 0
	}

	size := 1
	for _, n := range s {
		size *= n
	}

	return size
}

// N returns the size along the given axis.
func (s Shape) N(axis int) int {
	if axis < 0 || axis >= len(s) {
		return 0
	}

	return s[axis]
}

// minExtent returns the smallest per-axis extent, or 0 for an empty shape.
// It drives the discretization-scaled zero-mean tolerance, where the coarsest
// axis dominates the O(h^2) quadrature error.
func (s Shape) minExtent() int {
	if len(s) == 0 {
		return 0
	}

	m := s[0]
	for _, n := range s[1:] {
		if n < m {
			m = n
		}
	}

	return m
}
