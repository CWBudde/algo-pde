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
