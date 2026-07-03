package bc

// BCType represents the type of boundary condition.
type BCType int

const (
	// Periodic boundary condition: u(0) = u(L), u'(0) = u'(L).
	// The domain wraps around.
	Periodic BCType = iota

	// Dirichlet boundary condition: u = g on the boundary.
	// For homogeneous Dirichlet: u = 0 at boundaries.
	Dirichlet

	// Neumann boundary condition: the derivative along the positive axis
	// direction, ∂u/∂x_axis = g, is prescribed on the boundary. This is the
	// positive-axis derivative, not the outward normal: at a low face the
	// outward normal points in the −axis direction, so there g = −∂u/∂n, while
	// at a high face g = +∂u/∂n. See the poisson package docs (Grid
	// Conventions) and ApplyNeumannRHS. For homogeneous Neumann: g = 0 at all
	// boundaries.
	Neumann
)

// String returns the string representation of the boundary condition type.
func (b BCType) String() string {
	switch b {
	case Periodic:
		return "Periodic"
	case Dirichlet:
		return "Dirichlet"
	case Neumann:
		return "Neumann"
	default:
		return "Unknown"
	}
}

// HasNullspace returns true if this boundary condition type has a nullspace
// (i.e., a constant mode with zero eigenvalue).
// Periodic and Neumann have nullspaces; Dirichlet does not.
func (b BCType) HasNullspace() bool {
	return b == Periodic || b == Neumann
}

// HasZeroEigenvalue reports whether the given BC has a zero eigenvalue
// (nullspace / constant mode). It is equivalent to b.HasNullspace().
func HasZeroEigenvalue(b BCType) bool {
	return b.HasNullspace()
}

// ZeroEigenvalueIndex returns the index of the zero eigenvalue for BC types
// that have one. Returns -1 if no zero eigenvalue exists.
func ZeroEigenvalueIndex(b BCType) int {
	switch b {
	case Periodic, Neumann:
		return 0 // The m=0 mode has zero eigenvalue.
	case Dirichlet:
		return -1 // Dirichlet has no zero eigenvalue.
	default:
		return -1
	}
}
