package poisson

// BCType represents the type of boundary condition.
type BCType int

const (
	// Periodic boundary condition: u(0) = u(L), u'(0) = u'(L).
	// The domain wraps around.
	Periodic BCType = iota

	// Dirichlet boundary condition: u = g on the boundary.
	// For homogeneous Dirichlet: u = 0 at boundaries.
	Dirichlet

	// Neumann boundary condition: ∂u/∂n = g on the boundary.
	// For homogeneous Neumann: ∂u/∂n = 0 at boundaries.
	Neumann
)

// String returns the string representation of the boundary condition type.
func (bc BCType) String() string {
	switch bc {
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
func (bc BCType) HasNullspace() bool {
	return bc == Periodic || bc == Neumann
}

// AxisBC represents boundary conditions for a single axis.
// Currently supports the same BC type on both ends.
// Future: may support different left/right conditions.
type AxisBC struct {
	Type BCType
}

// NewAxisBC creates a new AxisBC with the given type.
func NewAxisBC(t BCType) AxisBC {
	return AxisBC{Type: t}
}

// BoundaryFace identifies a specific boundary face of the domain.
// The low/high names refer to the coordinate direction.
type BoundaryFace int

const (
	XLow BoundaryFace = iota
	XHigh
	YLow
	YHigh
	ZLow
	ZHigh
)

// BoundaryData associates boundary values with a face and BC type.
type BoundaryData struct {
	Face   BoundaryFace
	Type   BCType
	Values []float64
}

// BoundaryConditions is a collection of boundary data entries.
type BoundaryConditions []BoundaryData
