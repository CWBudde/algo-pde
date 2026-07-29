package poisson

import "github.com/CWBudde/algo-pde/bc"

// BCType represents the type of boundary condition. It is an alias for
// bc.BCType so that the public poisson API (poisson.BCType, poisson.Dirichlet,
// x.HasNullspace(), …) keeps working while the single source of truth lives in
// the leaf bc package.
type BCType = bc.BCType

// Boundary condition constants re-exported from the bc package.
const (
	// Periodic boundary condition: u(0) = u(L), u'(0) = u'(L).
	Periodic = bc.Periodic

	// Dirichlet boundary condition: u = g on the boundary.
	Dirichlet = bc.Dirichlet

	// Neumann boundary condition: the positive-axis derivative ∂u/∂x_axis = g is
	// prescribed on the boundary. See package docs (Grid Conventions) and
	// ApplyNeumannRHS.
	Neumann = bc.Neumann

	// DirichletNeumann is a mixed (per-face asymmetric) axis: Dirichlet on the
	// low (index-0) face, Neumann on the high face (quarter-wave DST-IV grid).
	DirichletNeumann = bc.DirichletNeumann

	// NeumannDirichlet is the mirror: Neumann low, Dirichlet high (DCT-IV grid).
	NeumannDirichlet = bc.NeumannDirichlet
)

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
