package poisson

import (
	"fmt"
)

// SolveWithBC computes the solution into dst for a given RHS and boundary data.
// The boundary data is applied as inhomogeneous Dirichlet/Neumann contributions.
func (p *Plan) SolveWithBC(dst, rhs []float64, bc BoundaryConditions) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	// Inhomogeneous BC lifting is a real-valued operation; a complex-alpha plan
	// is not supported here (there is no complex SolveWithBC).
	if imag(p.alphaComplex) != 0 {
		return ErrComplexPlan
	}

	size := p.size()
	if len(dst) != size || len(rhs) != size {
		return ErrSizeMismatch
	}

	if len(bc) == 0 {
		return p.Solve(dst, rhs)
	}

	// Validate everything (faces, types, duplicate faces, and value sizes) before
	// touching any data. In the InPlace path buf aliases the caller's rhs, so a
	// mid-loop error after partial mutation would corrupt it; full up-front
	// validation guarantees rhs is untouched whenever an error is returned.
	if err := p.validateBoundaryConditions(bc); err != nil {
		return err
	}

	// Faces are grouped by the lift they need. A Dirichlet face on a mixed
	// quarter-wave axis reflects as u₋₁ = 2g − u₀, so it lifts with twice the
	// vertex-centered contribution; its Neumann face uses the same ∓g/h lift as a
	// pure Neumann axis, so mixed and pure Neumann faces share one group.
	var dirichlet, mixedDirichlet, neumann BoundaryConditions
	for _, data := range bc {
		axis, _ := faceAxis(data.Face)
		switch data.Type {
		case Dirichlet:
			if isMixedAxisBC(p.bc[axis]) {
				mixedDirichlet = append(mixedDirichlet, data)
			} else {
				dirichlet = append(dirichlet, data)
			}
		case Neumann:
			neumann = append(neumann, data)
		default:
			return &ValidationError{
				Field:   fieldType,
				Message: "unsupported boundary condition",
			}
		}
	}

	workspace := p.work.get()
	defer p.work.put(workspace)

	buf := rhs
	if !p.opts.InPlace {
		buf = workspace.Real[:size]
		copy(buf, rhs)
	}

	shape := p.shape()
	h := p.h
	if len(dirichlet) > 0 {
		if err := ApplyDirichletRHS(buf, shape, h, dirichlet); err != nil {
			return err
		}
	}
	if len(mixedDirichlet) > 0 {
		if err := applyDirichletRHS(buf, shape, h, mixedDirichlet, 2.0); err != nil {
			return err
		}
	}
	if len(neumann) > 0 {
		if err := ApplyNeumannRHS(buf, shape, h, neumann); err != nil {
			return err
		}
	}

	return p.solve(dst, buf, workspace)
}

func (p *Plan) validateBoundaryConditions(bc BoundaryConditions) error {
	seen := make(map[BoundaryFace]bool, len(bc))
	for _, data := range bc {
		axis, ok := faceAxis(data.Face)
		if !ok || axis >= p.dim {
			return &ValidationError{
				Field:   fieldFace,
				Message: "boundary face not valid for plan dimension",
			}
		}

		if seen[data.Face] {
			return fmt.Errorf("%w: %v", ErrDuplicateFace, data.Face)
		}
		seen[data.Face] = true

		want, ok := expectedFaceType(p.bc[axis], data.Face)
		if !ok {
			return &ValidationError{
				Field:   fieldFace,
				Message: "boundary data not allowed for periodic axis",
			}
		}

		if want != data.Type {
			return &ValidationError{
				Field: fieldType,
				Message: fmt.Sprintf("boundary type %s does not match %v face of plan axis %s",
					data.Type, data.Face, p.bc[axis]),
			}
		}

		if err := p.validateFaceValues(data); err != nil {
			return err
		}
	}

	return nil
}

// validateFaceValues checks that a boundary face carries the correct number of
// values for the plan's shape, matching what ApplyDirichletRHS/ApplyNeumannRHS
// expect. Doing this up front lets SolveWithBC reject bad input before mutating
// the caller's rhs.
func (p *Plan) validateFaceValues(data BoundaryData) error {
	nx, ny, nz := p.n[0], p.n[1], p.n[2]

	var expected int
	switch data.Face {
	case XLow, XHigh:
		expected = ny * nz
	case YLow, YHigh:
		expected = nx * nz
	case ZLow, ZHigh:
		expected = nx * ny
	default:
		return &ValidationError{Field: fieldFace, Message: msgUnknownFace}
	}

	if len(data.Values) != expected {
		return &SizeError{
			Expected: expected,
			Got:      len(data.Values),
			Context:  fmt.Sprintf("%v face values", data.Face),
		}
	}

	return nil
}

// isMixedAxisBC reports whether an axis carries a per-face-asymmetric
// (quarter-wave) boundary condition.
func isMixedAxisBC(b BCType) bool {
	return b == DirichletNeumann || b == NeumannDirichlet
}

// isLowFace reports whether a face is the low (index-0) face of its axis.
func isLowFace(face BoundaryFace) bool {
	switch face {
	case XLow, YLow, ZLow:
		return true
	case XHigh, YHigh, ZHigh:
		return false
	default:
		return false
	}
}

// expectedFaceType returns the BoundaryData.Type a face must carry for the given
// axis BC, and whether that axis accepts boundary data at all (Periodic does
// not). For a mixed axis the low and high faces require different types.
func expectedFaceType(axisBC BCType, face BoundaryFace) (BCType, bool) {
	switch axisBC {
	case Dirichlet, Neumann:
		return axisBC, true
	case DirichletNeumann:
		if isLowFace(face) {
			return Dirichlet, true
		}
		return Neumann, true
	case NeumannDirichlet:
		if isLowFace(face) {
			return Neumann, true
		}
		return Dirichlet, true
	case Periodic:
		return Periodic, false
	default:
		return axisBC, false
	}
}

func faceAxis(face BoundaryFace) (int, bool) {
	switch face {
	case XLow, XHigh:
		return 0, true
	case YLow, YHigh:
		return 1, true
	case ZLow, ZHigh:
		return 2, true
	default:
		return 0, false
	}
}
