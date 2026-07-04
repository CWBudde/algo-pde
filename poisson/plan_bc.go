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

	var dirichlet, neumann BoundaryConditions
	for _, data := range bc {
		switch data.Type {
		case Dirichlet:
			dirichlet = append(dirichlet, data)
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

		if p.bc[axis] == Periodic {
			return &ValidationError{
				Field:   fieldFace,
				Message: "boundary data not allowed for periodic axis",
			}
		}

		if p.bc[axis] != data.Type {
			return &ValidationError{
				Field:   fieldType,
				Message: fmt.Sprintf("boundary type %s does not match plan axis %s", data.Type, p.bc[axis]),
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
