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

	if err := p.validateBoundaryConditions(bc); err != nil {
		return err
	}

	buf := rhs
	if !p.opts.InPlace {
		if len(p.work.Real) < size {
			p.work.Real = make([]float64, size)
		}
		buf = p.work.Real[:size]
		copy(buf, rhs)
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
				Field:   "Type",
				Message: "unsupported boundary condition",
			}
		}
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

	return p.Solve(dst, buf)
}

func (p *Plan) validateBoundaryConditions(bc BoundaryConditions) error {
	for _, data := range bc {
		axis, ok := faceAxis(data.Face)
		if !ok || axis >= p.dim {
			return &ValidationError{
				Field:   "Face",
				Message: "boundary face not valid for plan dimension",
			}
		}

		if p.bc[axis] == Periodic {
			return &ValidationError{
				Field:   "Face",
				Message: "boundary data not allowed for periodic axis",
			}
		}

		if p.bc[axis] != data.Type {
			return &ValidationError{
				Field:   "Type",
				Message: fmt.Sprintf("boundary type %s does not match plan axis %s", data.Type, p.bc[axis]),
			}
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
