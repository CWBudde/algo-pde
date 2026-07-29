package poisson

import "github.com/cwbudde/algo-pde/grid"

// ApplyDirichletRHS adds inhomogeneous Dirichlet boundary contributions to rhs
// for a vertex-centered Dirichlet axis (ghost = the boundary value). The rhs
// slice is modified in-place and uses row-major ordering.
func ApplyDirichletRHS(rhs []float64, shape grid.Shape, h [3]float64, bc BoundaryConditions) error {
	return applyDirichletRHS(rhs, shape, h, bc, 1.0)
}

// applyDirichletRHS is the shared implementation. scale is the ghost-reflection
// factor: 1 for a vertex-centered Dirichlet axis (ghost = g) and 2 for the
// Dirichlet face of a mixed quarter-wave axis, whose ghost is u₋₁ = 2g − u₀, so
// the boundary row gains 2g/h² instead of g/h².
func applyDirichletRHS(rhs []float64, shape grid.Shape, h [3]float64, bc BoundaryConditions, scale float64) error {
	if rhs == nil {
		return ErrNilBuffer
	}

	expected := shape.Size()
	if len(rhs) != expected {
		return &SizeError{
			Expected: expected,
			Got:      len(rhs),
			Context:  "ApplyDirichletRHS",
		}
	}

	dim := shape.Dim()
	nx, ny, nz := shape.N(0), shape.N(1), shape.N(2)
	plane := ny * nz

	for _, data := range bc {
		if data.Type != Dirichlet {
			return &ValidationError{
				Field:   fieldType,
				Message: "only Dirichlet boundary data is supported",
			}
		}

		switch data.Face {
		case XLow, XHigh:
			if dim < 1 {
				return &ValidationError{Field: fieldFace, Message: "X face not valid for this dimension"}
			}
			expectedFace := ny * nz
			if len(data.Values) != expectedFace {
				return &SizeError{
					Expected: expectedFace,
					Got:      len(data.Values),
					Context:  "X face values",
				}
			}

			invHx2 := 1.0 / (h[0] * h[0])
			base := 0
			if data.Face == XHigh {
				base = (nx - 1) * plane
			}
			for j := range ny {
				row := base + j*nz
				valRow := j * nz
				for k := range nz {
					rhs[row+k] += data.Values[valRow+k] * invHx2 * scale
				}
			}

		case YLow, YHigh:
			if dim < 2 {
				return &ValidationError{Field: fieldFace, Message: "Y face not valid for this dimension"}
			}
			expectedFace := nx * nz
			if len(data.Values) != expectedFace {
				return &SizeError{
					Expected: expectedFace,
					Got:      len(data.Values),
					Context:  "Y face values",
				}
			}

			invHy2 := 1.0 / (h[1] * h[1])
			j := 0
			if data.Face == YHigh {
				j = ny - 1
			}
			for i := range nx {
				base := i*plane + j*nz
				valRow := i * nz
				for k := range nz {
					rhs[base+k] += data.Values[valRow+k] * invHy2 * scale
				}
			}

		case ZLow, ZHigh:
			if dim < 3 {
				return &ValidationError{Field: fieldFace, Message: "Z face not valid for this dimension"}
			}
			expectedFace := nx * ny
			if len(data.Values) != expectedFace {
				return &SizeError{
					Expected: expectedFace,
					Got:      len(data.Values),
					Context:  "Z face values",
				}
			}

			invHz2 := 1.0 / (h[2] * h[2])
			k := 0
			if data.Face == ZHigh {
				k = nz - 1
			}
			for i := range nx {
				base := i * plane
				valRow := i * ny
				for j := range ny {
					rhs[base+j*nz+k] += data.Values[valRow+j] * invHz2 * scale
				}
			}

		default:
			return &ValidationError{Field: fieldFace, Message: msgUnknownFace}
		}
	}

	return nil
}
