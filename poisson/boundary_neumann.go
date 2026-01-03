package poisson

import "github.com/MeKo-Tech/algo-pde/grid"

// ApplyNeumannRHS adds inhomogeneous Neumann boundary contributions to rhs.
// Values represent the derivative along the positive axis direction at each face.
// The rhs slice is modified in-place and uses row-major ordering.
func ApplyNeumannRHS(rhs []float64, shape grid.Shape, h [3]float64, bc BoundaryConditions) error {
	if rhs == nil {
		return ErrNilBuffer
	}

	expected := shape.Size()
	if len(rhs) != expected {
		return &SizeError{
			Expected: expected,
			Got:      len(rhs),
			Context:  "ApplyNeumannRHS",
		}
	}

	dim := shape.Dim()
	nx, ny, nz := shape[0], shape[1], shape[2]
	plane := ny * nz

	for _, data := range bc {
		if data.Type != Neumann {
			return &ValidationError{
				Field:   "Type",
				Message: "only Neumann boundary data is supported",
			}
		}

		switch data.Face {
		case XLow, XHigh:
			if dim < 1 {
				return &ValidationError{Field: "Face", Message: "X face not valid for this dimension"}
			}
			expectedFace := ny * nz
			if len(data.Values) != expectedFace {
				return &SizeError{
					Expected: expectedFace,
					Got:      len(data.Values),
					Context:  "X face values",
				}
			}

			invHx := 1.0 / h[0]
			scale := -invHx
			if data.Face == XHigh {
				scale = invHx
			}

			base := 0
			if data.Face == XHigh {
				base = (nx - 1) * plane
			}
			for j := 0; j < ny; j++ {
				row := base + j*nz
				valRow := j * nz
				for k := 0; k < nz; k++ {
					rhs[row+k] += data.Values[valRow+k] * scale
				}
			}

		case YLow, YHigh:
			if dim < 2 {
				return &ValidationError{Field: "Face", Message: "Y face not valid for this dimension"}
			}
			expectedFace := nx * nz
			if len(data.Values) != expectedFace {
				return &SizeError{
					Expected: expectedFace,
					Got:      len(data.Values),
					Context:  "Y face values",
				}
			}

			invHy := 1.0 / h[1]
			scale := -invHy
			if data.Face == YHigh {
				scale = invHy
			}

			j := 0
			if data.Face == YHigh {
				j = ny - 1
			}
			for i := 0; i < nx; i++ {
				base := i*plane + j*nz
				valRow := i * nz
				for k := 0; k < nz; k++ {
					rhs[base+k] += data.Values[valRow+k] * scale
				}
			}

		case ZLow, ZHigh:
			if dim < 3 {
				return &ValidationError{Field: "Face", Message: "Z face not valid for this dimension"}
			}
			expectedFace := nx * ny
			if len(data.Values) != expectedFace {
				return &SizeError{
					Expected: expectedFace,
					Got:      len(data.Values),
					Context:  "Z face values",
				}
			}

			invHz := 1.0 / h[2]
			scale := -invHz
			if data.Face == ZHigh {
				scale = invHz
			}

			k := 0
			if data.Face == ZHigh {
				k = nz - 1
			}
			for i := 0; i < nx; i++ {
				base := i * plane
				valRow := i * ny
				for j := 0; j < ny; j++ {
					rhs[base+j*nz+k] += data.Values[valRow+j] * scale
				}
			}

		default:
			return &ValidationError{Field: "Face", Message: "unknown boundary face"}
		}
	}

	return nil
}
