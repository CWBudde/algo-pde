package fd

import (
	"fmt"

	"github.com/cwbudde/algo-pde/bc"
	"github.com/cwbudde/algo-pde/grid"
)

// Apply1D applies the 1D negative Laplacian stencil to src and writes into dst.
// The result is (2*u_i - u_{i-1} - u_{i+1}) / h^2 with boundary handling set by b.
// It is safe to call with dst == src.
//
// It returns ErrSizeMismatch if src is empty or dst does not have the same
// length as src, and ErrInvalidBC if b is not a supported boundary condition.
func Apply1D(dst, src []float64, h float64, b bc.BCType) error {
	if err := validBC(b); err != nil {
		return err
	}

	n := len(src)
	if n == 0 || len(dst) != n {
		return fmt.Errorf("%w: len(src)=%d len(dst)=%d", ErrSizeMismatch, len(src), len(dst))
	}

	invH2 := 1.0 / (h * h)
	if &dst[0] == &src[0] {
		tmp := make([]float64, n)
		copy(tmp, src)
		src = tmp
	}

	switch b {
	case bc.Periodic:
		for i := range n {
			left := src[(i-1+n)%n]
			right := src[(i+1)%n]
			dst[i] = (2.0*src[i] - left - right) * invH2
		}

	case bc.Dirichlet, bc.Neumann, bc.DirichletNeumann, bc.NeumannDirichlet:
		// All non-periodic BCs share one loop: interior neighbours are read
		// directly, and each boundary ghost is c·u_boundary where c is the
		// per-face reflection coefficient (0 vertex-Dirichlet, +1 Neumann,
		// -1 quarter-wave Dirichlet). This unifies pure and mixed axes.
		cLow := lowGhostCoeff(b)
		cHigh := highGhostCoeff(b)
		for i := range n {
			var left, right float64
			if i > 0 {
				left = src[i-1]
			} else {
				left = cLow * src[0]
			}

			if i+1 < n {
				right = src[i+1]
			} else {
				right = cHigh * src[n-1]
			}

			dst[i] = (2.0*src[i] - left - right) * invH2
		}
	}

	return nil
}

// Apply2D applies the 2D negative Laplacian stencil to src and writes into dst.
// The result is (2*u - u_{i-1} - u_{i+1})/hx^2 + (2*u - u_{j-1} - u_{j+1})/hy^2
// with per-axis boundary handling set by bcs. It is safe to call with dst == src.
//
// It returns ErrInvalidBC if either axis has an unsupported boundary condition
// and ErrSizeMismatch if dst/src do not match the expected nx*ny size.
func Apply2D(dst, src []float64, shape grid.Shape, h [2]float64, bcs [2]bc.BCType) error {
	for axis := range bcs {
		if err := validBC(bcs[axis]); err != nil {
			return err
		}
	}

	nx := shape.N(0)
	ny := shape.N(1)
	total := nx * ny
	if nx == 0 || ny == 0 || len(src) != total || len(dst) != total {
		return fmt.Errorf("%w: expected %d, len(src)=%d len(dst)=%d",
			ErrSizeMismatch, total, len(src), len(dst))
	}

	if &dst[0] == &src[0] {
		tmp := make([]float64, total)
		copy(tmp, src)
		src = tmp
	}

	invHx2 := 1.0 / (h[0] * h[0])
	invHy2 := 1.0 / (h[1] * h[1])

	for i := range nx {
		row := i * ny

		for j := range ny {
			idx := row + j
			u := src[idx]

			var left, right float64
			switch {
			case i > 0:
				left = src[(i-1)*ny+j]
			case bcs[0] == bc.Periodic:
				left = src[(nx-1)*ny+j]
			default:
				left = lowGhostCoeff(bcs[0]) * src[idx]
			}

			switch {
			case i+1 < nx:
				right = src[(i+1)*ny+j]
			case bcs[0] == bc.Periodic:
				right = src[j]
			default:
				right = highGhostCoeff(bcs[0]) * src[idx]
			}

			var down, up float64
			switch {
			case j > 0:
				down = src[row+j-1]
			case bcs[1] == bc.Periodic:
				down = src[row+ny-1]
			default:
				down = lowGhostCoeff(bcs[1]) * src[idx]
			}

			switch {
			case j+1 < ny:
				up = src[row+j+1]
			case bcs[1] == bc.Periodic:
				up = src[row]
			default:
				up = highGhostCoeff(bcs[1]) * src[idx]
			}

			dst[idx] = (2.0*u-left-right)*invHx2 + (2.0*u-down-up)*invHy2
		}
	}

	return nil
}

// Apply3D applies the 3D negative Laplacian stencil to src and writes into dst.
// The result sums 1D stencils in x/y/z with per-axis boundary handling set by bcs.
// It is safe to call with dst == src.
//
// It returns ErrInvalidBC if any axis has an unsupported boundary condition and
// ErrSizeMismatch if dst/src do not match the expected nx*ny*nz size.
func Apply3D(dst, src []float64, shape grid.Shape, h [3]float64, bcs [3]bc.BCType) error {
	for axis := range bcs {
		if err := validBC(bcs[axis]); err != nil {
			return err
		}
	}

	nx := shape.N(0)
	ny := shape.N(1)
	nz := shape.N(2)
	total := nx * ny * nz
	if nx == 0 || ny == 0 || nz == 0 || len(src) != total || len(dst) != total {
		return fmt.Errorf("%w: expected %d, len(src)=%d len(dst)=%d",
			ErrSizeMismatch, total, len(src), len(dst))
	}

	if &dst[0] == &src[0] {
		tmp := make([]float64, total)
		copy(tmp, src)
		src = tmp
	}

	invHx2 := 1.0 / (h[0] * h[0])
	invHy2 := 1.0 / (h[1] * h[1])
	invHz2 := 1.0 / (h[2] * h[2])

	plane := ny * nz
	for i := range nx {
		iPlane := i * plane
		for j := range ny {
			row := iPlane + j*nz
			for k := range nz {
				idx := row + k
				u := src[idx]

				var left, right float64
				switch {
				case i > 0:
					left = src[idx-plane]
				case bcs[0] == bc.Periodic:
					left = src[idx+plane*(nx-1)]
				default:
					left = lowGhostCoeff(bcs[0]) * src[idx]
				}

				switch {
				case i+1 < nx:
					right = src[idx+plane]
				case bcs[0] == bc.Periodic:
					right = src[idx-plane*(nx-1)]
				default:
					right = highGhostCoeff(bcs[0]) * src[idx]
				}

				var down, up float64
				switch {
				case j > 0:
					down = src[idx-nz]
				case bcs[1] == bc.Periodic:
					down = src[row+(ny-1)*nz+k]
				default:
					down = lowGhostCoeff(bcs[1]) * src[idx]
				}

				switch {
				case j+1 < ny:
					up = src[idx+nz]
				case bcs[1] == bc.Periodic:
					up = src[iPlane+k]
				default:
					up = highGhostCoeff(bcs[1]) * src[idx]
				}

				var back, front float64
				switch {
				case k > 0:
					back = src[idx-1]
				case bcs[2] == bc.Periodic:
					back = src[row+nz-1]
				default:
					back = lowGhostCoeff(bcs[2]) * src[idx]
				}

				switch {
				case k+1 < nz:
					front = src[idx+1]
				case bcs[2] == bc.Periodic:
					front = src[row]
				default:
					front = highGhostCoeff(bcs[2]) * src[idx]
				}

				dst[idx] = (2.0*u-left-right)*invHx2 +
					(2.0*u-down-up)*invHy2 +
					(2.0*u-back-front)*invHz2
			}
		}
	}

	return nil
}

// validBC returns ErrInvalidBC if b is not a supported boundary condition.
func validBC(b bc.BCType) error {
	switch b {
	case bc.Periodic, bc.Dirichlet, bc.Neumann, bc.DirichletNeumann, bc.NeumannDirichlet:
		return nil
	default:
		return fmt.Errorf("%w: %v", ErrInvalidBC, b)
	}
}

// lowGhostCoeff returns the reflection coefficient c for the ghost node just
// outside the low (index-0) boundary of a non-periodic axis, so that the ghost
// value is c·u_0. Even reflection (c=+1) enforces zero derivative (Neumann);
// odd reflection about the half-cell boundary (c=−1) enforces zero value on the
// quarter-wave grid (a Dirichlet face of a mixed axis); c=0 is the
// vertex-centred homogeneous Dirichlet (the ghost is the boundary node, value
// 0). Periodic axes are handled separately and never call this.
func lowGhostCoeff(b bc.BCType) float64 {
	switch b {
	case bc.Neumann, bc.NeumannDirichlet:
		return 1
	case bc.DirichletNeumann:
		return -1
	case bc.Dirichlet, bc.Periodic:
		return 0
	default:
		return 0
	}
}

// highGhostCoeff is the low-face analogue for the high (index n-1) boundary.
func highGhostCoeff(b bc.BCType) float64 {
	switch b {
	case bc.Neumann, bc.DirichletNeumann:
		return 1
	case bc.NeumannDirichlet:
		return -1
	case bc.Dirichlet, bc.Periodic:
		return 0
	default:
		return 0
	}
}
