package fd

import (
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// Apply1D applies the 1D negative Laplacian stencil to src and writes into dst.
// The result is (2*u_i - u_{i-1} - u_{i+1}) / h^2 with boundary handling set by bc.
// It is safe to call with dst == src.
func Apply1D(dst, src []float64, h float64, bc poisson.BCType) {
	n := len(src)
	if n == 0 || len(dst) != n {
		return
	}

	invH2 := 1.0 / (h * h)
	if &dst[0] == &src[0] {
		tmp := make([]float64, n)
		copy(tmp, src)
		src = tmp
	}

	switch bc {
	case poisson.Periodic:
		for i := range n {
			left := src[(i-1+n)%n]
			right := src[(i+1)%n]
			dst[i] = (2.0*src[i] - left - right) * invH2
		}

	case poisson.Dirichlet:
		for i := range n {
			left := 0.0
			if i > 0 {
				left = src[i-1]
			}

			right := 0.0
			if i+1 < n {
				right = src[i+1]
			}

			dst[i] = (2.0*src[i] - left - right) * invH2
		}

	case poisson.Neumann:
		for i := range n {
			var left, right float64
			switch {
			case i == 0:
				left = src[0]
				if n == 1 {
					right = src[0]
				} else {
					right = src[1]
				}
			case i == n-1:
				left = src[n-2]
				right = src[n-1]
			default:
				left = src[i-1]
				right = src[i+1]
			}

			dst[i] = (2.0*src[i] - left - right) * invH2
		}
	}
}

// Apply2D applies the 2D negative Laplacian stencil to src and writes into dst.
// The result is (2*u - u_{i-1} - u_{i+1})/hx^2 + (2*u - u_{j-1} - u_{j+1})/hy^2
// with per-axis boundary handling set by bc. It is safe to call with dst == src.
func Apply2D(dst, src []float64, shape grid.Shape, h [2]float64, bc [2]poisson.BCType) {
	nx := shape[0]
	ny := shape[1]
	if nx == 0 || ny == 0 {
		return
	}

	total := nx * ny
	if len(src) != total || len(dst) != total {
		return
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
			case bc[0] == poisson.Periodic:
				left = src[(nx-1)*ny+j]
			case bc[0] == poisson.Neumann:
				left = src[idx]
			default:
				left = 0
			}

			switch {
			case i+1 < nx:
				right = src[(i+1)*ny+j]
			case bc[0] == poisson.Periodic:
				right = src[j]
			case bc[0] == poisson.Neumann:
				right = src[idx]
			default:
				right = 0
			}

			var down, up float64
			switch {
			case j > 0:
				down = src[row+j-1]
			case bc[1] == poisson.Periodic:
				down = src[row+ny-1]
			case bc[1] == poisson.Neumann:
				down = src[idx]
			default:
				down = 0
			}

			switch {
			case j+1 < ny:
				up = src[row+j+1]
			case bc[1] == poisson.Periodic:
				up = src[row]
			case bc[1] == poisson.Neumann:
				up = src[idx]
			default:
				up = 0
			}

			dst[idx] = (2.0*u-left-right)*invHx2 + (2.0*u-down-up)*invHy2
		}
	}
}

// Apply3D applies the 3D negative Laplacian stencil to src and writes into dst.
// The result sums 1D stencils in x/y/z with per-axis boundary handling set by bc.
// It is safe to call with dst == src.
func Apply3D(dst, src []float64, shape grid.Shape, h [3]float64, bc [3]poisson.BCType) {
	nx := shape[0]
	ny := shape[1]
	nz := shape[2]
	if nx == 0 || ny == 0 || nz == 0 {
		return
	}

	total := nx * ny * nz
	if len(src) != total || len(dst) != total {
		return
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
				case bc[0] == poisson.Periodic:
					left = src[idx+plane*(nx-1)]
				case bc[0] == poisson.Neumann:
					left = src[idx]
				default:
					left = 0
				}

				switch {
				case i+1 < nx:
					right = src[idx+plane]
				case bc[0] == poisson.Periodic:
					right = src[idx-plane*(nx-1)]
				case bc[0] == poisson.Neumann:
					right = src[idx]
				default:
					right = 0
				}

				var down, up float64
				switch {
				case j > 0:
					down = src[idx-nz]
				case bc[1] == poisson.Periodic:
					down = src[row+(ny-1)*nz+k]
				case bc[1] == poisson.Neumann:
					down = src[idx]
				default:
					down = 0
				}

				switch {
				case j+1 < ny:
					up = src[idx+nz]
				case bc[1] == poisson.Periodic:
					up = src[iPlane+k]
				case bc[1] == poisson.Neumann:
					up = src[idx]
				default:
					up = 0
				}

				var back, front float64
				switch {
				case k > 0:
					back = src[idx-1]
				case bc[2] == poisson.Periodic:
					back = src[row+nz-1]
				case bc[2] == poisson.Neumann:
					back = src[idx]
				default:
					back = 0
				}

				switch {
				case k+1 < nz:
					front = src[idx+1]
				case bc[2] == poisson.Periodic:
					front = src[row]
				case bc[2] == poisson.Neumann:
					front = src[idx]
				default:
					front = 0
				}

				dst[idx] = (2.0*u-left-right)*invHx2 +
					(2.0*u-down-up)*invHy2 +
					(2.0*u-back-front)*invHz2
			}
		}
	}
}
