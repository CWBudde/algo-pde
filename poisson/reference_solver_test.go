package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const referenceSolverTol = 1e-10

func TestReferenceSolve2D_Dirichlet(t *testing.T) {
	for _, n := range []int{8, 16} {
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h

		u := make([]float64, n*n)
		for i := range n {
			x := float64(i+1) * h
			for j := range n {
				y := float64(j+1) * h
				u[i*n+j] = math.Sin(math.Pi*x/L) * math.Sin(math.Pi*y/L)
			}
		}

		rhs := make([]float64, n*n)
		fd.Apply2D(rhs, u, grid.NewShape2D(n, n), [2]float64{h, h}, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet})

		plan, err := poisson.NewPlan(
			2,
			[]int{n, n},
			[]float64{h, h},
			[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet},
		)
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		spectral := make([]float64, n*n)
		if err := plan.Solve(spectral, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		dense := solveDensePoisson2DDirichlet(n, n, h, h, rhs)

		if max := maxAbsDiff(dense, spectral); max > referenceSolverTol {
			t.Fatalf("n=%d max spectral-dense error %g exceeds tol %g", n, max, referenceSolverTol)
		}

		if max := maxAbsDiff(dense, u); max > referenceSolverTol {
			t.Fatalf("n=%d max dense-manufactured error %g exceeds tol %g", n, max, referenceSolverTol)
		}
	}
}

func solveDensePoisson2DDirichlet(nx, ny int, hx, hy float64, rhs []float64) []float64 {
	n := nx * ny
	a := make([]float64, n*n)
	b := make([]float64, n)
	copy(b, rhs)

	invHx2 := 1.0 / (hx * hx)
	invHy2 := 1.0 / (hy * hy)

	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			idx := i*ny + j
			a[idx*n+idx] = 2.0*invHx2 + 2.0*invHy2

			if i > 0 {
				a[idx*n+(idx-ny)] = -invHx2
			}
			if i+1 < nx {
				a[idx*n+(idx+ny)] = -invHx2
			}
			if j > 0 {
				a[idx*n+(idx-1)] = -invHy2
			}
			if j+1 < ny {
				a[idx*n+(idx+1)] = -invHy2
			}
		}
	}

	return solveDenseLinearSystem(a, b)
}

func solveDenseLinearSystem(a []float64, b []float64) []float64 {
	n := len(b)
	if len(a) != n*n {
		return nil
	}

	for k := 0; k < n; k++ {
		pivotRow := k
		pivotVal := math.Abs(a[k*n+k])
		for i := k + 1; i < n; i++ {
			val := math.Abs(a[i*n+k])
			if val > pivotVal {
				pivotVal = val
				pivotRow = i
			}
		}

		if pivotVal == 0 {
			return nil
		}

		if pivotRow != k {
			for j := k; j < n; j++ {
				a[k*n+j], a[pivotRow*n+j] = a[pivotRow*n+j], a[k*n+j]
			}
			b[k], b[pivotRow] = b[pivotRow], b[k]
		}

		pivot := a[k*n+k]
		for i := k + 1; i < n; i++ {
			factor := a[i*n+k] / pivot
			if factor == 0 {
				continue
			}
			a[i*n+k] = 0
			for j := k + 1; j < n; j++ {
				a[i*n+j] -= factor * a[k*n+j]
			}
			b[i] -= factor * b[k]
		}
	}

	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := b[i]
		for j := i + 1; j < n; j++ {
			sum -= a[i*n+j] * x[j]
		}
		x[i] = sum / a[i*n+i]
	}

	return x
}
