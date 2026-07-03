package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

const (
	referenceSolverTol = 1e-10
	referenceRandomTol = 1e-9
)

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
		if err := fd.Apply2D(rhs, u, grid.NewShape2D(n, n), [2]float64{h, h}, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}); err != nil {
			t.Fatal(err)
		}

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

// TestReferenceSolve2D_Random compares the spectral solver against a dense
// Gaussian-elimination reference on a deterministic RANDOM right-hand side. A
// random RHS excites every mode, so a single mishandled eigenvalue/transform in
// any BC combination shows up as a mismatch. The dense operator is built by
// applying the library's own finite-difference stencil (fd.Apply2D, the negative
// Laplacian) to unit basis vectors, guaranteeing the two paths solve the same
// discrete system. The dense solve (partial-pivot elimination) is fully
// independent of the spectral pipeline.
func TestReferenceSolve2D_Random(t *testing.T) {
	cases := []struct {
		name   string
		nx, ny int
		hx, hy float64
		bc     [2]poisson.BCType
		alpha  float64
	}{
		{"Neumann", 10, 8, 1.0 / 10, 1.0 / 8, [2]poisson.BCType{poisson.Neumann, poisson.Neumann}, 0},
		{"Periodic", 8, 12, 1.0 / 8, 1.0 / 12, [2]poisson.BCType{poisson.Periodic, poisson.Periodic}, 0},
		{"DirichletNeumann", 12, 10, 1.0 / 13, 1.0 / 10, [2]poisson.BCType{poisson.Dirichlet, poisson.Neumann}, 0},
		{"AnisotropicDirichlet", 10, 14, 0.5 / 11, 1.3 / 15, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, 0},
		{"HelmholtzDirichlet", 10, 12, 1.0 / 11, 1.0 / 13, [2]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, 3.5},
	}

	seed := int64(20260703)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			N := tc.nx * tc.ny
			raw := randomField(seed, N)
			seed++

			bcSlice := []poisson.BCType{tc.bc[0], tc.bc[1]}
			nullspace := allNullspace(bcSlice) && tc.alpha == 0

			// Dense system.
			a := buildDenseOperator2D(tc.nx, tc.ny, tc.hx, tc.hy, tc.bc, tc.alpha)
			b := make([]float64, N)
			copy(b, raw)

			var opts []poisson.Option
			if nullspace {
				// Project the RHS onto the range of the singular operator...
				mean := sliceMean(b)
				for i := range b {
					b[i] -= mean
				}
				// ...and pin the constant nullspace by replacing equation 0 with
				// the mean-zero constraint sum(x)=0, giving a nonsingular dense
				// system whose unique solution has zero mean.
				for c := range N {
					a[c] = 1
				}
				b[0] = 0
				opts = append(opts, poisson.WithSubtractMean())
			}

			dense := solveDenseLinearSystem(a, b)
			if dense == nil {
				t.Fatalf("dense solve returned nil (singular system)")
			}

			var (
				plan *poisson.Plan
				err  error
			)
			if tc.alpha == 0 {
				plan, err = poisson.NewPlan(2, []int{tc.nx, tc.ny}, []float64{tc.hx, tc.hy}, bcSlice, opts...)
			} else {
				plan, err = poisson.NewHelmholtzPlan(2, []int{tc.nx, tc.ny}, []float64{tc.hx, tc.hy}, bcSlice, tc.alpha, opts...)
			}
			if err != nil {
				t.Fatalf("plan construction failed: %v", err)
			}

			solveRHS := make([]float64, N)
			copy(solveRHS, raw)

			spectral := make([]float64, N)
			if err := plan.Solve(spectral, solveRHS); err != nil {
				t.Fatalf("spectral solve failed: %v", err)
			}

			if nullspace {
				// The dense and spectral solutions are each defined only up to an
				// additive constant; compare after removing their means.
				subtractMeanInPlace(dense)
				subtractMeanInPlace(spectral)
			}

			scale := 0.0
			for _, v := range dense {
				if av := math.Abs(v); av > scale {
					scale = av
				}
			}
			if scale == 0 {
				scale = 1
			}

			if rel := maxAbsDiff(dense, spectral) / scale; rel > referenceRandomTol {
				t.Fatalf("spectral-dense relative error %g exceeds tol %g", rel, referenceRandomTol)
			}
		})
	}
}

// buildDenseOperator2D materializes the N×N matrix of the discrete operator
// (α·I − Δ) for the given per-axis BCs by applying fd.Apply2D (the negative
// Laplacian) to each unit basis vector. Column c is the operator's action on
// e_c, so a[r*N+c] is the (r,c) entry.
func buildDenseOperator2D(nx, ny int, hx, hy float64, bc [2]poisson.BCType, alpha float64) []float64 {
	N := nx * ny
	a := make([]float64, N*N)
	e := make([]float64, N)
	col := make([]float64, N)
	shape := grid.NewShape2D(nx, ny)

	for c := range N {
		for i := range e {
			e[i] = 0
		}
		e[c] = 1

		if err := fd.Apply2D(col, e, shape, [2]float64{hx, hy}, bc); err != nil {
			panic(err)
		}
		for r := range N {
			a[r*N+c] = col[r]
		}
		a[c*N+c] += alpha
	}

	return a
}

func subtractMeanInPlace(v []float64) {
	m := sliceMean(v)
	for i := range v {
		v[i] -= m
	}
}

func solveDensePoisson2DDirichlet(nx, ny int, hx, hy float64, rhs []float64) []float64 {
	n := nx * ny
	a := make([]float64, n*n)
	b := make([]float64, n)
	copy(b, rhs)

	invHx2 := 1.0 / (hx * hx)
	invHy2 := 1.0 / (hy * hy)

	for i := range nx {
		for j := range ny {
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

	for k := range n {
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
