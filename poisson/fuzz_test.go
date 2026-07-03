package poisson_test

import (
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

func FuzzPlanSolveBasic(f *testing.F) {
	f.Add(1, 8, 8, 8, 1.0, 1.0, 1.0, int(poisson.Periodic), int(poisson.Dirichlet), int(poisson.Neumann))
	f.Add(2, 6, 5, 4, 1.0, 0.5, 1.0, int(poisson.Dirichlet), int(poisson.Dirichlet), int(poisson.Dirichlet))
	f.Add(3, 4, 4, 4, 1.0, 1.0, 1.0, int(poisson.Periodic), int(poisson.Dirichlet), int(poisson.Neumann))

	f.Fuzz(func(t *testing.T, dim, nx, ny, nz int, hx, hy, hz float64, bc0, bc1, bc2 int) {
		if dim < 1 || dim > 3 {
			t.Skip()
		}

		// nz is now an independent dimension input (previously nz was tied to nx).
		// All extents are clamped to a small sane range so the solve stays cheap
		// and every axis satisfies the per-transform minimum size.
		hx, ok0 := clampSpacing(hx)
		hy, ok1 := clampSpacing(hy)
		hz, ok2 := clampSpacing(hz)
		if !ok0 || !ok1 || !ok2 {
			t.Skip()
		}

		n := []int{clampDim(nx), clampDim(ny), clampDim(nz)}
		h := []float64{hx, hy, hz}
		bc := []poisson.BCType{fuzzBC(bc0), fuzzBC(bc1), fuzzBC(bc2)}

		n = n[:dim]
		h = h[:dim]
		bc = bc[:dim]

		size := 1
		for _, v := range n {
			size *= v
		}

		nullspace := true
		for _, b := range bc {
			if !b.HasNullspace() {
				nullspace = false
				break
			}
		}

		var opts []poisson.Option
		if nullspace {
			// A fully Neumann/periodic problem is only solvable for a mean-zero
			// RHS; let the solver project it so the residual property is testable.
			opts = append(opts, poisson.WithSubtractMean())
		}

		plan, err := poisson.NewPlan(dim, n, h, bc, opts...)
		if err != nil {
			return
		}

		rhs := make([]float64, size)
		for i := range rhs {
			rhs[i] = math.Sin(float64(i)*0.5) + float64((i%7)-3)*0.1
		}

		dst := make([]float64, size)
		if err := plan.Solve(dst, rhs); err != nil {
			// A degenerate-but-in-range input may be rejected (e.g. an
			// inconsistent mean); that is acceptable. Just don't assert residual.
			return
		}

		// (a) The solution must be finite everywhere.
		for i, v := range dst {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("non-finite solution at %d: %g (dim=%d n=%v h=%v bc=%v)", i, v, dim, n, h, bc)
			}
		}

		// (b) Reapplying the finite-difference operator to the solution must
		// reproduce the RHS. For a nullspace problem the operator annihilates the
		// constant mode, so only the mean-projected RHS is reproducible.
		want := make([]float64, size)
		copy(want, rhs)
		if nullspace {
			mean := 0.0
			for _, v := range want {
				mean += v
			}
			mean /= float64(size)
			for i := range want {
				want[i] -= mean
			}
		}

		residual := make([]float64, size)
		switch dim {
		case 1:
			fd.Apply1D(residual, dst, h[0], bc[0])
		case 2:
			fd.Apply2D(residual, dst, grid.NewShape2D(n[0], n[1]),
				[2]float64{h[0], h[1]}, [2]poisson.BCType{bc[0], bc[1]})
		case 3:
			fd.Apply3D(residual, dst, grid.NewShape3D(n[0], n[1], n[2]),
				[3]float64{h[0], h[1], h[2]}, [3]poisson.BCType{bc[0], bc[1], bc[2]})
		}

		scale := 0.0
		for _, v := range want {
			if a := math.Abs(v); a > scale {
				scale = a
			}
		}
		if scale == 0 {
			scale = 1
		}

		maxErr := 0.0
		for i := range residual {
			if d := math.Abs(residual[i] - want[i]); d > maxErr {
				maxErr = d
			}
		}
		if rel := maxErr / scale; rel > 1e-6 {
			t.Fatalf("residual rel error %g exceeds 1e-6 (dim=%d n=%v h=%v bc=%v)", rel, dim, n, h, bc)
		}
	})
}

// clampDim folds an arbitrary fuzz integer into the small extent range [2, 10].
// The lower bound of 2 satisfies the DCT-I minimum used for Neumann axes.
func clampDim(v int) int {
	if v < 0 {
		v = -v
	}
	return v%9 + 2
}

// clampSpacing rejects non-positive / non-finite spacings and folds valid ones
// into [0.1, 10] to keep the discrete operator well conditioned.
func clampSpacing(h float64) (float64, bool) {
	if math.IsNaN(h) || math.IsInf(h, 0) || h <= 0 {
		return 0, false
	}
	for h < 0.1 {
		h *= 10
	}
	for h > 10 {
		h /= 10
	}
	return h, true
}

func fuzzBC(v int) poisson.BCType {
	switch v % 3 {
	case 0:
		return poisson.Periodic
	case 1:
		return poisson.Dirichlet
	default:
		return poisson.Neumann
	}
}
