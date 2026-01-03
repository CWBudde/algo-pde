package poisson_test

import (
	"testing"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

func FuzzPlanSolveBasic(f *testing.F) {
	f.Add(1, 8, 8, 1.0, 1.0, 1.0, int(poisson.Periodic), int(poisson.Dirichlet), int(poisson.Neumann))
	f.Add(2, 6, 5, 1.0, 0.5, 1.0, int(poisson.Dirichlet), int(poisson.Dirichlet), int(poisson.Dirichlet))
	f.Add(3, 4, 4, 1.0, 1.0, 1.0, int(poisson.Periodic), int(poisson.Dirichlet), int(poisson.Neumann))

	f.Fuzz(func(t *testing.T, dim, nx, ny int, hx, hy, hz float64, bc0, bc1, bc2 int) {
		if dim < 1 || dim > 3 {
			t.Skip()
		}

		if nx < 1 || ny < 1 {
			t.Skip()
		}

		if hx <= 0 || hy <= 0 || hz <= 0 {
			t.Skip()
		}

		n := []int{nx, ny, nx}
		h := []float64{hx, hy, hz}
		bc := []poisson.BCType{
			fuzzBC(bc0),
			fuzzBC(bc1),
			fuzzBC(bc2),
		}

		n = n[:dim]
		h = h[:dim]
		bc = bc[:dim]

		plan, err := poisson.NewPlan(dim, n, h, bc)
		if err != nil {
			return
		}

		size := 1
		for _, v := range n {
			size *= v
		}
		if size == 0 {
			return
		}

		rhs := make([]float64, size)
		for i := range rhs {
			rhs[i] = float64((i%7)-3) * 0.1
		}

		dst := make([]float64, size)
		_ = plan.Solve(dst, rhs)
	})
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
