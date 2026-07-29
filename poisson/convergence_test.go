package poisson_test

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-pde/poisson"
)

const convergenceMinRate = 1.8

func TestConvergence1D_Dirichlet(t *testing.T) {
	sizes := []int{32, 64, 128}
	errors := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h
		hs[idx] = h

		k := math.Pi / L
		lambda := k * k

		u := make([]float64, n)
		for i := range n {
			x := float64(i+1) * h
			u[i] = math.Sin(math.Pi * x / L)
		}

		plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.Dirichlet})
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		rhs := make([]float64, n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		got := make([]float64, n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errors[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errors)
}

func TestConvergence2D_Dirichlet(t *testing.T) {
	sizes := []int{16, 32, 64}
	errors := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n+1)
		L := float64(n+1) * h
		hs[idx] = h

		k := math.Pi / L
		lambda := 2.0 * k * k

		u := make([]float64, n*n)
		for i := range n {
			x := float64(i+1) * h
			for j := range n {
				y := float64(j+1) * h
				u[i*n+j] = math.Sin(math.Pi*x/L) * math.Sin(math.Pi*y/L)
			}
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

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		got := make([]float64, n*n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errors[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errors)
}

func TestConvergence2D_Neumann(t *testing.T) {
	sizes := []int{16, 32, 64}
	errs := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n)
		hs[idx] = h
		L := float64(n) * h

		kx := math.Pi / L
		ky := math.Pi / L
		lambda := kx*kx + ky*ky

		// cos(kx·x)cos(ky·y) has vanishing normal derivative on all faces (a
		// compatible Neumann field). Sample at cell centers (i+½)h.
		u := make([]float64, n*n)
		for i := range n {
			x := (float64(i) + 0.5) * h
			for j := range n {
				y := (float64(j) + 0.5) * h
				u[i*n+j] = math.Cos(kx*x) * math.Cos(ky*y)
			}
		}
		meanU := sliceMean(u)

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i] // continuous -Δu
		}

		plan, err := poisson.NewPlan(
			2,
			[]int{n, n},
			[]float64{h, h},
			[]poisson.BCType{poisson.Neumann, poisson.Neumann},
			poisson.WithSubtractMean(),
			poisson.WithSolutionMean(meanU),
		)
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		got := make([]float64, n*n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errs[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errs)
}

func TestConvergence2D_Mixed_DirichletNeumann(t *testing.T) {
	sizes := []int{16, 32, 64}
	errs := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		hx := 1.0 / float64(n+1) // Dirichlet axis: vertex-centered, L = (n+1)hx
		hy := 1.0 / float64(n)   // Neumann axis: cell-centered, L = n·hy
		hs[idx] = hx
		Lx := float64(n+1) * hx
		Ly := float64(n) * hy

		kx := math.Pi / Lx
		ky := math.Pi / Ly
		lambda := kx*kx + ky*ky

		// sin(kx·x) vanishes at the Dirichlet boundaries; cos(ky·y) has zero
		// slope at the Neumann boundaries.
		u := make([]float64, n*n)
		for i := range n {
			x := float64(i+1) * hx
			for j := range n {
				y := (float64(j) + 0.5) * hy
				u[i*n+j] = math.Sin(kx*x) * math.Cos(ky*y)
			}
		}

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		plan, err := poisson.NewPlan(
			2,
			[]int{n, n},
			[]float64{hx, hy},
			[]poisson.BCType{poisson.Dirichlet, poisson.Neumann},
		)
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		got := make([]float64, n*n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errs[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errs)
}

func TestConvergence1D_DirichletNeumann(t *testing.T) {
	// Quarter-wave grid (cell-centered, L = n·h). The lowest DirichletNeumann
	// mode sin(πx/2L) is zero at the Dirichlet low face (x=0) and has zero slope
	// at the Neumann high face (x=L).
	sizes := []int{16, 32, 64}
	errs := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n)
		hs[idx] = h
		L := float64(n) * h

		k := math.Pi / (2.0 * L)
		lambda := k * k

		u := make([]float64, n)
		for i := range n {
			x := (float64(i) + 0.5) * h
			u[i] = math.Sin(k * x)
		}

		rhs := make([]float64, n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, []poisson.BCType{poisson.DirichletNeumann})
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		got := make([]float64, n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errs[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errs)
}

func TestConvergence2D_Mixed_DNxND(t *testing.T) {
	// x: DirichletNeumann (sin(πx/2Lx), Dirichlet low / Neumann high);
	// y: NeumannDirichlet (cos(πy/2Ly), Neumann low / Dirichlet high).
	// Both axes use the quarter-wave cell-centered grid.
	sizes := []int{16, 32, 64}
	errs := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		hx := 1.0 / float64(n)
		hy := 0.8 / float64(n)
		hs[idx] = hx
		Lx := float64(n) * hx
		Ly := float64(n) * hy

		kx := math.Pi / (2.0 * Lx)
		ky := math.Pi / (2.0 * Ly)
		lambda := kx*kx + ky*ky

		u := make([]float64, n*n)
		for i := range n {
			x := (float64(i) + 0.5) * hx
			for j := range n {
				y := (float64(j) + 0.5) * hy
				u[i*n+j] = math.Sin(kx*x) * math.Cos(ky*y)
			}
		}

		rhs := make([]float64, n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		plan, err := poisson.NewPlan(
			2,
			[]int{n, n},
			[]float64{hx, hy},
			[]poisson.BCType{poisson.DirichletNeumann, poisson.NeumannDirichlet},
		)
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		got := make([]float64, n*n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errs[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errs)
}

func TestConvergence3D_Dirichlet(t *testing.T) {
	sizes := []int{8, 16, 32}
	errs := make([]float64, len(sizes))
	hs := make([]float64, len(sizes))

	for idx, n := range sizes {
		h := 1.0 / float64(n+1)
		hs[idx] = h
		L := float64(n+1) * h

		k := math.Pi / L
		lambda := 3.0 * k * k

		u := make([]float64, n*n*n)
		for i := range n {
			x := float64(i+1) * h
			for j := range n {
				y := float64(j+1) * h
				for kk := range n {
					z := float64(kk+1) * h
					u[(i*n+j)*n+kk] = math.Sin(k*x) * math.Sin(k*y) * math.Sin(k*z)
				}
			}
		}

		rhs := make([]float64, n*n*n)
		for i := range rhs {
			rhs[i] = lambda * u[i]
		}

		plan, err := poisson.NewPlan(
			3,
			[]int{n, n, n},
			[]float64{h, h, h},
			[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet, poisson.Dirichlet},
		)
		if err != nil {
			t.Fatalf("NewPlan failed: %v", err)
		}

		got := make([]float64, n*n*n)
		if err := plan.Solve(got, rhs); err != nil {
			t.Fatalf("Solve failed: %v", err)
		}

		errs[idx] = maxAbsDiff(got, u)
	}

	checkConvergenceRates(t, hs, errs)
}

func checkConvergenceRates(t *testing.T, hs, errors []float64) {
	t.Helper()

	for i := range len(errors) - 1 {
		if errors[i] == 0 || errors[i+1] == 0 {
			t.Fatalf("zero error encountered: %g -> %g", errors[i], errors[i+1])
		}
		rate := math.Log(errors[i+1]/errors[i]) / math.Log(hs[i+1]/hs[i])
		if rate < convergenceMinRate {
			t.Fatalf("convergence rate %.3f below threshold %.1f", rate, convergenceMinRate)
		}
	}
}
