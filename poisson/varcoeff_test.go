package poisson_test

import (
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/CWBudde/algo-pde/fd"
	"github.com/CWBudde/algo-pde/grid"
	"github.com/CWBudde/algo-pde/poisson"
)

// positiveCoeff builds a strictly positive, spatially varying coefficient field
// in [1.1, 1.9) from a deterministic seed.
func positiveCoeff(seed int64, size int) []float64 {
	raw := randomField(seed, size)
	a := make([]float64, size)
	for i := range a {
		a[i] = 1.5 + 0.4*raw[i]
	}
	return a
}

// TestVariableCoeff_ReducesToFDApply pins the operator against the constant-
// coefficient fd stencil: with a ≡ 1, L_a must equal fd.Apply exactly.
func TestVariableCoeff_ReducesToFDApply(t *testing.T) {
	t.Run("1D", func(t *testing.T) {
		const n = 16
		ones := make([]float64, n)
		for i := range ones {
			ones[i] = 1
		}
		src := randomField(1, n)
		for _, b := range residualBCTypes {
			t.Run(bcName(b), func(t *testing.T) {
				plan, err := poisson.NewVariableCoeffPlan(1, []int{n}, []float64{0.1}, []poisson.BCType{b}, ones)
				if err != nil {
					t.Fatalf("NewVariableCoeffPlan: %v", err)
				}
				got := make([]float64, n)
				if err := plan.ApplyOperator(got, src); err != nil {
					t.Fatalf("ApplyOperator: %v", err)
				}
				want := make([]float64, n)
				if err := fd.Apply1D(want, src, 0.1, b); err != nil {
					t.Fatalf("fd.Apply1D: %v", err)
				}
				if d := maxAbsDiff(got, want); d > 1e-12 {
					t.Fatalf("L_a(a=1) != fd.Apply1D: max diff %.3e", d)
				}
			})
		}
	})

	t.Run("2D", func(t *testing.T) {
		const nx, ny = 12, 14
		size := nx * ny
		ones := make([]float64, size)
		for i := range ones {
			ones[i] = 1
		}
		src := randomField(2, size)
		h := [2]float64{0.1, 0.13}
		shape := grid.NewShape2D(nx, ny)
		pairs := [][2]poisson.BCType{
			{poisson.Dirichlet, poisson.Neumann},
			{poisson.Periodic, poisson.Dirichlet},
			{poisson.DirichletNeumann, poisson.NeumannDirichlet},
		}
		for _, bcs := range pairs {
			t.Run(bcName(bcs[0])+"_"+bcName(bcs[1]), func(t *testing.T) {
				plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, h[:], bcs[:], ones)
				if err != nil {
					t.Fatalf("NewVariableCoeffPlan: %v", err)
				}
				got := make([]float64, size)
				if err := plan.ApplyOperator(got, src); err != nil {
					t.Fatalf("ApplyOperator: %v", err)
				}
				want := make([]float64, size)
				if err := fd.Apply2D(want, src, shape, h, bcs); err != nil {
					t.Fatalf("fd.Apply2D: %v", err)
				}
				if d := maxAbsDiff(got, want); d > 1e-12 {
					t.Fatalf("L_a(a=1) != fd.Apply2D: max diff %.3e", d)
				}
			})
		}
	})

	t.Run("3D", func(t *testing.T) {
		const nx, ny, nz = 6, 8, 6
		size := nx * ny * nz
		ones := make([]float64, size)
		for i := range ones {
			ones[i] = 1
		}
		src := randomField(3, size)
		h := [3]float64{0.1, 0.12, 0.08}
		shape := grid.NewShape3D(nx, ny, nz)
		bcs := [3]poisson.BCType{poisson.Dirichlet, poisson.Periodic, poisson.Neumann}
		plan, err := poisson.NewVariableCoeffPlan(3, []int{nx, ny, nz}, h[:], bcs[:], ones)
		if err != nil {
			t.Fatalf("NewVariableCoeffPlan: %v", err)
		}
		got := make([]float64, size)
		if err := plan.ApplyOperator(got, src); err != nil {
			t.Fatalf("ApplyOperator: %v", err)
		}
		want := make([]float64, size)
		if err := fd.Apply3D(want, src, shape, h, bcs); err != nil {
			t.Fatalf("fd.Apply3D: %v", err)
		}
		if d := maxAbsDiff(got, want); d > 1e-12 {
			t.Fatalf("L_a(a=1) != fd.Apply3D: max diff %.3e", d)
		}
	})
}

// TestVariableCoeff_ApplyOperatorAliased checks that ApplyOperator produces the
// same result whether or not dst aliases src (the aliased path copies src into
// pooled scratch first).
func TestVariableCoeff_ApplyOperatorAliased(t *testing.T) {
	const nx, ny = 12, 14
	size := nx * ny
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Neumann}
	a := positiveCoeff(31, size)
	plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{0.1, 0.13}, bcs, a)
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}

	src := randomField(41, size)
	want := make([]float64, size)
	if err := plan.ApplyOperator(want, src); err != nil {
		t.Fatalf("ApplyOperator: %v", err)
	}

	buf := append([]float64(nil), src...)
	if err := plan.ApplyOperator(buf, buf); err != nil {
		t.Fatalf("ApplyOperator (aliased): %v", err)
	}
	if d := maxAbsDiff(buf, want); d > 0 {
		t.Fatalf("aliased ApplyOperator differs from non-aliased: max diff %.3e", d)
	}
}

// solveResidual solves L_a u = f and returns the relative residual of the
// reapplied operator, projecting the mean out for nullspace problems.
func solveResidual(t *testing.T, plan *poisson.VariableCoeffPlan, bcs []poisson.BCType, raw []float64) float64 {
	t.Helper()
	solveRHS, wantRHS := prepareResidualRHS(raw, bcs)

	u := make([]float64, len(solveRHS))
	stats, err := plan.Solve(u, solveRHS)
	if err != nil {
		t.Fatalf("Solve (%d iters, resid %.3e): %v", stats.Iterations, stats.Residual, err)
	}

	back := make([]float64, len(u))
	if err := plan.ApplyOperator(back, u); err != nil {
		t.Fatalf("ApplyOperator: %v", err)
	}
	if allNullspace(bcs) {
		mean := sliceMean(back)
		for i := range back {
			back[i] -= mean
		}
	}
	return relResidualError(back, wantRHS)
}

func TestVariableCoeff_RandomResidual1D(t *testing.T) {
	const n = 16
	seed := int64(1)
	for _, b := range residualBCTypes {
		t.Run(bcName(b), func(t *testing.T) {
			a := positiveCoeff(seed, n)
			plan, err := poisson.NewVariableCoeffPlan(1, []int{n}, []float64{0.1},
				[]poisson.BCType{b}, a, poisson.WithTolerance(1e-11))
			if err != nil {
				t.Fatalf("NewVariableCoeffPlan: %v", err)
			}
			raw := randomField(seed+50, n)
			if e := solveResidual(t, plan, []poisson.BCType{b}, raw); e > 1e-8 {
				t.Fatalf("residual too large: %.3e", e)
			}
		})
		seed++
	}
}

func TestVariableCoeff_RandomResidual2D(t *testing.T) {
	const nx, ny = 12, 14
	size := nx * ny
	seed := int64(100)
	for _, bx := range residualBCTypes {
		for _, by := range residualBCTypes {
			bcs := []poisson.BCType{bx, by}
			t.Run(bcName(bx)+"_"+bcName(by), func(t *testing.T) {
				a := positiveCoeff(seed, size)
				plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{0.1, 0.13},
					bcs, a, poisson.WithTolerance(1e-11))
				if err != nil {
					t.Fatalf("NewVariableCoeffPlan: %v", err)
				}
				raw := randomField(seed+5000, size)
				if e := solveResidual(t, plan, bcs, raw); e > 1e-8 {
					t.Fatalf("residual too large: %.3e", e)
				}
			})
			seed++
		}
	}
}

func TestVariableCoeff_RandomResidual3D(t *testing.T) {
	const nx, ny, nz = 6, 8, 6
	size := nx * ny * nz
	seed := int64(1000)
	triples := [][3]poisson.BCType{
		{poisson.Dirichlet, poisson.Neumann, poisson.Periodic},
		{poisson.Neumann, poisson.Neumann, poisson.Neumann},
		{poisson.DirichletNeumann, poisson.Dirichlet, poisson.NeumannDirichlet},
	}
	for _, tr := range triples {
		bcs := tr[:]
		t.Run(bcName(tr[0])+"_"+bcName(tr[1])+"_"+bcName(tr[2]), func(t *testing.T) {
			a := positiveCoeff(seed, size)
			plan, err := poisson.NewVariableCoeffPlan(3, []int{nx, ny, nz}, []float64{0.1, 0.12, 0.08},
				bcs, a, poisson.WithTolerance(1e-11))
			if err != nil {
				t.Fatalf("NewVariableCoeffPlan: %v", err)
			}
			raw := randomField(seed+7000, size)
			if e := solveResidual(t, plan, bcs, raw); e > 1e-8 {
				t.Fatalf("residual too large: %.3e", e)
			}
		})
		seed++
	}
}

// TestVariableCoeff_ArithmeticAveraging exercises the arithmetic-mean face
// option: it defines a different (but self-consistent) operator, so its own
// residual must still vanish.
func TestVariableCoeff_ArithmeticAveraging(t *testing.T) {
	const nx, ny = 12, 14
	size := nx * ny
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Neumann}
	a := positiveCoeff(9, size)
	plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{0.1, 0.13}, bcs, a,
		poisson.WithArithmeticAveraging(), poisson.WithTolerance(1e-11))
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}
	raw := randomField(99, size)
	if e := solveResidual(t, plan, bcs, raw); e > 1e-8 {
		t.Fatalf("residual too large: %.3e", e)
	}
}

// TestVariableCoeff_OperatorSymmetricAndInvertible2D builds the dense matrix of
// L_a by applying it to unit vectors, asserts it is symmetric (SPD structure,
// which validates CG), and checks that Solve recovers a manufactured discrete
// solution b = L_a·x_true.
func TestVariableCoeff_OperatorSymmetricAndInvertible2D(t *testing.T) {
	const nx, ny = 4, 6
	size := nx * ny
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.DirichletNeumann}
	a := positiveCoeff(7, size)
	plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{0.1, 0.13}, bcs, a,
		poisson.WithTolerance(1e-12))
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}

	// Dense operator: column j = L_a · e_j.
	dense := make([][]float64, size)
	e := make([]float64, size)
	col := make([]float64, size)
	for j := range size {
		for i := range e {
			e[i] = 0
		}
		e[j] = 1
		if err := plan.ApplyOperator(col, e); err != nil {
			t.Fatalf("ApplyOperator: %v", err)
		}
		dense[j] = append([]float64(nil), col...)
	}
	for i := range size {
		for j := range size {
			if d := math.Abs(dense[j][i] - dense[i][j]); d > 1e-12 {
				t.Fatalf("operator not symmetric at (%d,%d): |A_ij-A_ji|=%.3e", i, j, d)
			}
		}
	}

	// Manufactured discrete solution.
	xTrue := randomField(21, size)
	b := make([]float64, size)
	if err := plan.ApplyOperator(b, xTrue); err != nil {
		t.Fatalf("ApplyOperator: %v", err)
	}
	got := make([]float64, size)
	stats, err := plan.Solve(got, b)
	if err != nil {
		t.Fatalf("Solve: %v", err)
	}
	if d := maxAbsDiff(got, xTrue); d > 1e-7 {
		t.Fatalf("Solve did not recover manufactured solution (%d iters): max diff %.3e", stats.Iterations, d)
	}
}

// TestVariableCoeff_ExactPreconditionerOneIteration confirms that a constant
// coefficient makes the spectral preconditioner exact, so PCG converges in a
// single iteration.
func TestVariableCoeff_ExactPreconditionerOneIteration(t *testing.T) {
	const nx, ny = 10, 12
	size := nx * ny
	a := make([]float64, size)
	for i := range a {
		a[i] = 2.0
	}
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}
	plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{0.1, 0.1}, bcs, a)
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}
	rhs := randomField(5, size)
	u := make([]float64, size)
	stats, err := plan.Solve(u, rhs)
	if err != nil {
		t.Fatalf("Solve: %v", err)
	}
	if stats.Iterations != 1 {
		t.Fatalf("exact preconditioner should converge in 1 iteration, got %d", stats.Iterations)
	}
}

// TestVariableCoeff_HighContrastConverges checks the preconditioned solve on a
// high-contrast smooth coefficient converges to tolerance within a small
// iteration budget.
func TestVariableCoeff_HighContrastConverges(t *testing.T) {
	const nx, ny = 32, 32
	size := nx * ny
	a := make([]float64, size)
	for i := range nx {
		x := float64(i) / nx
		for j := range ny {
			y := float64(j) / ny
			// contrast ~ exp(3) ≈ 20
			a[i*ny+j] = math.Exp(1.5 * math.Sin(2*math.Pi*x) * math.Cos(2*math.Pi*y))
		}
	}
	bcs := []poisson.BCType{poisson.Periodic, poisson.Periodic}
	plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{1.0 / nx, 1.0 / ny}, bcs, a)
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}
	rhs := randomField(13, size)
	// Periodic (nullspace): compatible RHS must be mean-zero; Solve projects it.
	u := make([]float64, size)
	stats, err := plan.Solve(u, rhs)
	if err != nil {
		t.Fatalf("Solve (%d iters, resid %.3e): %v", stats.Iterations, stats.Residual, err)
	}
	t.Logf("high-contrast periodic: %d iterations, residual %.3e", stats.Iterations, stats.Residual)
	// 1024 unknowns; the spectral preconditioner makes convergence depend on the
	// coefficient contrast (~20), not the grid, so a small count is expected.
	if stats.Iterations > 60 {
		t.Fatalf("preconditioner ineffective: %d iterations for contrast ~20", stats.Iterations)
	}
}

// TestVariableCoeff_Concurrent runs one shared plan across many goroutines and
// checks each result matches a serial baseline. Run under -race.
func TestVariableCoeff_Concurrent(t *testing.T) {
	const nx, ny = 24, 20
	size := nx * ny
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Neumann}
	a := positiveCoeff(4, size)
	plan, err := poisson.NewVariableCoeffPlan(2, []int{nx, ny}, []float64{0.1, 0.12}, bcs, a)
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}

	const goroutines = 8
	rhs := make([][]float64, goroutines)
	serial := make([][]float64, goroutines)
	for g := range goroutines {
		rhs[g] = randomField(int64(g)+200, size)
		serial[g] = make([]float64, size)
		if _, err := plan.Solve(serial[g], rhs[g]); err != nil {
			t.Fatalf("serial Solve[%d]: %v", g, err)
		}
	}

	var wg sync.WaitGroup
	errs := make([]error, goroutines)
	out := make([][]float64, goroutines)
	for g := range goroutines {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			out[g] = make([]float64, size)
			_, errs[g] = plan.Solve(out[g], rhs[g])
		}(g)
	}
	wg.Wait()

	for g := range goroutines {
		if errs[g] != nil {
			t.Fatalf("concurrent Solve[%d]: %v", g, errs[g])
		}
		if d := maxAbsDiff(out[g], serial[g]); d > 1e-9 {
			t.Fatalf("goroutine %d diverged from serial: max diff %.3e", g, d)
		}
	}
}

func TestVariableCoeff_Validation(t *testing.T) {
	const n = 8
	good := make([]float64, n)
	for i := range good {
		good[i] = 1
	}
	bcs := []poisson.BCType{poisson.Dirichlet}

	t.Run("wrong coeff length", func(t *testing.T) {
		_, err := poisson.NewVariableCoeffPlan(1, []int{n}, []float64{0.1}, bcs, make([]float64, n-1))
		if !errors.Is(err, poisson.ErrInvalidCoefficient) {
			t.Fatalf("got %v, want ErrInvalidCoefficient", err)
		}
	})

	t.Run("non-positive coeff", func(t *testing.T) {
		bad := append([]float64(nil), good...)
		bad[3] = 0
		_, err := poisson.NewVariableCoeffPlan(1, []int{n}, []float64{0.1}, bcs, bad)
		if !errors.Is(err, poisson.ErrInvalidCoefficient) {
			t.Fatalf("got %v, want ErrInvalidCoefficient", err)
		}
	})

	t.Run("NaN coeff", func(t *testing.T) {
		bad := append([]float64(nil), good...)
		bad[1] = math.NaN()
		_, err := poisson.NewVariableCoeffPlan(1, []int{n}, []float64{0.1}, bcs, bad)
		if !errors.Is(err, poisson.ErrInvalidCoefficient) {
			t.Fatalf("got %v, want ErrInvalidCoefficient", err)
		}
	})

	plan, err := poisson.NewVariableCoeffPlan(1, []int{n}, []float64{0.1}, bcs, good)
	if err != nil {
		t.Fatalf("NewVariableCoeffPlan: %v", err)
	}

	t.Run("nil buffer", func(t *testing.T) {
		if _, err := plan.Solve(nil, good); !errors.Is(err, poisson.ErrNilBuffer) {
			t.Fatalf("got %v, want ErrNilBuffer", err)
		}
	})

	t.Run("size mismatch", func(t *testing.T) {
		if _, err := plan.Solve(make([]float64, n-1), good); !errors.Is(err, poisson.ErrSizeMismatch) {
			t.Fatalf("got %v, want ErrSizeMismatch", err)
		}
	})

	t.Run("not converged", func(t *testing.T) {
		// A high-contrast problem with a 1-iteration budget and a tight tolerance
		// cannot converge.
		size := 32 * 32
		a := make([]float64, size)
		for i := range 32 {
			for j := range 32 {
				a[i*32+j] = math.Exp(3 * math.Sin(0.4*float64(i)) * math.Cos(0.4*float64(j)))
			}
		}
		hard, err := poisson.NewVariableCoeffPlan(2, []int{32, 32}, []float64{0.03, 0.03},
			[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, a,
			poisson.WithMaxIterations(1), poisson.WithTolerance(1e-14))
		if err != nil {
			t.Fatalf("NewVariableCoeffPlan: %v", err)
		}
		rhs := randomField(77, size)
		stats, err := hard.Solve(make([]float64, size), rhs)
		if !errors.Is(err, poisson.ErrNotConverged) {
			t.Fatalf("got %v, want ErrNotConverged", err)
		}
		if stats.Iterations != 1 {
			t.Fatalf("expected 1 iteration under the cap, got %d", stats.Iterations)
		}
	})
}
