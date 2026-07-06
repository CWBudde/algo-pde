package poisson_test

import (
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// zeroMasked returns a copy of v with the masked (false) entries set to zero.
func zeroMasked(v []float64, mask []bool) []float64 {
	out := append([]float64(nil), v...)
	for i := range out {
		if !mask[i] {
			out[i] = 0
		}
	}
	return out
}

// diskMask2D marks the cells whose centre lies within a disk inscribed in the
// grid as active; the corners fall outside and are masked (solid). The disk sits
// strictly inside the box, so the active region is fully surrounded by masked
// cells — the immersed Dirichlet pins remove any nullspace for every outer BC.
func diskMask2D(nx, ny int) []bool {
	mask := make([]bool, nx*ny)
	cx, cy := float64(nx-1)/2, float64(ny-1)/2
	r := 0.4 * math.Min(float64(nx), float64(ny))
	for i := range nx {
		for j := range ny {
			dx, dy := float64(i)-cx, float64(j)-cy
			mask[i*ny+j] = dx*dx+dy*dy <= r*r
		}
	}
	return mask
}

// ballMask3D is the 3D analogue of diskMask2D.
func ballMask3D(nx, ny, nz int) []bool {
	mask := make([]bool, nx*ny*nz)
	cx, cy, cz := float64(nx-1)/2, float64(ny-1)/2, float64(nz-1)/2
	r := 0.4 * math.Min(float64(nx), math.Min(float64(ny), float64(nz)))
	for i := range nx {
		for j := range ny {
			for k := range nz {
				dx, dy, dz := float64(i)-cx, float64(j)-cy, float64(k)-cz
				mask[(i*ny+j)*nz+k] = dx*dx+dy*dy+dz*dz <= r*r
			}
		}
	}
	return mask
}

// intervalMask1D marks the middle half of the line active, masking the two ends.
func intervalMask1D(n int) []bool {
	mask := make([]bool, n)
	for i := range mask {
		mask[i] = i >= n/4 && i < 3*n/4
	}
	return mask
}

// countActive reports how many cells are active.
func countActive(mask []bool) int {
	c := 0
	for _, m := range mask {
		if m {
			c++
		}
	}
	return c
}

// nonNullspaceBCTypes are the boundary conditions whose operator has no
// nullspace, so an all-active (unmasked) grid is still solvable — the cases the
// reduction-to-fd check can exercise.
var nonNullspaceBCTypes = []poisson.BCType{
	poisson.Dirichlet,
	poisson.DirichletNeumann,
	poisson.NeumannDirichlet,
}

// TestMasked_ReducesToFDApply pins the operator: with an all-active mask (no
// masking) the masked operator must equal fd.Apply exactly.
func TestMasked_ReducesToFDApply(t *testing.T) {
	t.Run("1D", func(t *testing.T) {
		const n = 24
		const h = 0.1
		for _, b := range nonNullspaceBCTypes {
			t.Run(bcName(b), func(t *testing.T) {
				mask := make([]bool, n)
				for i := range mask {
					mask[i] = true
				}
				plan, err := poisson.NewMaskedPlan(1, []int{n}, []float64{h}, []poisson.BCType{b}, mask)
				if err != nil {
					t.Fatalf("NewMaskedPlan: %v", err)
				}
				src := randomField(1, n)
				got := make([]float64, n)
				if err := plan.ApplyOperator(got, src); err != nil {
					t.Fatalf("ApplyOperator: %v", err)
				}
				want := make([]float64, n)
				if err := fd.Apply1D(want, src, h, b); err != nil {
					t.Fatal(err)
				}
				if d := maxAbsDiff(got, want); d > 1e-12 {
					t.Fatalf("masked operator != fd.Apply1D: max diff %.3e", d)
				}
			})
		}
	})

	t.Run("2D", func(t *testing.T) {
		const nx, ny = 14, 16
		hx, hy := 0.1, 0.13
		mask := make([]bool, nx*ny)
		for i := range mask {
			mask[i] = true
		}
		bcs := []poisson.BCType{poisson.Dirichlet, poisson.NeumannDirichlet}
		plan, err := poisson.NewMaskedPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, mask)
		if err != nil {
			t.Fatalf("NewMaskedPlan: %v", err)
		}
		src := randomField(2, nx*ny)
		got := make([]float64, nx*ny)
		if err := plan.ApplyOperator(got, src); err != nil {
			t.Fatalf("ApplyOperator: %v", err)
		}
		want := make([]float64, nx*ny)
		if err := fd.Apply2D(want, src, grid.NewShape2D(nx, ny), [2]float64{hx, hy},
			[2]poisson.BCType{bcs[0], bcs[1]}); err != nil {
			t.Fatal(err)
		}
		if d := maxAbsDiff(got, want); d > 1e-12 {
			t.Fatalf("masked operator != fd.Apply2D: max diff %.3e", d)
		}
	})

	t.Run("3D", func(t *testing.T) {
		const nx, ny, nz = 8, 10, 6
		hx, hy, hz := 0.1, 0.12, 0.08
		mask := make([]bool, nx*ny*nz)
		for i := range mask {
			mask[i] = true
		}
		bcs := []poisson.BCType{poisson.Dirichlet, poisson.DirichletNeumann, poisson.Dirichlet}
		plan, err := poisson.NewMaskedPlan(3, []int{nx, ny, nz}, []float64{hx, hy, hz}, bcs, mask)
		if err != nil {
			t.Fatalf("NewMaskedPlan: %v", err)
		}
		src := randomField(3, nx*ny*nz)
		got := make([]float64, nx*ny*nz)
		if err := plan.ApplyOperator(got, src); err != nil {
			t.Fatalf("ApplyOperator: %v", err)
		}
		want := make([]float64, nx*ny*nz)
		if err := fd.Apply3D(want, src, grid.NewShape3D(nx, ny, nz), [3]float64{hx, hy, hz},
			[3]poisson.BCType{bcs[0], bcs[1], bcs[2]}); err != nil {
			t.Fatal(err)
		}
		if d := maxAbsDiff(got, want); d > 1e-12 {
			t.Fatalf("masked operator != fd.Apply3D: max diff %.3e", d)
		}
	})
}

// TestMasked_OperatorConsistency checks the masked operator against an
// independent reference on a masked 2D grid: an active row equals fd.Apply of the
// masked-zeroed input (a masked neighbour reads as the u = 0 ghost), and a masked
// row is the identity.
func TestMasked_OperatorConsistency(t *testing.T) {
	const nx, ny = 14, 16
	hx, hy := 0.1, 0.13
	mask := diskMask2D(nx, ny)
	bcs := []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}
	plan, err := poisson.NewMaskedPlan(2, []int{nx, ny}, []float64{hx, hy}, bcs, mask)
	if err != nil {
		t.Fatalf("NewMaskedPlan: %v", err)
	}

	src := randomField(11, nx*ny)
	got := make([]float64, nx*ny)
	if err := plan.ApplyOperator(got, src); err != nil {
		t.Fatalf("ApplyOperator: %v", err)
	}

	// Reference: active rows = fd.Apply(masked-zeroed src); masked rows = src.
	z := zeroMasked(src, mask)
	ref := make([]float64, nx*ny)
	if err := fd.Apply2D(ref, z, grid.NewShape2D(nx, ny), [2]float64{hx, hy},
		[2]poisson.BCType{bcs[0], bcs[1]}); err != nil {
		t.Fatal(err)
	}
	for idx := range ref {
		if !mask[idx] {
			ref[idx] = src[idx]
		}
	}
	if d := maxAbsDiff(got, ref); d > 1e-12 {
		t.Fatalf("masked operator inconsistent with fd reference: max diff %.3e", d)
	}

	// Aliased call (dst == src) must give the same result.
	aliased := append([]float64(nil), src...)
	if err := plan.ApplyOperator(aliased, aliased); err != nil {
		t.Fatalf("aliased ApplyOperator: %v", err)
	}
	if d := maxAbsDiff(aliased, got); d != 0 {
		t.Fatalf("aliased ApplyOperator differs from non-aliased: max diff %.3e", d)
	}
}

// solveMaskedResidual solves −Δu = f on the active cells and returns the relative
// residual of the reapplied operator. On the active cells the operator must
// reproduce the RHS; on the masked cells both sides are zero.
func solveMaskedResidual(t *testing.T, plan *poisson.MaskedPlan, mask []bool, raw []float64) float64 {
	t.Helper()
	u := make([]float64, len(raw))
	stats, err := plan.Solve(u, raw)
	if err != nil {
		t.Fatalf("Solve (%d iters, resid %.3e): %v", stats.Iterations, stats.Residual, err)
	}
	back := make([]float64, len(u))
	if err := plan.ApplyOperator(back, u); err != nil {
		t.Fatalf("ApplyOperator: %v", err)
	}
	return relResidualError(back, zeroMasked(raw, mask))
}

func TestMasked_RandomResidual1D(t *testing.T) {
	const n = 24
	mask := intervalMask1D(n)
	seed := int64(1)
	for _, b := range residualBCTypes {
		t.Run(bcName(b), func(t *testing.T) {
			plan, err := poisson.NewMaskedPlan(1, []int{n}, []float64{0.1},
				[]poisson.BCType{b}, mask, poisson.WithMaskTolerance(1e-11))
			if err != nil {
				t.Fatalf("NewMaskedPlan: %v", err)
			}
			raw := randomField(seed+50, n)
			if e := solveMaskedResidual(t, plan, mask, raw); e > 1e-8 {
				t.Fatalf("residual too large: %.3e", e)
			}
		})
		seed++
	}
}

func TestMasked_RandomResidual2D(t *testing.T) {
	const nx, ny = 14, 16
	mask := diskMask2D(nx, ny)
	seed := int64(100)
	for _, bx := range residualBCTypes {
		for _, by := range residualBCTypes {
			bcs := []poisson.BCType{bx, by}
			t.Run(bcName(bx)+"_"+bcName(by), func(t *testing.T) {
				plan, err := poisson.NewMaskedPlan(2, []int{nx, ny}, []float64{0.1, 0.13},
					bcs, mask, poisson.WithMaskTolerance(1e-11))
				if err != nil {
					t.Fatalf("NewMaskedPlan: %v", err)
				}
				raw := randomField(seed+5000, nx*ny)
				if e := solveMaskedResidual(t, plan, mask, raw); e > 1e-8 {
					t.Fatalf("residual too large: %.3e", e)
				}
			})
			seed++
		}
	}
}

func TestMasked_RandomResidual3D(t *testing.T) {
	const nx, ny, nz = 8, 10, 6
	mask := ballMask3D(nx, ny, nz)
	seed := int64(1000)
	triples := [][3]poisson.BCType{
		{poisson.Dirichlet, poisson.Neumann, poisson.Periodic},
		{poisson.Neumann, poisson.Neumann, poisson.Neumann},
		{poisson.DirichletNeumann, poisson.Dirichlet, poisson.NeumannDirichlet},
	}
	for _, tr := range triples {
		bcs := tr[:]
		t.Run(bcName(tr[0])+"_"+bcName(tr[1])+"_"+bcName(tr[2]), func(t *testing.T) {
			plan, err := poisson.NewMaskedPlan(3, []int{nx, ny, nz}, []float64{0.1, 0.12, 0.08},
				bcs, mask, poisson.WithMaskTolerance(1e-11))
			if err != nil {
				t.Fatalf("NewMaskedPlan: %v", err)
			}
			raw := randomField(seed+7000, nx*ny*nz)
			if e := solveMaskedResidual(t, plan, mask, raw); e > 1e-8 {
				t.Fatalf("residual too large: %.3e", e)
			}
		})
		seed++
	}
}

// TestMasked_OperatorSymmetricAndInvertible2D builds the dense matrix of the
// masked operator by applying it to unit vectors, asserts it is symmetric (SPD
// structure, which validates CG), and checks Solve recovers a manufactured
// discrete solution b = L·x_true. The outer BCs are all-nullspace (Neumann ×
// Periodic); the masked Dirichlet pins are what make the operator nonsingular.
func TestMasked_OperatorSymmetricAndInvertible2D(t *testing.T) {
	const nx, ny = 6, 6
	size := nx * ny
	mask := diskMask2D(nx, ny)
	bcs := []poisson.BCType{poisson.Neumann, poisson.Periodic}
	plan, err := poisson.NewMaskedPlan(2, []int{nx, ny}, []float64{0.1, 0.13}, bcs, mask,
		poisson.WithMaskTolerance(1e-12))
	if err != nil {
		t.Fatalf("NewMaskedPlan: %v", err)
	}

	// Dense operator: column j = L · e_j.
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

	// Manufactured discrete solution: zero on the masked cells (the solution's
	// support), so b = L·x_true is in the operator's range.
	xTrue := zeroMasked(randomField(21, size), mask)
	b := make([]float64, size)
	if err := plan.ApplyOperator(b, xTrue); err != nil {
		t.Fatalf("ApplyOperator: %v", err)
	}
	got := make([]float64, size)
	stats, err := plan.Solve(got, b)
	if err != nil {
		t.Fatalf("Solve (%d iters): %v", stats.Iterations, err)
	}
	if d := maxAbsDiff(got, xTrue); d > 1e-7 {
		t.Fatalf("Solve did not recover the manufactured solution: max diff %.3e", d)
	}
}

// TestMasked_ConvergenceGridIndependent solves the same disk-in-square problem
// at two resolutions; the spectral preconditioner keeps the PCG iteration count
// bounded and essentially grid-independent.
func TestMasked_ConvergenceGridIndependent(t *testing.T) {
	const iterCap = 80
	solveDisk := func(n int) int {
		t.Helper()
		mask := diskMask2D(n, n)
		h := 1.0 / float64(n+1)
		plan, err := poisson.NewMaskedPlan(2, []int{n, n}, []float64{h, h},
			[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, mask,
			poisson.WithMaskTolerance(1e-8))
		if err != nil {
			t.Fatalf("NewMaskedPlan(%d): %v", n, err)
		}
		rhs := randomField(int64(n), n*n)
		u := make([]float64, n*n)
		stats, err := plan.Solve(u, rhs)
		if err != nil {
			t.Fatalf("Solve(%d) (%d iters, resid %.3e): %v", n, stats.Iterations, stats.Residual, err)
		}
		return stats.Iterations
	}

	iters32 := solveDisk(32)
	iters64 := solveDisk(64)
	t.Logf("disk PCG iterations: 32x32=%d, 64x64=%d", iters32, iters64)
	if iters32 > iterCap || iters64 > iterCap {
		t.Fatalf("iteration count not grid-independent: 32x32=%d, 64x64=%d (cap %d)",
			iters32, iters64, iterCap)
	}
}

// TestMasked_Concurrent pins the concurrency contract: one shared plan hammered
// from many goroutines must match a serial baseline. Run under -race.
func TestMasked_Concurrent(t *testing.T) {
	const nx, ny = 24, 20
	mask := diskMask2D(nx, ny)
	plan, err := poisson.NewMaskedPlan(2, []int{nx, ny},
		[]float64{1.0 / float64(nx+1), 1.0 / float64(ny+1)},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, mask,
		poisson.WithMaskParallelism(2))
	if err != nil {
		t.Fatalf("NewMaskedPlan: %v", err)
	}
	runConcurrentSolves(t, nx*ny, func(dst, rhs []float64) error {
		_, err := plan.Solve(dst, rhs)
		return err
	})
}

func TestMasked_Validation(t *testing.T) {
	const nx, ny = 8, 8
	size := nx * ny
	h := []float64{0.1, 0.1}
	n := []int{nx, ny}
	dirichlet := []poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}
	neumann := []poisson.BCType{poisson.Neumann, poisson.Neumann}

	t.Run("wrong mask length", func(t *testing.T) {
		mask := make([]bool, size-1)
		if _, err := poisson.NewMaskedPlan(2, n, h, dirichlet, mask); !errors.Is(err, poisson.ErrInvalidMask) {
			t.Fatalf("got %v, want ErrInvalidMask", err)
		}
	})

	t.Run("no active cells", func(t *testing.T) {
		mask := make([]bool, size) // all false
		if _, err := poisson.NewMaskedPlan(2, n, h, dirichlet, mask); !errors.Is(err, poisson.ErrInvalidMask) {
			t.Fatalf("got %v, want ErrInvalidMask", err)
		}
	})

	t.Run("all-active under nullspace outer", func(t *testing.T) {
		mask := make([]bool, size)
		for i := range mask {
			mask[i] = true
		}
		if _, err := poisson.NewMaskedPlan(2, n, h, neumann, mask); !errors.Is(err, poisson.ErrInvalidMask) {
			t.Fatalf("got %v, want ErrInvalidMask", err)
		}
	})

	t.Run("all-active under Dirichlet outer is valid", func(t *testing.T) {
		mask := make([]bool, size)
		for i := range mask {
			mask[i] = true
		}
		if _, err := poisson.NewMaskedPlan(2, n, h, dirichlet, mask); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	mask := diskMask2D(nx, ny)
	plan, err := poisson.NewMaskedPlan(2, n, h, dirichlet, mask)
	if err != nil {
		t.Fatalf("NewMaskedPlan: %v", err)
	}

	t.Run("nil buffer", func(t *testing.T) {
		if _, err := plan.Solve(nil, make([]float64, size)); !errors.Is(err, poisson.ErrNilBuffer) {
			t.Fatalf("got %v, want ErrNilBuffer", err)
		}
	})

	t.Run("size mismatch", func(t *testing.T) {
		if _, err := plan.Solve(make([]float64, size), make([]float64, size-1)); !errors.Is(err, poisson.ErrSizeMismatch) {
			t.Fatalf("got %v, want ErrSizeMismatch", err)
		}
	})

	t.Run("not converged", func(t *testing.T) {
		hard, err := poisson.NewMaskedPlan(2, n, h, dirichlet, mask,
			poisson.WithMaskMaxIterations(1), poisson.WithMaskTolerance(1e-14))
		if err != nil {
			t.Fatalf("NewMaskedPlan: %v", err)
		}
		u := make([]float64, size)
		stats, err := hard.Solve(u, randomField(7, size))
		if !errors.Is(err, poisson.ErrNotConverged) {
			t.Fatalf("got %v, want ErrNotConverged", err)
		}
		if stats.Iterations != 1 {
			t.Fatalf("expected 1 iteration, got %d", stats.Iterations)
		}
	})
}

func TestMasked_Mask(t *testing.T) {
	const nx, ny = 8, 8
	mask := diskMask2D(nx, ny)
	plan, err := poisson.NewMaskedPlan(2, []int{nx, ny}, []float64{0.1, 0.1},
		[]poisson.BCType{poisson.Dirichlet, poisson.Dirichlet}, mask)
	if err != nil {
		t.Fatalf("NewMaskedPlan: %v", err)
	}
	got := plan.Mask()
	if len(got) != len(mask) {
		t.Fatalf("Mask length %d, want %d", len(got), len(mask))
	}
	for i := range mask {
		if got[i] != mask[i] {
			t.Fatalf("Mask mismatch at %d", i)
		}
	}
	// Mutating the returned slice must not affect the plan.
	got[0] = !got[0]
	if plan.Mask()[0] == got[0] {
		t.Fatal("Mask() returned a live reference, not a copy")
	}
	if countActive(mask) == 0 {
		t.Fatal("test mask has no active cells")
	}
}
