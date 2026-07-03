package poisson_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/MeKo-Tech/algo-pde/fd"
	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/poisson"
)

// The residual tests below solve a *random* RHS and assert that reapplying the
// finite-difference operator to the solution reproduces that RHS. Unlike the
// manufactured-solution tests, a random RHS excites the entire spectrum, so a
// single corrupted eigenvalue at any index (not just the low modes probed by
// sine/cosine manufactured fields) makes the residual blow up. This is the
// highest-leverage correctness check in the suite.
//
// For a nullspace-bearing problem (every axis Neumann/Periodic, alpha == 0) the
// operator annihilates the constant mode, so only a mean-zero RHS is in its
// range and the solution is defined only up to an additive constant. We remove
// the RHS mean up front (and pass WithSubtractMean so the solver tolerates any
// residual roundoff mean); reapplying the operator then reproduces the
// mean-zero RHS regardless of the arbitrary solution constant.

const randomResidualRelTol = 1e-9

// randomField fills a slice of length size with deterministic pseudo-random
// values in [-1, 1) drawn from the given seed.
func randomField(seed int64, size int) []float64 {
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test data
	out := make([]float64, size)
	for i := range out {
		out[i] = 2*rng.Float64() - 1
	}
	return out
}

// allNullspace reports whether every axis carries a nullspace-bearing boundary
// condition, i.e. the alpha == 0 operator is singular on the constant mode.
func allNullspace(bcs []poisson.BCType) bool {
	for _, bc := range bcs {
		if !bc.HasNullspace() {
			return false
		}
	}
	return true
}

// prepareResidualRHS returns the RHS to solve and the RHS the reapplied
// operator must reproduce. They differ only for nullspace problems, where the
// mean is projected out.
func prepareResidualRHS(raw []float64, bcs []poisson.BCType) (solveRHS, wantRHS []float64) {
	solveRHS = make([]float64, len(raw))
	copy(solveRHS, raw)

	if allNullspace(bcs) {
		mean := sliceMean(solveRHS)
		for i := range solveRHS {
			solveRHS[i] -= mean
		}
	}

	wantRHS = make([]float64, len(solveRHS))
	copy(wantRHS, solveRHS)
	return solveRHS, wantRHS
}

// relResidualError returns max|residual - want| divided by the RHS magnitude.
func relResidualError(residual, want []float64) float64 {
	scale := 0.0
	for _, v := range want {
		if abs := math.Abs(v); abs > scale {
			scale = abs
		}
	}
	if scale == 0 {
		scale = 1
	}
	return maxAbsDiff(residual, want) / scale
}

// bc1DName / bc2DName / bc3DName give readable subtest labels.
func bcName(bc poisson.BCType) string {
	switch bc {
	case poisson.Periodic:
		return "Periodic"
	case poisson.Dirichlet:
		return "Dirichlet"
	case poisson.Neumann:
		return "Neumann"
	default:
		return "Unknown"
	}
}

var residualBCTypes = []poisson.BCType{poisson.Periodic, poisson.Dirichlet, poisson.Neumann}

func residualOpts(bcs []poisson.BCType) []poisson.Option {
	if allNullspace(bcs) {
		return []poisson.Option{poisson.WithSubtractMean()}
	}
	return nil
}

func TestRandomRHSResidual1D(t *testing.T) {
	const n = 33
	const h = 1.0 / float64(n)

	seed := int64(1)
	for _, bc := range residualBCTypes {
		t.Run(bcName(bc), func(t *testing.T) {
			bcs := []poisson.BCType{bc}
			raw := randomField(seed, n)
			seed++

			solveRHS, wantRHS := prepareResidualRHS(raw, bcs)

			plan, err := poisson.NewPlan(1, []int{n}, []float64{h}, bcs, residualOpts(bcs)...)
			if err != nil {
				t.Fatalf("NewPlan failed: %v", err)
			}

			sol := make([]float64, n)
			if err := plan.Solve(sol, solveRHS); err != nil {
				t.Fatalf("Solve failed: %v", err)
			}

			residual := make([]float64, n)
			fd.Apply1D(residual, sol, h, bc)

			if e := relResidualError(residual, wantRHS); e > randomResidualRelTol {
				t.Fatalf("relative residual error %g exceeds tol %g", e, randomResidualRelTol)
			}
		})
	}
}

func TestRandomRHSResidual2D(t *testing.T) {
	// Distinct extents and spacings per axis catch axis-swapped indexing and
	// eigenvalue mix-ups that equal-sized grids would hide.
	const nx, ny = 20, 24
	hx, hy := 1.0/float64(nx), 0.75/float64(ny)

	seed := int64(100)
	for _, bcx := range residualBCTypes {
		for _, bcy := range residualBCTypes {
			t.Run(bcName(bcx)+"_"+bcName(bcy), func(t *testing.T) {
				bcs := []poisson.BCType{bcx, bcy}
				raw := randomField(seed, nx*ny)
				seed++

				solveRHS, wantRHS := prepareResidualRHS(raw, bcs)

				plan, err := poisson.NewPlan(
					2,
					[]int{nx, ny},
					[]float64{hx, hy},
					bcs,
					residualOpts(bcs)...,
				)
				if err != nil {
					t.Fatalf("NewPlan failed: %v", err)
				}

				sol := make([]float64, nx*ny)
				if err := plan.Solve(sol, solveRHS); err != nil {
					t.Fatalf("Solve failed: %v", err)
				}

				residual := make([]float64, nx*ny)
				fd.Apply2D(
					residual,
					sol,
					grid.NewShape2D(nx, ny),
					[2]float64{hx, hy},
					[2]poisson.BCType{bcx, bcy},
				)

				if e := relResidualError(residual, wantRHS); e > randomResidualRelTol {
					t.Fatalf("relative residual error %g exceeds tol %g", e, randomResidualRelTol)
				}
			})
		}
	}
}

func TestRandomRHSResidual3D(t *testing.T) {
	const nx, ny, nz = 12, 16, 10
	hx, hy, hz := 1.0/float64(nx), 0.6/float64(ny), 1.3/float64(nz)

	seed := int64(1000)
	for _, bcx := range residualBCTypes {
		for _, bcy := range residualBCTypes {
			for _, bcz := range residualBCTypes {
				t.Run(bcName(bcx)+"_"+bcName(bcy)+"_"+bcName(bcz), func(t *testing.T) {
					bcs := []poisson.BCType{bcx, bcy, bcz}
					raw := randomField(seed, nx*ny*nz)
					seed++

					solveRHS, wantRHS := prepareResidualRHS(raw, bcs)

					plan, err := poisson.NewPlan(
						3,
						[]int{nx, ny, nz},
						[]float64{hx, hy, hz},
						bcs,
						residualOpts(bcs)...,
					)
					if err != nil {
						t.Fatalf("NewPlan failed: %v", err)
					}

					sol := make([]float64, nx*ny*nz)
					if err := plan.Solve(sol, solveRHS); err != nil {
						t.Fatalf("Solve failed: %v", err)
					}

					residual := make([]float64, nx*ny*nz)
					fd.Apply3D(
						residual,
						sol,
						grid.NewShape3D(nx, ny, nz),
						[3]float64{hx, hy, hz},
						[3]poisson.BCType{bcx, bcy, bcz},
					)

					if e := relResidualError(residual, wantRHS); e > randomResidualRelTol {
						t.Fatalf("relative residual error %g exceeds tol %g", e, randomResidualRelTol)
					}
				})
			}
		}
	}
}
