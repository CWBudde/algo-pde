// Package pde provides fast spectral solvers for partial differential equations.
//
// This package builds on algo-fft to provide efficient O(N log N) solvers for:
//   - Poisson equation: -Δu = f
//   - Helmholtz equation: (α - Δ)u = f
//
// The solvers support various boundary conditions:
//   - Periodic: u(0) = u(L), u'(0) = u'(L)
//   - Dirichlet: u = g on boundary
//   - Neumann: ∂u/∂n = g on boundary
//   - Mixed: different conditions per axis
//
// # Architecture
//
// The library uses a plan-based API similar to FFTW and algo-fft:
//
//  1. Create a plan with grid size, spacing, and boundary conditions
//  2. The plan pre-computes eigenvalues and allocates work buffers
//  3. Call Solve() repeatedly with different right-hand sides
//
// Plans are safe for concurrent Solve() calls and should be reused
// for multiple solves on the same grid.
//
// # Packages
//
//   - poisson: Poisson and Helmholtz solvers
//   - r2r: Real-to-real transforms (DST/DCT) via FFT
//   - grid: Grid shapes, strides, and indexing utilities
//   - fd: Finite difference operators and eigenvalues
//
// # Example
//
//	// Create a 2D periodic Poisson solver
//	plan, err := poisson.NewPlan2DPeriodic(
//	    128, 128,         // grid size
//	    1.0/128, 1.0/128, // grid spacing
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Solve -Δu = f
//	rhs := make([]float64, 128*128)
//	sol := make([]float64, 128*128)
//	// ... fill rhs with source term ...
//	plan.Solve(sol, rhs)
package pde
