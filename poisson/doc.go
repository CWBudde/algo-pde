// Package poisson provides fast spectral solvers for the Poisson equation.
//
// The Poisson equation is:
//
//	-Δu = f
//
// where Δ is the Laplacian operator and f is the source term.
// The solver also supports Helmholtz / screened Poisson forms:
//
//	(α - Δ)u = f
//
// For diffusion steps u - νΔu = f, divide by ν to set α = 1/ν and RHS = f/ν.
//
// # Boundary Conditions
//
// The solver supports three types of boundary conditions:
//
//   - Periodic: u(0) = u(L), useful for problems with periodic symmetry
//   - Dirichlet: u = 0 at boundaries, models fixed-value boundaries
//   - Neumann: ∂u/∂n = 0 at boundaries, models no-flux boundaries
//
// Mixed boundary conditions (different BC per axis) are also supported.
//
// # Plan-Based API
//
// The solver uses a plan-based API for efficiency:
//
//  1. Create a plan once with NewPlan2DPeriodic or NewPlan
//  2. The plan pre-computes eigenvalues and allocates buffers
//  3. Call Solve() repeatedly for different right-hand sides
//
// Example:
//
//	plan, err := NewPlan2DPeriodic(128, 128, 1.0/128, 1.0/128)
//	if err != nil {
//	    return err
//	}
//
//	rhs := make([]float64, 128*128)
//	sol := make([]float64, 128*128)
//	// ... fill rhs ...
//
//	if err := plan.Solve(sol, rhs); err != nil {
//	    return err
//	}
//
// For inhomogeneous Dirichlet/Neumann data, use SolveWithBC and provide
// boundary values per face. The solver applies the boundary contributions
// before solving.
//
// # Nullspace Handling
//
// Periodic and Neumann boundary conditions have a nullspace (constant mode).
// The solver handles this by:
//
//   - NullspaceZeroMode: Set zero-mode to zero (default)
//   - NullspaceSubtractMean: Automatically subtract mean from RHS
//   - NullspaceError: Return error if nullspace exists
//
// # Performance
//
// The solver has O(N log N) complexity where N is the total number of grid points.
// Plans should be reused for multiple solves to avoid repeated setup costs.
// The Solve method is designed for zero allocations when using pre-made plans.
package poisson
