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
// # Grid Conventions
//
// The node placement, domain length, and sample points differ per boundary
// condition. Sampling the source term f (or boundary data g) at the wrong
// points converges smoothly to the wrong answer, so this matters. For an axis
// with n unknowns and spacing h:
//
//	BC          Node placement    Node i sits at   Domain length   Transform
//	Periodic    node-centered     i*h              n*h             FFT
//	Dirichlet   vertex-centered   (i+1)*h          (n+1)*h         DST-I
//	Neumann     cell-centered     (i+1/2)*h        n*h             DCT-II
//
// Periodic (n=4): unknowns wrap, u(0) = u(L), L = n*h. Nodes coincide with the
// left boundary; the right boundary is the periodic image of node 0.
//
//	u_0    u_1    u_2    u_3   (= u_0)
//	 o------o------o------o------o
//	 0      h      2h     3h     4h = L
//
// Dirichlet (n=4): the boundaries are vertices where u is prescribed (g at x=0
// and x=L). The n unknowns are the interior vertices; the domain spans
// (n+1)*h. The DST-I basis sin(pi*(i+1)*m/(n+1)) vanishes at both boundaries.
//
//	u=g_low                                u=g_high
//	  x------o------o------o------o------x
//	  0      h      2h     3h     4h     5h = (n+1)h = L
//	         u_0    u_1    u_2    u_3
//
// Neumann (n=4): the unknowns are cell centers; the boundaries fall halfway
// between the outermost node and its ghost, at x=0 and x=L=n*h. The DCT-II
// basis cos(pi*(i+1/2)*m/n) has zero slope at both boundaries.
//
//	dU/dx=g_low                        dU/dx=g_high
//	  |   u_0    u_1    u_2    u_3   |
//	  |    o      o      o      o    |
//	  0   h/2    3h/2   5h/2   7h/2  4h = n*h = L
//
// For a mixed problem each axis follows its own row of the table, so a
// Dirichlet-x / Neumann-y grid samples f at ((i+1)*hx, (j+1/2)*hy).
//
// # Neumann Sign Convention
//
// Inhomogeneous Neumann data (ApplyNeumannRHS and the g values passed to
// SolveWithBC) is the derivative along the POSITIVE axis direction, dU/dx_axis,
// at each face — not the outward normal derivative. At a high face the outward
// normal points along +axis, so the two agree; at a low face the outward
// normal points along -axis, so the outward-normal derivative is the negation
// of the value you pass. Pass +dU/dx at both the low and the high face.
//
// # Eigenvalues
//
// The solver diagonalizes the second-order negative Laplacian stencil
// (-u_{i-1} + 2*u_i - u_{i+1})/h^2 in the transform basis. The per-axis
// eigenvalues (all >= 0) are:
//
//	Periodic:    (2 - 2*cos(2*pi*k/n)) / h^2,       k = 0 .. n-1
//	Dirichlet:   (2 - 2*cos(pi*m/(n+1))) / h^2,     m = 1 .. n
//	Neumann:     (2 - 2*cos(pi*m/n)) / h^2,         m = 0 .. n-1
//
// The full d-dimensional eigenvalue for a mode is the sum of the per-axis
// eigenvalues, and Solve divides each transformed coefficient by (alpha + that
// sum). Periodic (k=0) and Neumann (m=0) contribute a zero eigenvalue — the
// constant nullspace mode — handled per the Nullspace Handling section below.
//
// # Memory Layout
//
// All fields (rhs, solution, per-face boundary values) are flat float64 slices
// in row-major (C) order: for shape [nx, ny, nz] the element at (i, j, k) is at
// index i*(ny*nz) + j*nz + k. The X axis (axis 0) is the slowest-varying and Z
// (axis 2) the fastest. 1D uses [n, 1, 1] and 2D uses [nx, ny, 1]. Per-face
// boundary slices drop the face's own axis and keep the remaining axes in the
// same order: an X face is ny*nz values indexed j*nz + k.
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
