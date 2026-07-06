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
// The solver supports these boundary conditions:
//
//   - Periodic: u(0) = u(L), useful for problems with periodic symmetry
//   - Dirichlet: u = 0 at boundaries, models fixed-value boundaries
//   - Neumann: ∂u/∂n = 0 at boundaries, models no-flux boundaries
//   - DirichletNeumann / NeumannDirichlet: a single axis with a Dirichlet
//     condition on one face and a Neumann condition on the other (per-face
//     asymmetric)
//
// Mixed boundary conditions (different BC per axis) are also supported: pass a
// per-axis BCType slice to NewPlan.
//
// True Robin conditions (a·u + b·∂u/∂n = g with a,b ≠ 0) are intentionally not
// supported: their eigenvectors are phase-shifted sinusoids whose wavenumbers
// solve a transcendental equation, so they have neither a closed-form
// eigenvalue nor a DFT-based fast transform — they do not fit this solver's
// O(N log N) design. The DirichletNeumann / NeumannDirichlet axes are the
// per-face-asymmetric cases that *do* admit a fast quarter-wave transform.
//
// # Grid Conventions
//
// Each axis is discretized independently, and the boundary condition on that
// axis fixes three things: which transform is used, where the n sample points
// sit relative to the physical domain, and how long the domain is. Sample the
// source term f at exactly these node coordinates — sampling at the wrong
// points converges smoothly to the wrong answer with no error reported.
// (Inhomogeneous boundary data passed to SolveWithBC is sampled on the
// boundary face instead; see below.)
//
//	BC                 Transform  Node x_i (i = 0..n-1)  Domain length L
//	Periodic           FFT        i·h                    n·h
//	Dirichlet          DST-I      (i+1)·h                (n+1)·h
//	Neumann            DCT-II     (i+½)·h                n·h
//	DirichletNeumann   DST-IV     (i+½)·h                n·h
//	NeumannDirichlet   DCT-IV     (i+½)·h                n·h
//
// The physical boundaries lie off the grid for Dirichlet and Neumann axes:
//
//	Periodic (x_0 at 0; the right boundary x=L wraps back to x_0):
//	    x=0                                      x=L=n·h
//	     •---------•---------•-- ... --•---------|
//	    x_0       x_1       x_2       x_{n-1}   (≡ x_0)
//
//	Dirichlet (vertex-centered; boundaries at x=0 and x=(n+1)h are OFF-grid):
//	    x=0        h        2h            (n+1)h = L
//	     |---------•---------•-- ... --•---------|
//	   u fixed    x_0       x_1       x_{n-1}   u fixed
//
//	Neumann (cell-centered; boundaries at x=0 and x=n·h are half a cell OFF-grid):
//	    x=0   ½h       3/2h              (n-½)h   L=n·h
//	     |----•---------•----- ... ------•----|
//	         x_0       x_1             x_{n-1}
//
// DirichletNeumann and NeumannDirichlet share the Neumann cell-centered grid
// (nodes at (i+½)·h, L = n·h) but pin different faces. On a DirichletNeumann
// axis the low face (x=0) is Dirichlet and the high face (x=n·h) is Neumann;
// NeumannDirichlet swaps them. Both use quarter-wave transforms (DST-IV / DCT-IV
// respectively) and have strictly positive eigenvalues
// λ_m = (2 − 2·cos(π(m+½)/n))/h², so a mixed axis contributes no nullspace:
//
//	DirichletNeumann (Dirichlet low, Neumann high):
//	    x=0   ½h                          (n-½)h   L=n·h
//	     |----•---------•----- ... ------•----|
//	  u fixed x_0       x_1            x_{n-1}  ∂u/∂x=g
//
// Because each axis uses its own rule, a mixed-BC plan is a rectangle whose
// side lengths follow different formulas per axis. To make each axis span the
// unit interval, for example, choose hx = 1/(nx+1) on a Dirichlet axis but
// hy = 1/ny on a Neumann, periodic, or mixed quarter-wave axis. See the
// examples/ programs (dirichlet2d, neumann2d, periodic2d, mixed2d, mixed_dn)
// for worked samplings.
//
// Inhomogeneous boundary data supplied to SolveWithBC lives on the physical
// boundary face, not at an interior node. A face's Values array is indexed by
// the grid nodes of the tangential axes, but the normal coordinate is fixed at
// the face itself. For a Dirichlet XLow face the value is u at x=0 (not at the
// first interior node x=h); for XHigh it is u at x=(n+1)h. For a Neumann face
// it is the derivative evaluated at that face (x=0 or x=n·h). See
// examples/inhomogeneous_bc.
//
// Neumann sign convention: the boundary value g supplied to SolveWithBC is the
// derivative along the positive axis direction, ∂u/∂x_axis, at that face — not
// the outward-normal derivative. At a low face the outward normal points in the
// −axis direction, so there g = −∂u/∂n; at a high face g = +∂u/∂n.
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
// # Complex Helmholtz
//
// NewComplexHelmholtzPlan solves (α - Δ)u = f for a complex coefficient α and a
// real right-hand side f; the solution u is complex and is returned by
// SolveComplex(dst []complex128, rhs []float64). It reuses the same transform
// pipeline: because every axis transform is linear, the real and imaginary
// parts of each spectral mode are carried independently, so a single complex
// per-mode division yields the complex solution for every boundary condition.
//
// A positive imaginary part of α is a complex shift (damping): it keeps the
// operator non-singular near resonances, so SolveComplex returns a finite field
// instead of ErrResonant. With α = -k² + i·ε this models a driven, lightly
// damped acoustic room at wavenumber k (standing-wave room modes). A purely
// real α behaves like NewHelmholtzPlan and still guards genuine resonances.
//
// # Variable Coefficients
//
// NewVariableCoeffPlan solves the variable-coefficient elliptic equation
//
//	−∇·(a(x)∇u) = f
//
// for a fixed, strictly positive coefficient field a(x). A spatially varying
// coefficient makes the operator non-separable, so the spectral transforms can
// no longer diagonalize it. Instead the discrete operator L_a is inverted by
// preconditioned conjugate gradient (PCG), using the fast spectral solve of the
// nearby constant-coefficient operator −c̄·Δ (c̄ = mean(a) by default) as the
// preconditioner. Because that preconditioner removes the grid-dependence of the
// conditioning, the iteration count depends on the coefficient contrast
// max(a)/min(a), not on N — a handful of iterations for modest contrast.
//
// L_a is the standard second-order flux discretization: along one axis
//
//	(L_a u)_i = [ a_{i+½}(u_i − u_{i+1}) + a_{i−½}(u_i − u_{i−1}) ] / h²,
//
// where a_{i+½} is the coefficient averaged onto the cell face (harmonic by
// default, arithmetic via WithArithmeticAveraging). It reuses the same per-face
// ghost rule as fd.Apply for every boundary condition — so with a ≡ 1 it is
// exactly the constant-coefficient stencil — and is symmetric positive
// (semi)definite, which is what makes CG applicable. Sample a(x) at the same
// node coordinates as f (see Grid Conventions). Solve returns SolveStats
// (iteration count and final relative residual) and ErrNotConverged if the
// tolerance is not reached within the iteration limit; ApplyOperator exposes L_a
// for residual checks. All five boundary conditions are supported and mixable
// per axis; true Robin conditions remain out of scope.
//
// # Masked Domains (Immersed Boundary)
//
// NewMaskedPlan solves the Poisson equation on a non-rectangular domain (a disk,
// an L-shape, a room with an obstacle) embedded in the enclosing rectangular
// grid:
//
//	−Δu = f   inside the domain,   u = 0   on the exterior.
//
// A boolean mask sampled at the grid nodes selects the geometry: a true entry
// marks a cell inside the physical domain (active, where −Δu = f is solved), a
// false entry a cell outside it (masked/solid, where u is pinned to zero). On
// the active cells the standard negative-Laplacian stencil is applied with the
// bounding-box boundary conditions bcs at the rectangle's faces; a masked
// neighbour is read as a homogeneous Dirichlet (u = 0) ghost, so the immersed
// boundary is u = 0. The operator is thus block-diagonal — the principal
// submatrix L_AA of the bounding-box negative Laplacian on the active cells,
// plus the identity on the masked cells — and is symmetric positive definite
// whenever at least one cell is masked, so it is inverted by preconditioned
// conjugate gradient (PCG). The preconditioner is one fast O(N log N) spectral
// solve of the full rectangle followed by zeroing the masked entries, which
// keeps the iteration count essentially grid-independent.
//
// Solve returns SolveStats (iteration count and final relative residual, both
// measured over the active cells) and ErrNotConverged if the tolerance is not
// reached within the iteration limit; the masked cells of the solution are zero.
// ApplyOperator exposes the masked operator for residual checks. Sample f at the
// same node coordinates as the underlying spectral plan (see Grid Conventions).
//
// The immersed boundary is approximated by the staircase of cell edges, so
// accuracy at the boundary is first order in h even though the interior stencil
// is second order. Only a homogeneous Dirichlet immersed boundary (u = 0 on the
// solid) is supported; inhomogeneous immersed Dirichlet data (u = g) and Neumann
// / no-flux immersed boundaries (∂u/∂n = 0, sound-hard walls) remain out of
// scope.
//
// # Performance
//
// The solver has O(N log N) complexity where N is the total number of grid points.
// Plans should be reused for multiple solves to avoid repeated setup costs.
// The Solve method is designed for zero allocations when using pre-made plans.
//
// # Eigenvalues and Memory Layout
//
// Each axis diagonalizes the discrete second-order Laplacian in its own basis,
// with eigenvalues (see also the fd package):
//
//	Periodic          (m = 0..n-1):  λ_m = (2 - 2·cos(2πm/n))      / h²
//	Dirichlet         (m = 1..n):    λ_m = (2 - 2·cos(πm/(n+1)))   / h²
//	Neumann           (m = 0..n-1):  λ_m = (2 - 2·cos(πm/n))       / h²
//	DirichletNeumann  (m = 0..n-1):  λ_m = (2 - 2·cos(π(m+½)/n))   / h²
//	NeumannDirichlet  (m = 0..n-1):  λ_m = (2 - 2·cos(π(m+½)/n))   / h²
//
// Periodic and Neumann have λ = 0 at m = 0 (the constant mode); Dirichlet and
// the mixed Dirichlet/Neumann axes do not (a Dirichlet face pins the constant). In multiple dimensions the per-axis eigenvalues add, so the operator for
// mode (i, j, k) of the Helmholtz form (α - Δ) is:
//
//	α + λ_x(i) + λ_y(j) + λ_z(k)
//
// Solve transforms the RHS along every axis, divides each spectral coefficient
// by that value, then inverse-transforms. Buffers are row-major with axis 0
// slowest and the last axis fastest, so the spectral coefficient for mode
// (i, j, k) lives at index i·(ny·nz) + j·nz + k.
package poisson
