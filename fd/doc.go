// Package fd provides finite difference operators and eigenvalue computations.
//
// This package implements the mathematical foundations for the spectral Poisson solver:
//   - Eigenvalue formulas for the discrete Laplacian
//   - Laplacian stencil application (for testing and validation)
//
// # Eigenvalues
//
// For a standard second-order finite difference Laplacian on a uniform grid,
// the eigenvalues depend on the boundary condition:
//
// Periodic (m = 0..N-1):
//
//	λ_m = (2 - 2*cos(2πm/N)) / h²
//
// Dirichlet (m = 1..N):
//
//	λ_m = (2 - 2*cos(πm/(N+1))) / h²
//
// Neumann (m = 0..N-1):
//
//	λ_m = (2 - 2*cos(πm/N)) / h²
//
// In multiple dimensions, eigenvalues sum:
//
//	Λ(i,j,k) = λ_x(i) + λ_y(j) + λ_z(k)
//
// # Nullspace
//
// Periodic and Neumann have λ_0 = 0 (the constant mode).
// This must be handled specially in the solver.
package fd
