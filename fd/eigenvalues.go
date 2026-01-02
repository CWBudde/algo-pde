package fd

import (
	"math"

	"github.com/MeKo-Tech/algo-pde/poisson"
)

// Eigenvalues computes the 1D eigenvalues of the discrete Laplacian
// for the given boundary condition type.
//
// Parameters:
//   - n: number of grid points
//   - h: grid spacing
//   - bc: boundary condition type
//
// Returns a slice of eigenvalues. The length depends on the BC:
//   - Periodic: n eigenvalues (m = 0..n-1)
//   - Dirichlet: n eigenvalues (m = 1..n, but stored 0..n-1)
//   - Neumann: n eigenvalues (m = 0..n-1)
func Eigenvalues(n int, h float64, bc poisson.BCType) []float64 {
	eig := make([]float64, n)
	h2 := h * h

	switch bc {
	case poisson.Periodic:
		// λ_m = (2 - 2*cos(2πm/N)) / h²
		for m := range n {
			eig[m] = (2.0 - 2.0*math.Cos(2.0*math.Pi*float64(m)/float64(n))) / h2
		}

	case poisson.Dirichlet:
		// λ_m = (2 - 2*cos(πm/(N+1))) / h² for m = 1..N
		// Stored at index m-1
		for m := 1; m <= n; m++ {
			eig[m-1] = (2.0 - 2.0*math.Cos(math.Pi*float64(m)/float64(n+1))) / h2
		}

	case poisson.Neumann:
		// λ_m = (2 - 2*cos(πm/N)) / h² for m = 0..N-1
		for m := range n {
			eig[m] = (2.0 - 2.0*math.Cos(math.Pi*float64(m)/float64(n))) / h2
		}
	}

	return eig
}

// EigenvaluesPeriodic computes eigenvalues for periodic BC.
// λ_m = (2 - 2*cos(2πm/N)) / h².
func EigenvaluesPeriodic(n int, h float64) []float64 {
	return Eigenvalues(n, h, poisson.Periodic)
}

// EigenvaluesDirichlet computes eigenvalues for Dirichlet BC.
// λ_m = (2 - 2*cos(πm/(N+1))) / h² for m = 1..N.
func EigenvaluesDirichlet(n int, h float64) []float64 {
	return Eigenvalues(n, h, poisson.Dirichlet)
}

// EigenvaluesNeumann computes eigenvalues for Neumann BC.
// λ_m = (2 - 2*cos(πm/N)) / h².
func EigenvaluesNeumann(n int, h float64) []float64 {
	return Eigenvalues(n, h, poisson.Neumann)
}

// HasZeroEigenvalue returns true if the given BC has a zero eigenvalue
// (nullspace / constant mode).
func HasZeroEigenvalue(bc poisson.BCType) bool {
	return bc == poisson.Periodic || bc == poisson.Neumann
}

// ZeroEigenvalueIndex returns the index of the zero eigenvalue for BC types
// that have one. Returns -1 if no zero eigenvalue exists.
func ZeroEigenvalueIndex(bc poisson.BCType) int {
	switch bc {
	case poisson.Periodic, poisson.Neumann:
		return 0 // The m=0 mode has zero eigenvalue
	case poisson.Dirichlet:
		return -1 // Dirichlet has no zero eigenvalue
	default:
		return -1
	}
}
