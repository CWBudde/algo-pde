package bc

import (
	"errors"
	"math"
)

// ErrInvalidBC is returned when an unknown boundary condition type is passed.
var ErrInvalidBC = errors.New("bc: invalid boundary condition type")

// Eigenvalues computes the 1D eigenvalues of the discrete negative Laplacian
// for the given boundary condition type.
//
// Parameters:
//   - n: number of grid points
//   - h: grid spacing
//   - b: boundary condition type
//
// Returns a slice of eigenvalues. The length depends on the BC:
//   - Periodic: n eigenvalues (m = 0..n-1)
//   - Dirichlet: n eigenvalues (m = 1..n, but stored 0..n-1)
//   - Neumann: n eigenvalues (m = 0..n-1)
//
// It returns ErrInvalidBC if b is not a supported boundary condition.
func Eigenvalues(n int, h float64, b BCType) ([]float64, error) {
	switch b {
	case Periodic:
		return EigenvaluesPeriodic(n, h), nil
	case Dirichlet:
		return EigenvaluesDirichlet(n, h), nil
	case Neumann:
		return EigenvaluesNeumann(n, h), nil
	default:
		return nil, ErrInvalidBC
	}
}

// EigenvaluesPeriodic computes eigenvalues for periodic BC.
// λ_m = (2 - 2*cos(2πm/N)) / h² for m = 0..N-1.
func EigenvaluesPeriodic(n int, h float64) []float64 {
	eig := make([]float64, n)
	h2 := h * h
	for m := range n {
		eig[m] = (2.0 - 2.0*math.Cos(2.0*math.Pi*float64(m)/float64(n))) / h2
	}

	return eig
}

// EigenvaluesDirichlet computes eigenvalues for Dirichlet BC.
// λ_m = (2 - 2*cos(πm/(N+1))) / h² for m = 1..N, stored at index m-1.
func EigenvaluesDirichlet(n int, h float64) []float64 {
	eig := make([]float64, n)
	h2 := h * h
	for m := 1; m <= n; m++ {
		eig[m-1] = (2.0 - 2.0*math.Cos(math.Pi*float64(m)/float64(n+1))) / h2
	}

	return eig
}

// EigenvaluesNeumann computes eigenvalues for Neumann BC.
// λ_m = (2 - 2*cos(πm/N)) / h² for m = 0..N-1.
func EigenvaluesNeumann(n int, h float64) []float64 {
	eig := make([]float64, n)
	h2 := h * h
	for m := range n {
		eig[m] = (2.0 - 2.0*math.Cos(math.Pi*float64(m)/float64(n))) / h2
	}

	return eig
}
