// Package bc defines boundary condition types and the discrete negative
// Laplacian eigenvalue formulas shared across the solver packages.
//
// It is a leaf package: it imports only the standard library so that both the
// finite-difference package (fd) and the Poisson solver package (poisson) can
// depend on it without creating an import cycle. The eigenvalue closed forms
// live here as the single source of truth.
package bc
