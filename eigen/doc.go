// Package eigen solves real symmetric generalized eigenproblems.
//
// Solve computes the lowest eigenpairs of
//
//	A x = lambda B x
//
// for symmetric A and symmetric positive-definite B. The implementation is a
// deterministic block locally-optimal preconditioned conjugate-gradient
// iteration. It uses operators rather than requiring a particular matrix
// representation; SparseSymmetric is provided for assembled finite-element
// systems.
package eigen
