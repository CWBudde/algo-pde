// Package plate assembles and solves transverse structural modes of thin,
// orthotropic plates with line stiffeners.
//
// The finite element has three degrees of freedom per mesh node: transverse
// displacement and two normal rotations. It is a linear Mindlin-Reissner
// triangular plate with centroid reduced shear integration, orthotropic bending and
// shear constitutive matrices, consistent translational/rotary mass, and
// Euler-Bernoulli line-rib contributions. This is a structural plate model;
// it is not a scalar Laplacian or acoustic Helmholtz approximation.
// Reduced shear integration mitigates, but does not eliminate, thin-plate shear
// locking. Callers should verify frequency convergence on their mesh; this
// package does not claim to implement a discrete Kirchhoff triangle (DKT).
package plate
