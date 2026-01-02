// Package r2r provides real-to-real transforms (DST/DCT) implemented via FFT.
//
// These transforms are used internally by the Poisson solver to handle
// Dirichlet and Neumann boundary conditions.
//
// # Discrete Sine Transform (DST)
//
// DST is used for Dirichlet boundary conditions where u = 0 at boundaries.
// The transform diagonalizes the discrete Laplacian with these BCs.
//
// # Discrete Cosine Transform (DCT)
//
// DCT is used for Neumann boundary conditions where ∂u/∂n = 0 at boundaries.
// The transform diagonalizes the discrete Laplacian with these BCs.
//
// # Implementation
//
// Both DST and DCT are implemented via FFT using standard embedding techniques:
//   - DST: Odd extension of the data, then FFT
//   - DCT: Even extension of the data, then FFT
//
// This leverages the optimized FFT implementation from algo-fft.
package r2r
