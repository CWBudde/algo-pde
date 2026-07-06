package poisson

// Variable-coefficient elliptic solver:
//
//	−∇·(a(x)∇u) = f
//
// with a spatially varying, strictly positive coefficient a(x). A varying
// coefficient makes the operator non-separable, so the spectral plan can no
// longer invert it directly. Instead the discrete operator L_a (a symmetric
// positive (semi-)definite matrix) is inverted by preconditioned conjugate
// gradient (PCG), using the fast constant-coefficient spectral solve of the
// nearby operator −c̄·Δ as the preconditioner. The spectral solve captures the
// elliptic smoothing of the constant part of a in O(N log N), so PCG converges
// in a handful of iterations even for a high-contrast coefficient.
//
// Discretization. L_a uses the standard second-order flux form; along one axis
//
//	(L_a u)_i = [ a_{i+½}(u_i − u_{i+1}) + a_{i−½}(u_i − u_{i−1}) ] / h²,
//
// where a_{i+½} is the coefficient averaged onto the face between nodes i and
// i+1 (harmonic by default, arithmetic via WithArithmeticAveraging). Summing the
// per-axis contributions gives the full operator. Boundaries reuse the same
// ghost reflection the constant-coefficient stencil uses (fd.Apply): the ghost
// value is c·u_boundary with c = 0 for a vertex-centered Dirichlet face, +1 for
// Neumann, −1 for a quarter-wave (mixed-axis) Dirichlet face, and the boundary
// face coefficient is the nearest cell's value (the even extension of a). With
// a ≡ 1 the operator reduces exactly to fd.Apply.
//
// Sample a(x) at the same node coordinates as f — see the package Grid
// Conventions; the per-axis BC fixes where those nodes sit.

import "math"

// Default PCG controls.
const (
	defaultVarCoeffTol     = 1e-8
	defaultVarCoeffMaxIter = 1000
)

// SolveStats reports the outcome of a variable-coefficient solve. Iterations is
// the number of PCG iterations performed; Residual is the final relative
// residual ‖f − L_a u‖₂ / ‖f‖₂ (both measured on the mean-projected RHS for
// nullspace-bearing boundary conditions).
type SolveStats struct {
	Iterations int
	Residual   float64
}

// varCoeffConfig holds the resolved options for a variable-coefficient plan.
type varCoeffConfig struct {
	tol        float64
	maxIter    int
	arithmetic bool
	cbar       float64 // preconditioner coefficient; 0 means "use mean(a)"
	workers    int
}

func defaultVarCoeffConfig() varCoeffConfig {
	return varCoeffConfig{
		tol:     defaultVarCoeffTol,
		maxIter: defaultVarCoeffMaxIter,
	}
}

// VarCoeffOption configures a VariableCoeffPlan. It is a distinct type from the
// spectral-plan Option so that iterative-solver knobs (tolerance, iteration
// cap, averaging) never silently no-op on a plain spectral plan.
type VarCoeffOption func(*varCoeffConfig)

// WithTolerance sets the relative residual tolerance ‖r‖₂ ≤ tol·‖f‖₂ at which
// the PCG iteration stops. The default is 1e-8. Non-positive values are ignored.
func WithTolerance(relTol float64) VarCoeffOption {
	return func(c *varCoeffConfig) {
		if relTol > 0 {
			c.tol = relTol
		}
	}
}

// WithMaxIterations caps the number of PCG iterations before Solve returns
// ErrNotConverged. The default is 1000. Non-positive values are ignored.
func WithMaxIterations(n int) VarCoeffOption {
	return func(c *varCoeffConfig) {
		if n > 0 {
			c.maxIter = n
		}
	}
}

// WithArithmeticAveraging averages the coefficient onto cell faces with the
// arithmetic mean (a_i+a_{i+1})/2 instead of the default harmonic mean
// 2·a_i·a_{i+1}/(a_i+a_{i+1}). Harmonic averaging is flux-conservative and more
// robust across sharp coefficient jumps; arithmetic is marginally cheaper and
// fine for smooth coefficients.
func WithArithmeticAveraging() VarCoeffOption {
	return func(c *varCoeffConfig) {
		c.arithmetic = true
	}
}

// WithPreconditionerCoefficient overrides the constant coefficient c̄ of the
// spectral preconditioner −c̄·Δ. The default is the arithmetic mean of a, which
// is a robust choice; tuning it (e.g. toward the geometric mean for high
// contrast) can reduce the iteration count. Non-positive values are ignored.
func WithPreconditionerCoefficient(c float64) VarCoeffOption {
	return func(cfg *varCoeffConfig) {
		if c > 0 {
			cfg.cbar = c
		}
	}
}

// WithParallelism sets the worker count of the inner spectral preconditioner
// (forwarded to the spectral plan's WithWorkers). 0 uses GOMAXPROCS.
func WithParallelism(n int) VarCoeffOption {
	return func(c *varCoeffConfig) {
		c.workers = n
	}
}

// cgScratch holds the per-call PCG work vectors. It is drawn from a pool per
// Solve call so solves are safe for concurrent use and allocation-free in the
// steady state.
type cgScratch struct {
	r  []float64 // residual
	z  []float64 // preconditioned residual
	p  []float64 // search direction
	ap []float64 // operator applied to p
}

// VariableCoeffPlan solves −∇·(a(x)∇u) = f for a fixed, strictly positive
// coefficient field a. It is reusable and safe for concurrent Solve calls.
type VariableCoeffPlan struct {
	dim       int
	n         []int
	size      int
	invH2     []float64
	stride    []int
	bcs       []BCType
	nullspace bool // operator is singular (all axes nullspace-bearing)

	a      []float64   // coefficient at each node
	faceHi [][]float64 // per axis: coefficient of the +axis face of each node

	precond *Plan // spectral solve of −c̄·Δ (the preconditioner)
	cbar    float64
	tol     float64
	maxIter int

	pool *residentPool[cgScratch]
}

// NewVariableCoeffPlan creates a plan that solves −∇·(a(x)∇u) = f on a dim-
// dimensional grid (dim 1, 2, or 3) with extents n, spacings h, and per-axis
// boundary conditions bcs. The coefficient field a is sampled at the same nodes
// as the RHS (row-major, index i·ny·nz + j·nz + k in 3D) and must be finite and
// strictly positive everywhere.
//
// It returns ErrInvalidCoefficient if a is the wrong length or non-positive/
// non-finite, and the same construction errors as NewPlan for bad dim/n/h/bcs.
func NewVariableCoeffPlan(dim int, n []int, h []float64, bcs []BCType, a []float64, opts ...VarCoeffOption) (*VariableCoeffPlan, error) {
	cfg := defaultVarCoeffConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// The inner spectral plan validates dim/n/h/bcs (and rejects unknown BCs),
	// so build it first and reuse its checks. Nullspace-bearing operators need
	// mean subtraction so the preconditioner tolerates the iterate's residual
	// mean; a Dirichlet or mixed face removes the nullspace.
	null := allNullBCs(bcs)
	planOpts := []Option{WithWorkers(cfg.workers)}
	if null {
		planOpts = append(planOpts, WithSubtractMean())
	}
	precond, err := NewPlan(dim, n, h, bcs, planOpts...)
	if err != nil {
		return nil, err
	}

	size := 1
	for axis := range dim {
		size *= n[axis]
	}

	if len(a) != size {
		return nil, ErrInvalidCoefficient
	}
	for _, v := range a {
		if v <= 0 || math.IsInf(v, 0) || math.IsNaN(v) {
			return nil, ErrInvalidCoefficient
		}
	}

	p := &VariableCoeffPlan{
		dim:       dim,
		n:         append([]int(nil), n[:dim]...),
		size:      size,
		invH2:     make([]float64, dim),
		stride:    make([]int, dim),
		bcs:       append([]BCType(nil), bcs[:dim]...),
		nullspace: null,
		a:         append([]float64(nil), a...),
		faceHi:    make([][]float64, dim),
		precond:   precond,
		tol:       cfg.tol,
		maxIter:   cfg.maxIter,
	}

	p.stride[dim-1] = 1
	for axis := dim - 2; axis >= 0; axis-- {
		p.stride[axis] = p.stride[axis+1] * p.n[axis+1]
	}
	for axis := range dim {
		p.invH2[axis] = 1.0 / (h[axis] * h[axis])
	}

	p.cbar = cfg.cbar
	if p.cbar <= 0 {
		mean, _ := meanAndMaxAbs(p.a)
		p.cbar = mean
	}

	p.computeFaceCoeffs(cfg.arithmetic)

	p.pool = newResidentPool[cgScratch](1)
	p.pool.put(p.newScratch())

	return p, nil
}

// allNullBCs reports whether every axis is nullspace-bearing, so the whole
// operator is singular on the constant mode.
func allNullBCs(bcs []BCType) bool {
	for _, b := range bcs {
		if !b.HasNullspace() {
			return false
		}
	}
	return true
}

// faceMean averages two positive cell coefficients onto the face between them.
// The harmonic branch uses the reciprocal form 2/(1/ai+1/aj) rather than the
// algebraically equal 2·ai·aj/(ai+aj): the latter overflows to Inf/NaN when the
// product ai·aj overflows even though both inputs are finite, whereas the
// reciprocal form never forms that product and stays bounded by 2·min(ai, aj).
func faceMean(ai, aj float64, arithmetic bool) float64 {
	if arithmetic {
		return 0.5 * (ai + aj)
	}
	return 2.0 / (1.0/ai + 1.0/aj)
}

// ApplyOperator writes the variable-coefficient operator L_a·src into dst. It is
// the operator that Solve inverts, exposed for residual checks. It is safe to
// call with dst == src.
func (p *VariableCoeffPlan) ApplyOperator(dst, src []float64) error {
	if dst == nil || src == nil {
		return ErrNilBuffer
	}
	if len(dst) != p.size || len(src) != p.size {
		return ErrSizeMismatch
	}
	// The matvec reads neighbours of src while writing dst, so an aliased call
	// needs an intact copy of src. Borrow a scratch buffer from the pool rather
	// than allocating, keeping ApplyOperator allocation-free even in a
	// residual-check loop.
	if p.size > 0 && &dst[0] == &src[0] {
		s := p.getScratch()
		defer p.pool.put(s)
		copy(s.r, src)
		p.applyVarCoeff(dst, s.r)
		return nil
	}
	p.applyVarCoeff(dst, src)
	return nil
}

// ghostCoeffLow / ghostCoeffHigh give the reflection coefficient c for the ghost
// node just outside a non-periodic boundary (ghost value = c·u_boundary),
// mirroring fd's low/high ghost coefficients: 0 vertex-Dirichlet, +1 Neumann,
// −1 quarter-wave (mixed-axis) Dirichlet.
func ghostCoeffLow(b BCType) float64 {
	switch b {
	case Neumann, NeumannDirichlet:
		return 1
	case DirichletNeumann:
		return -1
	default: // Dirichlet, Periodic (periodic handled separately)
		return 0
	}
}

func ghostCoeffHigh(b BCType) float64 {
	switch b {
	case Neumann, DirichletNeumann:
		return 1
	case NeumannDirichlet:
		return -1
	default:
		return 0
	}
}

// Solve computes the solution of −∇·(a∇u) = rhs into dst by preconditioned
// conjugate gradient and returns convergence statistics. For nullspace-bearing
// boundary conditions (all-Neumann or all-Periodic) the RHS mean is projected
// out and dst is returned with zero mean. It returns ErrNotConverged (with
// partial stats) if the tolerance is not reached within the iteration limit.
func (p *VariableCoeffPlan) Solve(dst, rhs []float64) (SolveStats, error) {
	if dst == nil || rhs == nil {
		return SolveStats{}, ErrNilBuffer
	}
	if len(dst) != p.size || len(rhs) != p.size {
		return SolveStats{}, ErrSizeMismatch
	}

	s := p.getScratch()
	defer p.pool.put(s)

	r, z, dir, ap := s.r, s.z, s.p, s.ap

	// x0 = 0, so the initial residual is the (mean-projected) RHS.
	copy(r, rhs)
	if p.nullspace {
		mean, _ := meanAndMaxAbs(r)
		for i := range r {
			r[i] -= mean
		}
	}
	for i := range dst {
		dst[i] = 0
	}

	bnorm := math.Sqrt(dotProduct(r, r))
	if bnorm == 0 {
		return SolveStats{Iterations: 0, Residual: 0}, nil
	}
	tolAbs := p.tol * bnorm

	if err := p.applyPrecond(z, r); err != nil {
		return SolveStats{}, err
	}
	copy(dir, z)
	rz := dotProduct(r, z)

	for k := range p.maxIter {
		p.applyVarCoeff(ap, dir)
		pAp := dotProduct(dir, ap)
		if pAp <= 0 {
			// A non-positive curvature should not occur for an SPD operator with
			// a mean-projected direction; bail rather than divide by ~0.
			return SolveStats{Iterations: k + 1, Residual: math.Sqrt(dotProduct(r, r)) / bnorm}, ErrNotConverged
		}
		alpha := rz / pAp
		for i := range dst {
			dst[i] += alpha * dir[i]
			r[i] -= alpha * ap[i]
		}

		rnorm := math.Sqrt(dotProduct(r, r))
		if rnorm <= tolAbs {
			p.finalizeMean(dst)
			return SolveStats{Iterations: k + 1, Residual: rnorm / bnorm}, nil
		}

		if err := p.applyPrecond(z, r); err != nil {
			return SolveStats{Iterations: k + 1, Residual: rnorm / bnorm}, err
		}
		rzNew := dotProduct(r, z)
		beta := rzNew / rz
		for i := range dir {
			dir[i] = z[i] + beta*dir[i]
		}
		rz = rzNew
	}

	p.finalizeMean(dst)
	return SolveStats{Iterations: p.maxIter, Residual: math.Sqrt(dotProduct(r, r)) / bnorm}, ErrNotConverged
}

// applyPrecond computes z = M⁻¹r = (1/c̄)·(−Δ)⁻¹r via the inner spectral solve.
func (p *VariableCoeffPlan) applyPrecond(z, r []float64) error {
	if err := p.precond.Solve(z, r); err != nil {
		return err
	}
	if inv := 1.0 / p.cbar; inv != 1.0 {
		for i := range z {
			z[i] *= inv
		}
	}
	return nil
}

// finalizeMean pins the solution mean to zero for singular operators, where the
// solution is only determined up to an additive constant.
func (p *VariableCoeffPlan) finalizeMean(dst []float64) {
	if !p.nullspace {
		return
	}
	mean, _ := meanAndMaxAbs(dst)
	for i := range dst {
		dst[i] -= mean
	}
}

// computeFaceCoeffs precomputes, for each axis, the coefficient of the face on
// the high (+axis) side of every node. The low-side face of a node is the
// high-side face of its lower neighbour; boundary faces (read directly from a in
// the matvec) are set to the node's own coefficient for the periodic-free case.
func (p *VariableCoeffPlan) computeFaceCoeffs(arithmetic bool) {
	for axis := range p.dim {
		n := p.n[axis]
		stride := p.stride[axis]
		periodic := p.bcs[axis] == Periodic
		face := make([]float64, p.size)
		for idx := range p.size {
			i := (idx / stride) % n
			switch {
			case i < n-1:
				face[idx] = faceMean(p.a[idx], p.a[idx+stride], arithmetic)
			case periodic:
				face[idx] = faceMean(p.a[idx], p.a[idx-(n-1)*stride], arithmetic)
			default:
				face[idx] = p.a[idx] // boundary face; unused (matvec reads a directly)
			}
		}
		p.faceHi[axis] = face
	}
}

func (p *VariableCoeffPlan) newScratch() *cgScratch {
	return &cgScratch{
		r:  make([]float64, p.size),
		z:  make([]float64, p.size),
		p:  make([]float64, p.size),
		ap: make([]float64, p.size),
	}
}

func (p *VariableCoeffPlan) getScratch() *cgScratch {
	if s := p.pool.get(); s != nil {
		return s
	}
	return p.newScratch()
}

// applyVarCoeff computes dst = L_a·src. dst and src must not alias.
func (p *VariableCoeffPlan) applyVarCoeff(dst, src []float64) {
	for i := range dst {
		dst[i] = 0
	}

	for axis := range p.dim {
		n := p.n[axis]
		stride := p.stride[axis]
		invH2 := p.invH2[axis]
		face := p.faceHi[axis]
		periodic := p.bcs[axis] == Periodic
		cLow := ghostCoeffLow(p.bcs[axis])
		cHigh := ghostCoeffHigh(p.bcs[axis])

		for idx := range p.size {
			i := (idx / stride) % n
			u := src[idx]

			var uHigh, faceHiVal float64
			switch {
			case i < n-1:
				uHigh, faceHiVal = src[idx+stride], face[idx]
			case periodic:
				uHigh, faceHiVal = src[idx-(n-1)*stride], face[idx]
			default:
				uHigh, faceHiVal = cHigh*u, p.a[idx]
			}

			var uLow, faceLoVal float64
			switch {
			case i > 0:
				uLow, faceLoVal = src[idx-stride], face[idx-stride]
			case periodic:
				uLow, faceLoVal = src[idx+(n-1)*stride], face[idx+(n-1)*stride]
			default:
				uLow, faceLoVal = cLow*u, p.a[idx]
			}

			dst[idx] += (faceHiVal*(u-uHigh) + faceLoVal*(u-uLow)) * invH2
		}
	}
}

// dotProduct returns the Euclidean inner product of two equal-length vectors.
func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
