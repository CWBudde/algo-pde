package poisson

// Masked (immersed-boundary) Poisson solver for non-rectangular domains:
//
//	−Δu = f   inside the physical domain,   u = 0   on the exterior.
//
// A non-rectangular domain (a disk, an L-shape, a room with an obstacle) is
// embedded in the enclosing rectangular grid. A boolean mask marks which cells
// are inside the domain (active, where −Δu = f is solved) and which are outside
// (masked/solid, where u is pinned to zero). This lets the fast O(N log N)
// spectral solver of the full rectangle serve as the preconditioner of an
// iterative solve that enforces the domain restriction — the fictitious-domain
// idea.
//
// Operator. On the active cells the standard second-order negative-Laplacian
// stencil is applied, with the bounding-box boundary conditions bcs governing
// the ghost rule at the rectangle's faces (reusing fd's per-face reflection). A
// masked neighbour contributes value 0, which is exactly a homogeneous
// (vertex-centered) Dirichlet ghost — so the immersed boundary is u = 0. On the
// masked cells the operator is the identity (u = 0). The resulting matrix is
//
//	[ L_AA   0  ]
//	[  0    I_M ],
//
// where L_AA is the principal submatrix of the bounding-box negative Laplacian
// on the active cells. L_AA is a principal submatrix of a symmetric positive
// (semi-)definite matrix, and the masked Dirichlet pins remove any nullspace, so
// the operator is SPD whenever at least one cell is masked — preconditioned
// conjugate gradient (PCG) applies.
//
// Preconditioner. M⁻¹r = R·(−Δ)_box⁻¹·r: one fast spectral solve of the full
// rectangle, then restrict (zero the masked entries). The masked entries of the
// residual, search direction and solution stay identically zero throughout the
// iteration, so the "masked neighbour = 0" assumption in the matvec holds
// automatically and the masked identity rows never affect convergence.
//
// Accuracy. The domain boundary is approximated by the staircase of cell edges,
// so convergence at the immersed boundary is first order in h even though the
// interior stencil is second order. Sample f at the same node coordinates as the
// underlying spectral plan (see the package Grid Conventions).
//
// Scope. Only a homogeneous Dirichlet immersed boundary (u = 0 on the solid) is
// supported. Inhomogeneous immersed Dirichlet data (u = g on the boundary) and
// Neumann / no-flux immersed boundaries (∂u/∂n = 0, sound-hard walls) are out of
// scope for now — the latter is singular like a pure-Neumann problem and needs
// separate compatibility handling.

import "math"

// maskedConfig holds the resolved options for a masked-domain plan.
type maskedConfig struct {
	tol     float64
	maxIter int
	workers int
}

func defaultMaskedConfig() maskedConfig {
	return maskedConfig{
		tol:     defaultVarCoeffTol,
		maxIter: defaultVarCoeffMaxIter,
	}
}

// MaskedOption configures a MaskedPlan. It is a distinct type from the
// spectral-plan Option so that iterative-solver knobs (tolerance, iteration cap,
// parallelism) never silently no-op on a plain spectral plan.
type MaskedOption func(*maskedConfig)

// WithMaskTolerance sets the relative residual tolerance ‖r‖₂ ≤ tol·‖f‖₂ (the
// norm taken over the active cells) at which the PCG iteration stops. The
// default is 1e-8. Non-positive values are ignored.
func WithMaskTolerance(relTol float64) MaskedOption {
	return func(c *maskedConfig) {
		if relTol > 0 {
			c.tol = relTol
		}
	}
}

// WithMaskMaxIterations caps the number of PCG iterations before Solve returns
// ErrNotConverged. The default is 1000. Non-positive values are ignored.
func WithMaskMaxIterations(n int) MaskedOption {
	return func(c *maskedConfig) {
		if n > 0 {
			c.maxIter = n
		}
	}
}

// WithMaskParallelism sets the worker count of the inner spectral preconditioner
// (forwarded to the spectral plan's WithWorkers). 0 uses GOMAXPROCS.
func WithMaskParallelism(n int) MaskedOption {
	return func(c *maskedConfig) {
		c.workers = n
	}
}

// MaskedPlan solves −Δu = f on the active cells of a masked grid with u = 0 on
// the masked (solid) cells. It is reusable and safe for concurrent Solve calls.
type MaskedPlan struct {
	dim    int
	n      []int
	size   int
	invH2  []float64
	stride []int
	bcs    []BCType

	mask []bool // true = active (inside the domain), false = masked (solid, u = 0)

	precond *Plan // spectral solve of the bounding-box −Δ (the preconditioner)
	tol     float64
	maxIter int

	pool *residentPool[cgScratch]
}

// NewMaskedPlan creates a plan that solves −Δu = f on the active cells of a dim-
// dimensional grid (dim 1, 2, or 3) with extents n, spacings h, and bounding-box
// boundary conditions bcs, pinning u = 0 on the masked cells. mask is sampled at
// the same nodes as the RHS (row-major, index i·ny·nz + j·nz + k in 3D): a true
// entry marks an active (interior) cell, a false entry a masked (solid) cell.
//
// It returns ErrInvalidMask if mask is the wrong length, has no active cells, or
// leaves the operator singular (an all-active mask combined with all-nullspace
// bounding-box BCs — that is just an unmasked periodic/Neumann problem, which
// NewPlan with WithSubtractMean already solves). It returns the same
// construction errors as NewPlan for bad dim/n/h/bcs.
func NewMaskedPlan(dim int, n []int, h []float64, bcs []BCType, mask []bool, opts ...MaskedOption) (*MaskedPlan, error) {
	cfg := defaultMaskedConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// The inner spectral plan validates dim/n/h/bcs (and rejects unknown BCs),
	// so build it first and reuse its checks. When every axis is nullspace-
	// bearing the preconditioner is a singular periodic/Neumann solve, so it
	// needs mean subtraction to tolerate the residual's DC component; the masked
	// Dirichlet pins keep the outer PCG operator itself nonsingular.
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

	if len(mask) != size {
		return nil, ErrInvalidMask
	}
	active := 0
	for _, m := range mask {
		if m {
			active++
		}
	}
	if active == 0 {
		return nil, ErrInvalidMask
	}
	if null && active == size {
		return nil, ErrInvalidMask
	}

	p := &MaskedPlan{
		dim:     dim,
		n:       append([]int(nil), n[:dim]...),
		size:    size,
		invH2:   make([]float64, dim),
		stride:  make([]int, dim),
		bcs:     append([]BCType(nil), bcs[:dim]...),
		mask:    append([]bool(nil), mask...),
		precond: precond,
		tol:     cfg.tol,
		maxIter: cfg.maxIter,
	}

	p.stride[dim-1] = 1
	for axis := dim - 2; axis >= 0; axis-- {
		p.stride[axis] = p.stride[axis+1] * p.n[axis+1]
	}
	for axis := range dim {
		p.invH2[axis] = 1.0 / (h[axis] * h[axis])
	}

	p.pool = newResidentPool[cgScratch](1)
	p.pool.put(p.newScratch())

	return p, nil
}

// Mask returns a copy of the plan's active/masked flags (true = active).
func (p *MaskedPlan) Mask() []bool {
	return append([]bool(nil), p.mask...)
}

// Solve computes the solution of −Δu = rhs on the active cells into dst by
// preconditioned conjugate gradient and returns convergence statistics. The
// masked cells are pinned to zero in dst and their rhs entries are ignored. It
// returns ErrNotConverged (with partial stats) if the tolerance is not reached
// within the iteration limit.
func (p *MaskedPlan) Solve(dst, rhs []float64) (SolveStats, error) {
	if dst == nil || rhs == nil {
		return SolveStats{}, ErrNilBuffer
	}
	if len(dst) != p.size || len(rhs) != p.size {
		return SolveStats{}, ErrSizeMismatch
	}

	s := p.getScratch()
	defer p.pool.put(s)
	r, z, dir, ap := s.r, s.z, s.p, s.ap

	// x0 = 0 ⇒ the initial residual is the RHS restricted to the active cells.
	// The masked entries of r stay identically zero for the whole iteration, so
	// the masked cells never enter the CG dynamics.
	copy(r, rhs)
	for idx := range r {
		if !p.mask[idx] {
			r[idx] = 0
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
		p.applyMasked(ap, dir)
		pAp := dotProduct(dir, ap)
		if pAp <= 0 {
			// A non-positive curvature should not occur for an SPD operator; bail
			// rather than divide by ~0.
			return SolveStats{Iterations: k + 1, Residual: math.Sqrt(dotProduct(r, r)) / bnorm}, ErrNotConverged
		}
		alpha := rz / pAp
		for i := range dst {
			dst[i] += alpha * dir[i]
			r[i] -= alpha * ap[i]
		}

		rnorm := math.Sqrt(dotProduct(r, r))
		if rnorm <= tolAbs {
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

	return SolveStats{Iterations: p.maxIter, Residual: math.Sqrt(dotProduct(r, r)) / bnorm}, ErrNotConverged
}

// ApplyOperator writes the masked operator into dst: the negative-Laplacian
// stencil (with masked neighbours read as the u = 0 Dirichlet ghost) on the
// active rows, and the identity on the masked rows. It is the operator that
// Solve inverts, exposed for residual checks, and is safe to call with
// dst == src.
func (p *MaskedPlan) ApplyOperator(dst, src []float64) error {
	if dst == nil || src == nil {
		return ErrNilBuffer
	}
	if len(dst) != p.size || len(src) != p.size {
		return ErrSizeMismatch
	}

	// Borrow scratch rather than allocate. stencilIn is src with masked entries
	// zeroed (so an active cell's masked neighbours act as u = 0 ghosts
	// regardless of what the caller passed there); maskedOrig preserves the
	// original masked values for the identity rows, which matters when dst
	// aliases src (applyMasked would otherwise clobber them).
	s := p.getScratch()
	defer p.pool.put(s)
	stencilIn, maskedOrig := s.r, s.z
	copy(stencilIn, src)
	copy(maskedOrig, src)
	for idx := range stencilIn {
		if !p.mask[idx] {
			stencilIn[idx] = 0
		}
	}
	p.applyMasked(dst, stencilIn)
	for idx := range dst {
		if !p.mask[idx] {
			dst[idx] = maskedOrig[idx]
		}
	}
	return nil
}

// applyPrecond computes z = M⁻¹r: one full-rectangle spectral solve followed by
// the restriction R that zeroes the masked entries.
func (p *MaskedPlan) applyPrecond(z, r []float64) error {
	if err := p.precond.Solve(z, r); err != nil {
		return err
	}
	for idx := range z {
		if !p.mask[idx] {
			z[idx] = 0
		}
	}
	return nil
}

// applyMasked computes dst = L·src, where L is the masked operator. It assumes
// the masked entries of src are zero (an active cell reads a masked neighbour as
// the u = 0 ghost); the masked rows are set to the identity (dst = src). dst and
// src must not alias.
func (p *MaskedPlan) applyMasked(dst, src []float64) {
	for i := range dst {
		dst[i] = 0
	}

	for axis := range p.dim {
		n := p.n[axis]
		stride := p.stride[axis]
		invH2 := p.invH2[axis]
		periodic := p.bcs[axis] == Periodic
		cLow := ghostCoeffLow(p.bcs[axis])
		cHigh := ghostCoeffHigh(p.bcs[axis])

		for idx := range p.size {
			if !p.mask[idx] {
				continue // masked rows handled after the axis loop
			}
			i := (idx / stride) % n
			u := src[idx]

			var uHigh float64
			switch {
			case i < n-1:
				uHigh = src[idx+stride]
			case periodic:
				uHigh = src[idx-(n-1)*stride]
			default:
				uHigh = cHigh * u
			}

			var uLow float64
			switch {
			case i > 0:
				uLow = src[idx-stride]
			case periodic:
				uLow = src[idx+(n-1)*stride]
			default:
				uLow = cLow * u
			}

			dst[idx] += ((u - uHigh) + (u - uLow)) * invH2
		}
	}

	for idx := range p.size {
		if !p.mask[idx] {
			dst[idx] = src[idx]
		}
	}
}

func (p *MaskedPlan) newScratch() *cgScratch {
	return &cgScratch{
		r:  make([]float64, p.size),
		z:  make([]float64, p.size),
		p:  make([]float64, p.size),
		ap: make([]float64, p.size),
	}
}

func (p *MaskedPlan) getScratch() *cgScratch {
	if s := p.pool.get(); s != nil {
		return s
	}
	return p.newScratch()
}
