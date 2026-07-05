package poisson

// Helmholtz–Hodge projection of a periodic velocity field onto its
// divergence-free subspace.
//
// Given a velocity field u*, the projection finds the divergence-free field u
// nearest to u* by removing a pure-gradient (curl-free) component:
//
//	u = u* − ∇φ,   where   Δφ = ∇·u*.
//
// This is the pressure-projection step of an incompressible flow solver: after
// an unconstrained velocity update produces u* with ∇·u* ≠ 0, projection
// restores incompressibility, and φ plays the role of the pressure.
//
// Discretization. The gradient uses forward differences and the divergence uses
// backward differences on the same collocated periodic grid. Their composition
// D∘G is then exactly the standard second-order periodic Laplacian that the
// internal spectral Poisson plan inverts — the collocated-grid analogue of MAC
// staggering. Because of that exact match the projected field is divergence-free
// to solver round-off (~1e-13 in the default float64 path), not merely to the
// second-order truncation error of the operators.
//
// The internal plan runs with WithSubtractMean: the backward-difference
// divergence of a periodic field telescopes to zero analytically, but round-off
// leaves a tiny non-zero mean that the default zero-mode gate would reject.

// projScratch holds the per-call buffers a projection needs: the negated
// divergence fed to the Poisson solve and the resulting pressure field. It is
// drawn from a pool per Project call so projections are safe for concurrent use.
type projScratch struct {
	rhs []float64
	phi []float64
}

func projectionOptions(opts []Option) []Option {
	// Force subtract-mean handling last so it wins over any caller nullspace
	// choice; a genuine DC offset in the divergence is round-off, not signal.
	planOpts := make([]Option, 0, len(opts)+1)
	planOpts = append(planOpts, opts...)
	planOpts = append(planOpts, WithSubtractMean())
	return planOpts
}

// ProjectionPlan2D projects 2D periodic velocity fields onto their
// divergence-free part. It is reusable and safe for concurrent Project calls.
type ProjectionPlan2D struct {
	nx, ny int
	hx, hy float64
	plan   *Plan2DPeriodic
	pool   *residentPool[projScratch]
}

// NewProjectionPlan2D creates a projection plan for an nx×ny periodic grid with
// spacings hx, hy. Options are forwarded to the internal periodic Poisson plan
// (e.g. WithWorkers); WithFloat32/WithRealFFT lower the projection accuracy from
// ~1e-13 to ~1e-6 along with the solve.
func NewProjectionPlan2D(nx, ny int, hx, hy float64, opts ...Option) (*ProjectionPlan2D, error) {
	plan, err := NewPlan2DPeriodic(nx, ny, hx, hy, projectionOptions(opts)...)
	if err != nil {
		return nil, err
	}

	n := nx * ny
	pool := newResidentPool[projScratch](1)
	pool.put(&projScratch{rhs: make([]float64, n), phi: make([]float64, n)})

	return &ProjectionPlan2D{nx: nx, ny: ny, hx: hx, hy: hy, plan: plan, pool: pool}, nil
}

func (p *ProjectionPlan2D) getScratch() *projScratch {
	if s := p.pool.get(); s != nil {
		return s
	}
	n := p.nx * p.ny
	return &projScratch{rhs: make([]float64, n), phi: make([]float64, n)}
}

// Project makes the velocity field (u, v) divergence-free in place. u and v are
// row-major nx×ny grids (index i*ny+j; u is the x-component, v the y-component).
func (p *ProjectionPlan2D) Project(u, v []float64) error {
	if u == nil || v == nil {
		return ErrNilBuffer
	}

	n := p.nx * p.ny
	if len(u) != n || len(v) != n {
		return ErrSizeMismatch
	}

	s := p.getScratch()
	defer p.pool.put(s)

	// rhs = −∇·u*, so that solving (−Δ)φ = rhs gives Δφ = ∇·u*.
	p.divergence(s.rhs, u, v)
	for i := range s.rhs {
		s.rhs[i] = -s.rhs[i]
	}

	if err := p.plan.Solve(s.phi, s.rhs); err != nil {
		return err
	}

	p.subtractGradient(u, v, s.phi)
	return nil
}

// Divergence writes the discrete (backward-difference) divergence of the
// velocity field (u, v) into out. This is the same operator Project drives to
// zero, so out is at solver round-off for a field returned by Project.
func (p *ProjectionPlan2D) Divergence(out, u, v []float64) error {
	if out == nil || u == nil || v == nil {
		return ErrNilBuffer
	}

	n := p.nx * p.ny
	if len(out) != n || len(u) != n || len(v) != n {
		return ErrSizeMismatch
	}

	p.divergence(out, u, v)
	return nil
}

func (p *ProjectionPlan2D) divergence(out, u, v []float64) {
	nx, ny := p.nx, p.ny
	invHx := 1.0 / p.hx
	invHy := 1.0 / p.hy

	for i := range nx {
		im1 := (i - 1 + nx) % nx
		row := i * ny
		rowM := im1 * ny
		for j := range ny {
			jm1 := (j - 1 + ny) % ny
			idx := row + j
			dudx := (u[idx] - u[rowM+j]) * invHx
			dvdy := (v[idx] - v[row+jm1]) * invHy
			out[idx] = dudx + dvdy
		}
	}
}

func (p *ProjectionPlan2D) subtractGradient(u, v, phi []float64) {
	nx, ny := p.nx, p.ny
	invHx := 1.0 / p.hx
	invHy := 1.0 / p.hy

	for i := range nx {
		ip1 := (i + 1) % nx
		row := i * ny
		rowP := ip1 * ny
		for j := range ny {
			jp1 := (j + 1) % ny
			idx := row + j
			u[idx] -= (phi[rowP+j] - phi[idx]) * invHx
			v[idx] -= (phi[row+jp1] - phi[idx]) * invHy
		}
	}
}

// ProjectionPlan3D projects 3D periodic velocity fields onto their
// divergence-free part. It is reusable and safe for concurrent Project calls.
type ProjectionPlan3D struct {
	nx, ny, nz int
	hx, hy, hz float64
	plan       *Plan3DPeriodic
	pool       *residentPool[projScratch]
}

// NewProjectionPlan3D creates a projection plan for an nx×ny×nz periodic grid
// with spacings hx, hy, hz. Options are forwarded to the internal periodic
// Poisson plan.
func NewProjectionPlan3D(nx, ny, nz int, hx, hy, hz float64, opts ...Option) (*ProjectionPlan3D, error) {
	plan, err := NewPlan3DPeriodic(nx, ny, nz, hx, hy, hz, projectionOptions(opts)...)
	if err != nil {
		return nil, err
	}

	n := nx * ny * nz
	pool := newResidentPool[projScratch](1)
	pool.put(&projScratch{rhs: make([]float64, n), phi: make([]float64, n)})

	return &ProjectionPlan3D{
		nx: nx, ny: ny, nz: nz,
		hx: hx, hy: hy, hz: hz,
		plan: plan, pool: pool,
	}, nil
}

func (p *ProjectionPlan3D) getScratch() *projScratch {
	if s := p.pool.get(); s != nil {
		return s
	}
	n := p.nx * p.ny * p.nz
	return &projScratch{rhs: make([]float64, n), phi: make([]float64, n)}
}

// Project makes the velocity field (u, v, w) divergence-free in place. Each
// component is a row-major nx×ny×nz grid (index (i*ny+j)*nz+k).
func (p *ProjectionPlan3D) Project(u, v, w []float64) error {
	if u == nil || v == nil || w == nil {
		return ErrNilBuffer
	}

	n := p.nx * p.ny * p.nz
	if len(u) != n || len(v) != n || len(w) != n {
		return ErrSizeMismatch
	}

	s := p.getScratch()
	defer p.pool.put(s)

	p.divergence(s.rhs, u, v, w)
	for i := range s.rhs {
		s.rhs[i] = -s.rhs[i]
	}

	if err := p.plan.Solve(s.phi, s.rhs); err != nil {
		return err
	}

	p.subtractGradient(u, v, w, s.phi)
	return nil
}

// Divergence writes the discrete (backward-difference) divergence of (u, v, w)
// into out.
func (p *ProjectionPlan3D) Divergence(out, u, v, w []float64) error {
	if out == nil || u == nil || v == nil || w == nil {
		return ErrNilBuffer
	}

	n := p.nx * p.ny * p.nz
	if len(out) != n || len(u) != n || len(v) != n || len(w) != n {
		return ErrSizeMismatch
	}

	p.divergence(out, u, v, w)
	return nil
}

func (p *ProjectionPlan3D) divergence(out, u, v, w []float64) {
	nx, ny, nz := p.nx, p.ny, p.nz
	plane := ny * nz
	invHx := 1.0 / p.hx
	invHy := 1.0 / p.hy
	invHz := 1.0 / p.hz

	for i := range nx {
		im1 := (i - 1 + nx) % nx
		iPlane := i * plane
		iPlaneM := im1 * plane
		for j := range ny {
			jm1 := (j - 1 + ny) % ny
			row := iPlane + j*nz
			rowM := iPlane + jm1*nz
			for k := range nz {
				km1 := (k - 1 + nz) % nz
				idx := row + k
				dudx := (u[idx] - u[iPlaneM+j*nz+k]) * invHx
				dvdy := (v[idx] - v[rowM+k]) * invHy
				dwdz := (w[idx] - w[row+km1]) * invHz
				out[idx] = dudx + dvdy + dwdz
			}
		}
	}
}

func (p *ProjectionPlan3D) subtractGradient(u, v, w, phi []float64) {
	nx, ny, nz := p.nx, p.ny, p.nz
	plane := ny * nz
	invHx := 1.0 / p.hx
	invHy := 1.0 / p.hy
	invHz := 1.0 / p.hz

	for i := range nx {
		ip1 := (i + 1) % nx
		iPlane := i * plane
		iPlaneP := ip1 * plane
		for j := range ny {
			jp1 := (j + 1) % ny
			row := iPlane + j*nz
			rowP := iPlane + jp1*nz
			for k := range nz {
				kp1 := (k + 1) % nz
				idx := row + k
				u[idx] -= (phi[iPlaneP+j*nz+k] - phi[idx]) * invHx
				v[idx] -= (phi[rowP+k] - phi[idx]) * invHy
				w[idx] -= (phi[row+kp1] - phi[idx]) * invHz
			}
		}
	}
}
