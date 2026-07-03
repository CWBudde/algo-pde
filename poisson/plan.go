package poisson

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/grid"
)

// resonanceRelTol is the relative-cancellation threshold below which the
// Helmholtz operator is treated as singular. A mode is flagged resonant when
// |alpha + eigenvalues| drops below resonanceRelTol times the sum of the term
// magnitudes, i.e. the divide would amplify by more than ~1/resonanceRelTol and
// return a near-garbage field. Poisson problems never trip it: with alpha == 0
// all terms are non-negative, so |denom| == scale for every non-zero mode.
const resonanceRelTol = 1e-9

// Plan is a reusable Poisson/Helmholtz solver plan with per-axis boundary conditions.
type Plan struct {
	dim         int
	n           [3]int
	h           [3]float64
	bc          [3]BCType
	eig         [3][]float64
	tr          [3]AxisTransform
	work        *workspacePool
	realSize    int
	complexSize int
	opts        Options
	alpha       float64
}

// NewPlan creates a new Poisson plan with per-axis boundary conditions.
func NewPlan(dim int, n []int, h []float64, bc []BCType, opts ...Option) (*Plan, error) {
	return newPlanWithAlpha(dim, n, h, bc, 0, opts...)
}

// NewHelmholtzPlan creates a new Helmholtz plan for (alpha - Δ)u = f.
// Negative alpha values are allowed but may lead to singular operators when
// alpha cancels an eigenvalue; Solve will return ErrResonant in that case.
func NewHelmholtzPlan(dim int, n []int, h []float64, bc []BCType, alpha float64, opts ...Option) (*Plan, error) {
	return newPlanWithAlpha(dim, n, h, bc, alpha, opts...)
}

func newPlanWithAlpha(dim int, n []int, h []float64, bc []BCType, alpha float64, opts ...Option) (*Plan, error) {
	if dim < 1 || dim > 3 {
		return nil, &ValidationError{
			Field:   "dim",
			Message: "must be 1, 2, or 3",
		}
	}

	if len(n) != dim {
		return nil, &ValidationError{
			Field:   "n",
			Message: msgLenMustMatchDim,
		}
	}

	if len(h) != dim {
		return nil, &ValidationError{
			Field:   "h",
			Message: msgLenMustMatchDim,
		}
	}

	if len(bc) != dim {
		return nil, &ValidationError{
			Field:   "bc",
			Message: msgLenMustMatchDim,
		}
	}

	if !validAlpha(alpha) {
		return nil, ErrInvalidAlpha
	}

	options := ApplyOptions(DefaultOptions(), opts)
	options.Workers = effectiveWorkers(options.Workers)
	plan := &Plan{
		dim:   dim,
		n:     [3]int{1, 1, 1},
		h:     [3]float64{1, 1, 1},
		bc:    [3]BCType{Periodic, Periodic, Periodic},
		opts:  options,
		alpha: alpha,
	}

	size := 1
	for axis := range dim {
		if n[axis] < 1 {
			return nil, ErrInvalidSize
		}
		if !validSpacing(h[axis]) {
			return nil, ErrInvalidSpacing
		}

		switch bc[axis] {
		case Periodic, Dirichlet, Neumann:
		default:
			return nil, &ValidationError{
				Field:   fmt.Sprintf("bc[%d]", axis),
				Message: "unsupported boundary condition",
			}
		}

		plan.n[axis] = n[axis]
		plan.h[axis] = h[axis]
		plan.bc[axis] = bc[axis]
		size *= n[axis]
	}

	for axis := range dim {
		var err error
		switch plan.bc[axis] {
		case Periodic:
			plan.eig[axis] = eigenvaluesPeriodic(plan.n[axis], plan.h[axis])
			plan.tr[axis], err = newFFTAxisTransform(plan.n[axis], options.Workers)
		case Dirichlet:
			plan.eig[axis] = eigenvaluesDirichlet(plan.n[axis], plan.h[axis])
			plan.tr[axis], err = newDSTAxisTransform(plan.n[axis], options.Workers)
		case Neumann:
			plan.eig[axis] = eigenvaluesNeumann(plan.n[axis], plan.h[axis])
			plan.tr[axis], err = newDCTAxisTransform(plan.n[axis], options.Workers)
		}
		if err != nil {
			return nil, fmt.Errorf("axis %d: %w", axis, err)
		}
	}

	// The nullspace is fully determined by alpha and the boundary conditions,
	// both fixed at construction. If the problem has a nullspace, NullspaceError
	// makes every Solve fail, so reject the combination now rather than later.
	if plan.hasNullspace() && options.Nullspace == NullspaceError {
		return nil, ErrNullspace
	}

	realSize := 0
	if !options.InPlace {
		realSize = size
	}
	plan.realSize = realSize
	plan.complexSize = size
	plan.work = newWorkspacePool(realSize, size)

	return plan, nil
}

// Solve computes the solution into dst for a given RHS.
func (p *Plan) Solve(dst, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	size := p.size()
	if len(dst) != size || len(rhs) != size {
		return ErrSizeMismatch
	}

	workspace := p.work.get()
	defer p.work.put(workspace)

	return p.solve(dst, rhs, workspace)
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

// WorkBytes returns the size of the workspace buffers one Solve call uses, in
// bytes. Concurrent Solve calls each draw their own workspace, so the peak
// memory use is WorkBytes times the peak number of concurrent calls.
func (p *Plan) WorkBytes() int {
	return p.realSize*8 + p.complexSize*16
}

// solve runs the transform pipeline using the given per-call workspace.
func (p *Plan) solve(dst, rhs []float64, workspace *Workspace) error {
	hasNullspace := p.hasNullspace()

	offset := 0.0
	if hasNullspace {
		mean, maxAbs := meanAndMaxAbs(rhs)
		if p.opts.Nullspace == NullspaceZeroMode && !meanWithinTolerance(mean, maxAbs, p.meanRelTol()) {
			return ErrNonZeroMean
		}

		if p.opts.Nullspace == NullspaceSubtractMean {
			offset = mean
		}
	}

	for i, v := range rhs {
		workspace.Complex[i] = complex(v-offset, 0)
	}

	shape := p.shape()
	for axis := range p.dim {
		if err := p.tr[axis].Forward(workspace.Complex, shape, axis); err != nil {
			return fmt.Errorf("forward axis %d: %w", axis, err)
		}
	}

	if err := p.applyEigenvalues(workspace.Complex); err != nil {
		return err
	}

	for axis := p.dim - 1; axis >= 0; axis-- {
		if err := p.tr[axis].Inverse(workspace.Complex, shape, axis); err != nil {
			return fmt.Errorf("inverse axis %d: %w", axis, err)
		}
	}

	addMean := 0.0
	if hasNullspace && p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range workspace.Complex {
		dst[i] = real(workspace.Complex[i]) + addMean
	}

	return nil
}

func (p *Plan) shape() grid.Shape {
	return grid.Shape{p.n[0], p.n[1], p.n[2]}
}

func (p *Plan) size() int {
	size := 1
	for axis := range p.dim {
		size *= p.n[axis]
	}
	return size
}

// meanRelTol returns the zero-mean consistency tolerance for this plan. A
// Neumann axis samples the RHS at cell centers, whose midpoint quadrature
// leaves an O(h^2) mean even for a compatible problem, so the gate widens to
// O(1/n^2) on the coarsest Neumann axis. A pure-periodic problem (no Neumann
// axis) is integrated spectrally, so its compatible mean sits at roundoff and
// the gate stays tight — a real DC offset must still be rejected.
func (p *Plan) meanRelTol() float64 {
	minNeumann := 0
	for axis := range p.dim {
		if p.bc[axis] != Neumann {
			continue
		}
		if minNeumann == 0 || p.n[axis] < minNeumann {
			minNeumann = p.n[axis]
		}
	}

	if minNeumann == 0 {
		return meanRoundoffFloor
	}
	return discretizationMeanTol(minNeumann)
}

func (p *Plan) hasNullspace() bool {
	if p.alpha != 0 {
		return false
	}

	for axis := range p.dim {
		if !p.bc[axis].HasNullspace() {
			return false
		}
	}
	return true
}

func (p *Plan) applyEigenvalues(buf []complex128) error {
	_, ny, nz := p.n[0], p.n[1], p.n[2]
	strideYZ := ny * nz
	strideZ := nz
	allowZeroMode := p.hasNullspace()
	size := p.size()
	workers := clampWorkers(p.opts.Workers, size)

	return parallelFor(workers, size, func(_ int, start, end int) error {
		for idx := start; idx < end; idx++ {
			i := idx / strideYZ
			rem := idx % strideYZ
			j := rem / strideZ
			k := rem % strideZ

			// scale is the sum of the magnitudes of the terms that build denom.
			// For a well-conditioned mode denom == scale (all terms share sign),
			// but when a negative alpha nearly cancels the positive eigenvalue
			// sum, |denom| collapses far below scale. The ratio |denom|/scale is
			// exactly the catastrophic-cancellation conditioning of the divide,
			// so gating on it flags near-resonance before the ~1/|denom|
			// amplification turns the mode into garbage.
			denom := p.alpha + p.eig[0][i]
			scale := math.Abs(p.alpha) + math.Abs(p.eig[0][i])
			if p.dim > 1 {
				denom += p.eig[1][j]
				scale += math.Abs(p.eig[1][j])
			}
			if p.dim > 2 {
				denom += p.eig[2][k]
				scale += math.Abs(p.eig[2][k])
			}

			if math.Abs(denom) <= resonanceRelTol*scale {
				if allowZeroMode && i == 0 && (p.dim < 2 || j == 0) && (p.dim < 3 || k == 0) {
					buf[idx] = 0
					continue
				}
				return ErrResonant
			}

			buf[idx] /= complex(denom, 0)
		}
		return nil
	})
}
