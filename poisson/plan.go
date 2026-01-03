package poisson

import (
	"fmt"

	"github.com/MeKo-Tech/algo-pde/grid"
)

// Plan is a reusable Poisson/Helmholtz solver plan with per-axis boundary conditions.
type Plan struct {
	dim   int
	n     [3]int
	h     [3]float64
	bc    [3]BCType
	eig   [3][]float64
	tr    [3]AxisTransform
	work  Workspace
	opts  Options
	alpha float64
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
			Message: "length must match dim",
		}
	}

	if len(h) != dim {
		return nil, &ValidationError{
			Field:   "h",
			Message: "length must match dim",
		}
	}

	if len(bc) != dim {
		return nil, &ValidationError{
			Field:   "bc",
			Message: "length must match dim",
		}
	}

	options := ApplyOptions(DefaultOptions(), opts)
	plan := &Plan{
		dim:   dim,
		n:     [3]int{1, 1, 1},
		h:     [3]float64{1, 1, 1},
		bc:    [3]BCType{Periodic, Periodic, Periodic},
		opts:  options,
		alpha: alpha,
	}

	size := 1
	for axis := 0; axis < dim; axis++ {
		if n[axis] < 1 {
			return nil, ErrInvalidSize
		}
		if h[axis] <= 0 {
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

	for axis := 0; axis < dim; axis++ {
		var err error
		switch plan.bc[axis] {
		case Periodic:
			plan.eig[axis] = eigenvaluesPeriodic(plan.n[axis], plan.h[axis])
			plan.tr[axis], err = newFFTAxisTransform(plan.n[axis])
		case Dirichlet:
			plan.eig[axis] = eigenvaluesDirichlet(plan.n[axis], plan.h[axis])
			plan.tr[axis], err = newDSTAxisTransform(plan.n[axis])
		case Neumann:
			plan.eig[axis] = eigenvaluesNeumann(plan.n[axis], plan.h[axis])
			plan.tr[axis], err = newDCTAxisTransform(plan.n[axis])
		}
		if err != nil {
			return nil, fmt.Errorf("axis %d: %w", axis, err)
		}
	}

	plan.work = NewWorkspace(0, size)

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

	hasNullspace := p.hasNullspace()
	if hasNullspace && p.opts.Nullspace == NullspaceError {
		return ErrNullspace
	}

	offset := 0.0
	if hasNullspace {
		mean, maxAbs := meanAndMaxAbs(rhs)
		if p.opts.Nullspace == NullspaceZeroMode && !meanWithinTolerance(mean, maxAbs) {
			return ErrNonZeroMean
		}

		if p.opts.Nullspace == NullspaceSubtractMean {
			offset = mean
		}
	}

	for i, v := range rhs {
		p.work.Complex[i] = complex(v-offset, 0)
	}

	shape := p.shape()
	for axis := 0; axis < p.dim; axis++ {
		if err := p.tr[axis].Forward(p.work.Complex, shape, axis); err != nil {
			return fmt.Errorf("forward axis %d: %w", axis, err)
		}
	}

	if err := p.applyEigenvalues(); err != nil {
		return err
	}

	for axis := p.dim - 1; axis >= 0; axis-- {
		if err := p.tr[axis].Inverse(p.work.Complex, shape, axis); err != nil {
			return fmt.Errorf("inverse axis %d: %w", axis, err)
		}
	}

	addMean := 0.0
	if hasNullspace && p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range p.work.Complex {
		dst[i] = real(p.work.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

func (p *Plan) shape() grid.Shape {
	return grid.Shape{p.n[0], p.n[1], p.n[2]}
}

func (p *Plan) size() int {
	size := 1
	for axis := 0; axis < p.dim; axis++ {
		size *= p.n[axis]
	}
	return size
}

func (p *Plan) hasNullspace() bool {
	if p.alpha != 0 {
		return false
	}

	for axis := 0; axis < p.dim; axis++ {
		if !p.bc[axis].HasNullspace() {
			return false
		}
	}
	return true
}

func (p *Plan) applyEigenvalues() error {
	dims := make([]int, p.dim)
	for axis := 0; axis < p.dim; axis++ {
		dims[axis] = p.n[axis]
	}

	indices := make([]int, p.dim)
	allowZeroMode := p.hasNullspace()
	for idx := range p.work.Complex {
		denom := p.alpha
		for axis := 0; axis < p.dim; axis++ {
			denom += p.eig[axis][indices[axis]]
		}

		if denom == 0 {
			if allowZeroMode && isZeroMode(indices) {
				p.work.Complex[idx] = 0
			} else {
				return ErrResonant
			}
		} else {
			p.work.Complex[idx] /= complex(denom, 0)
		}

		for axis := p.dim - 1; axis >= 0; axis-- {
			indices[axis]++
			if indices[axis] < dims[axis] {
				break
			}
			indices[axis] = 0
		}
	}

	return nil
}

func isZeroMode(indices []int) bool {
	for _, idx := range indices {
		if idx != 0 {
			return false
		}
	}
	return true
}
