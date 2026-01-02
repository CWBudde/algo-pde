package poisson

import (
	"fmt"

	"github.com/MeKo-Tech/algo-pde/grid"
)

// Plan2DPeriodic is a reusable plan for solving 2D periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing hx, hy.
type Plan2DPeriodic struct {
	nx, ny int
	hx, hy float64
	eigX   []float64
	eigY   []float64
	fftX   *FFTPlan
	fftY   *FFTPlan
	work   Workspace
	opts   Options
	shape  grid.Shape
}

// NewPlan2DPeriodic creates a new 2D periodic Poisson plan.
func NewPlan2DPeriodic(nx, ny int, hx, hy float64, opts ...Option) (*Plan2DPeriodic, error) {
	if nx < 1 || ny < 1 {
		return nil, ErrInvalidSize
	}

	if hx <= 0 || hy <= 0 {
		return nil, ErrInvalidSpacing
	}

	fftX, err := NewFFTPlan(nx)
	if err != nil {
		return nil, err
	}

	fftY, err := NewFFTPlan(ny)
	if err != nil {
		return nil, err
	}

	options := ApplyOptions(DefaultOptions(), opts)

	return &Plan2DPeriodic{
		nx:    nx,
		ny:    ny,
		hx:    hx,
		hy:    hy,
		eigX:  eigenvaluesPeriodic(nx, hx),
		eigY:  eigenvaluesPeriodic(ny, hy),
		fftX:  fftX,
		fftY:  fftY,
		work:  NewWorkspace(0, nx*ny),
		opts:  options,
		shape: grid.NewShape2D(nx, ny),
	}, nil
}

// Solve computes the solution into dst for a given RHS.
func (p *Plan2DPeriodic) Solve(dst, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	if len(dst) != p.nx*p.ny || len(rhs) != p.nx*p.ny {
		return ErrSizeMismatch
	}

	if p.opts.Nullspace == NullspaceError {
		return ErrNullspace
	}

	mean, maxAbs := meanAndMaxAbs(rhs)
	if p.opts.Nullspace == NullspaceZeroMode && !meanWithinTolerance(mean, maxAbs) {
		return ErrNonZeroMean
	}

	offset := 0.0
	if p.opts.Nullspace == NullspaceSubtractMean {
		offset = mean
	}

	for i, v := range rhs {
		p.work.Complex[i] = complex(v-offset, 0)
	}

	if err := p.fftX.TransformLines(p.work.Complex, p.shape, 0, false); err != nil {
		return fmt.Errorf("FFT forward axis 0: %w", err)
	}

	if err := p.fftY.TransformLines(p.work.Complex, p.shape, 1, false); err != nil {
		return fmt.Errorf("FFT forward axis 1: %w", err)
	}

	for i := range p.nx {
		base := i * p.ny
		for j := range p.ny {
			denom := p.eigX[i] + p.eigY[j]
			if denom == 0 {
				p.work.Complex[base+j] = 0
				continue
			}

			p.work.Complex[base+j] /= complex(denom, 0)
		}
	}

	if err := p.fftY.TransformLines(p.work.Complex, p.shape, 1, true); err != nil {
		return fmt.Errorf("FFT inverse axis 1: %w", err)
	}

	if err := p.fftX.TransformLines(p.work.Complex, p.shape, 0, true); err != nil {
		return fmt.Errorf("FFT inverse axis 0: %w", err)
	}

	addMean := 0.0
	if p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range p.nx * p.ny {
		dst[i] = real(p.work.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan2DPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}
