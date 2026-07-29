package poisson

import (
	"context"
	"fmt"

	"github.com/CWBudde/algo-pde/bc"
	"github.com/CWBudde/algo-pde/grid"
)

// Plan1DPeriodic is a reusable plan for solving 1D periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing h.
type Plan1DPeriodic struct {
	n     int
	h     float64
	eig   []float64
	fft   *FFTPlan
	work  *workspacePool
	opts  Options
	shape grid.Shape
}

// NewPlan1DPeriodic creates a new 1D periodic Poisson plan.
func NewPlan1DPeriodic(nx int, hx float64, opts ...Option) (*Plan1DPeriodic, error) {
	if nx < 1 {
		return nil, ErrInvalidSize
	}

	if !validSpacing(hx) {
		return nil, ErrInvalidSpacing
	}

	options := ApplyOptions(DefaultOptions(), opts)
	options.Workers = effectiveWorkers(options.Workers)

	// A periodic problem always carries the constant nullspace, so
	// NullspaceError can never yield a usable Solve. Reject it up front rather
	// than failing on every call.
	if options.Nullspace == NullspaceError {
		return nil, ErrNullspace
	}

	fftPlan, err := NewFFTPlanWithWorkers(nx, options.Workers)
	if err != nil {
		return nil, err
	}

	return &Plan1DPeriodic{
		n:     nx,
		h:     hx,
		eig:   bc.EigenvaluesPeriodic(nx, hx),
		fft:   fftPlan,
		work:  newWorkspacePool(0, nx),
		opts:  options,
		shape: grid.NewShape1D(nx),
	}, nil
}

// Solve computes the solution into dst for a given RHS.
func (p *Plan1DPeriodic) Solve(dst, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	if len(dst) != p.n || len(rhs) != p.n {
		return ErrSizeMismatch
	}

	// Periodic quadrature is spectrally accurate, so a compatible RHS has a
	// mean at roundoff level: gate tightly and keep rejecting real DC offsets.
	mean, maxAbs := meanAndMaxAbs(rhs)
	if p.opts.Nullspace == NullspaceZeroMode && !meanWithinTolerance(mean, maxAbs, meanRoundoffFloor) {
		return ErrNonZeroMean
	}

	offset := 0.0
	if p.opts.Nullspace == NullspaceSubtractMean {
		offset = mean
	}

	workspace := p.work.get()
	defer p.work.put(workspace)

	for i, v := range rhs {
		workspace.Complex[i] = complex(v-offset, 0)
	}

	if err := p.fft.TransformLines(workspace.Complex, p.shape, 0, false); err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	workers := clampWorkers(p.opts.Workers, p.n)
	if err := parallelFor(workers, p.n, func(ctx context.Context, _ int, start, end int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		for i := start; i < end; i++ {
			if p.eig[i] == 0 {
				workspace.Complex[i] = 0
				continue
			}
			workspace.Complex[i] = divByReal(workspace.Complex[i], p.eig[i])
		}
		return nil
	}); err != nil {
		return err
	}

	if err := p.fft.TransformLines(workspace.Complex, p.shape, 0, true); err != nil {
		return fmt.Errorf("FFT inverse: %w", err)
	}

	addMean := 0.0
	if p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range p.n {
		dst[i] = real(workspace.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan1DPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}
