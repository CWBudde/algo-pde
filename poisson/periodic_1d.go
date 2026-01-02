package poisson

import (
	"fmt"
	"math"

	"github.com/MeKo-Tech/algo-pde/grid"
)

const meanTol = 1e-12

// Plan1DPeriodic is a reusable plan for solving 1D periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing h.
type Plan1DPeriodic struct {
	n     int
	h     float64
	eig   []float64
	fft   *FFTPlan
	work  Workspace
	opts  Options
	shape grid.Shape
}

// NewPlan1DPeriodic creates a new 1D periodic Poisson plan.
func NewPlan1DPeriodic(nx int, hx float64, opts ...Option) (*Plan1DPeriodic, error) {
	if nx < 1 {
		return nil, ErrInvalidSize
	}

	if hx <= 0 {
		return nil, ErrInvalidSpacing
	}

	fftPlan, err := NewFFTPlan(nx)
	if err != nil {
		return nil, err
	}

	options := ApplyOptions(DefaultOptions(), opts)

	return &Plan1DPeriodic{
		n:     nx,
		h:     hx,
		eig:   eigenvaluesPeriodic(nx, hx),
		fft:   fftPlan,
		work:  NewWorkspace(0, nx),
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

	if err := p.fft.TransformLines(p.work.Complex, p.shape, 0, false); err != nil {
		return fmt.Errorf("FFT forward: %w", err)
	}

	for i := range p.n {
		if p.eig[i] == 0 {
			p.work.Complex[i] = 0
			continue
		}

		p.work.Complex[i] /= complex(p.eig[i], 0)
	}

	if err := p.fft.TransformLines(p.work.Complex, p.shape, 0, true); err != nil {
		return fmt.Errorf("FFT inverse: %w", err)
	}

	addMean := 0.0
	if p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range p.n {
		dst[i] = real(p.work.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan1DPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

func meanAndMaxAbs(values []float64) (mean, maxAbs float64) {
	if len(values) == 0 {
		return 0, 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
		abs := math.Abs(v)
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	return sum / float64(len(values)), maxAbs
}

func meanWithinTolerance(mean, maxAbs float64) bool {
	return math.Abs(mean) <= meanTol*(1.0+maxAbs)
}
