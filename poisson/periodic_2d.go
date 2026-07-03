package poisson

import (
	"fmt"
	"log"

	algofft "github.com/cwbudde/algo-fft"
	"github.com/MeKo-Tech/algo-pde/grid"
)

// real2DWorkspace bundles a real FFT plan with its buffers for one Solve call.
// PlanReal2D carries mutable internal scratch, so each concurrent Solve needs
// its own instance (algo-fft < v0.6.13 additionally had a Clone that shared a
// stateful row plan; full construction sidesteps that entirely).
type real2DWorkspace struct {
	rfft  *algofft.PlanReal2D
	rbuf  []float32
	rspec []complex64
}

// Plan2DPeriodic is a reusable plan for solving 2D periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing hx, hy.
type Plan2DPeriodic struct {
	nx, ny int
	hx, hy float64
	eigX   []float64
	eigY   []float64
	fftX   *FFTPlan
	fftY   *FFTPlan
	work   *workspacePool
	rpool  *residentPool[real2DWorkspace]
	rhalf  int
	useR   bool
	opts   Options
	shape  grid.Shape
}

// NewPlan2DPeriodic creates a new 2D periodic Poisson plan.
func NewPlan2DPeriodic(nx, ny int, hx, hy float64, opts ...Option) (*Plan2DPeriodic, error) {
	if nx < 1 || ny < 1 {
		return nil, ErrInvalidSize
	}

	if !validSpacing(hx) || !validSpacing(hy) {
		return nil, ErrInvalidSpacing
	}

	options := ApplyOptions(DefaultOptions(), opts)
	options.Workers = effectiveWorkers(options.Workers)

	// A periodic problem always carries the constant nullspace, so
	// NullspaceError can never yield a usable Solve. Reject it up front.
	if options.Nullspace == NullspaceError {
		return nil, ErrNullspace
	}

	plan := &Plan2DPeriodic{
		nx:    nx,
		ny:    ny,
		hx:    hx,
		hy:    hy,
		eigX:  eigenvaluesPeriodic(nx, hx),
		eigY:  eigenvaluesPeriodic(ny, hy),
		work:  newWorkspacePool(0, nx*ny),
		opts:  options,
		shape: grid.NewShape2D(nx, ny),
	}

	if options.UseRealFFT {
		if ny%2 != 0 || ny < 2 || !isPowerOfTwo(nx) || !isPowerOfTwo(ny) {
			log.Printf("poisson: real FFT disabled for 2D plan (nx=%d, ny=%d): requires even ny and power-of-two sizes", nx, ny)
		} else {
			plan.rhalf = ny/2 + 1
			rws, err := plan.newRealWorkspace()
			if err != nil {
				log.Printf("poisson: real FFT disabled for 2D plan (nx=%d, ny=%d): %v", nx, ny, err)
			} else {
				plan.rpool = newResidentPool[real2DWorkspace](1)
				plan.rpool.put(rws)
				plan.useR = true
			}
		}
	}

	if !plan.useR {
		var err error
		plan.fftX, err = NewFFTPlanWithWorkers(nx, options.Workers)
		if err != nil {
			return nil, err
		}

		plan.fftY, err = NewFFTPlanWithWorkers(ny, options.Workers)
		if err != nil {
			return nil, err
		}
	}

	return plan, nil
}

// Solve computes the solution into dst for a given RHS.
func (p *Plan2DPeriodic) Solve(dst, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	if len(dst) != p.nx*p.ny || len(rhs) != p.nx*p.ny {
		return ErrSizeMismatch
	}

	mean, maxAbs := meanAndMaxAbs(rhs)
	if p.opts.Nullspace == NullspaceZeroMode && !meanWithinTolerance(mean, maxAbs, meanRelTol(min(p.nx, p.ny))) {
		return ErrNonZeroMean
	}

	offset := 0.0
	if p.opts.Nullspace == NullspaceSubtractMean {
		offset = mean
	}

	if p.useR {
		return p.solveReal(dst, rhs, offset)
	}

	workspace := p.work.get()
	defer p.work.put(workspace)

	for i, v := range rhs {
		workspace.Complex[i] = complex(v-offset, 0)
	}

	if err := p.fftX.TransformLines(workspace.Complex, p.shape, 0, false); err != nil {
		return fmt.Errorf("FFT forward axis 0: %w", err)
	}

	if err := p.fftY.TransformLines(workspace.Complex, p.shape, 1, false); err != nil {
		return fmt.Errorf("FFT forward axis 1: %w", err)
	}

	workers := clampWorkers(p.opts.Workers, p.nx)
	if err := parallelFor(workers, p.nx, func(_ int, start, end int) error {
		for i := start; i < end; i++ {
			base := i * p.ny
			for j := 0; j < p.ny; j++ {
				denom := p.eigX[i] + p.eigY[j]
				if denom == 0 {
					workspace.Complex[base+j] = 0
					continue
				}
				workspace.Complex[base+j] /= complex(denom, 0)
			}
		}
		return nil
	}); err != nil {
		return err
	}

	if err := p.fftY.TransformLines(workspace.Complex, p.shape, 1, true); err != nil {
		return fmt.Errorf("FFT inverse axis 1: %w", err)
	}

	if err := p.fftX.TransformLines(workspace.Complex, p.shape, 0, true); err != nil {
		return fmt.Errorf("FFT inverse axis 0: %w", err)
	}

	addMean := 0.0
	if p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range p.nx * p.ny {
		dst[i] = real(workspace.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan2DPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

// UsedRealFFT reports whether the plan runs the single-precision (float32)
// real-FFT path. It is false when WithRealFFT/WithFloat32 was not set or when
// the sizes did not qualify and the plan fell back to the float64 complex FFT.
func (p *Plan2DPeriodic) UsedRealFFT() bool {
	return p.useR
}

func (p *Plan2DPeriodic) newRealWorkspace() (*real2DWorkspace, error) {
	rfft, err := algofft.NewPlanReal2D(p.nx, p.ny)
	if err != nil {
		return nil, err
	}

	return &real2DWorkspace{
		rfft:  rfft,
		rbuf:  make([]float32, p.nx*p.ny),
		rspec: make([]complex64, p.nx*p.rhalf),
	}, nil
}

func (p *Plan2DPeriodic) getRealWorkspace() (*real2DWorkspace, error) {
	if rws := p.rpool.get(); rws != nil {
		return rws, nil
	}
	return p.newRealWorkspace()
}

func (p *Plan2DPeriodic) solveReal(dst, rhs []float64, offset float64) error {
	rws, err := p.getRealWorkspace()
	if err != nil {
		return fmt.Errorf("real FFT workspace: %w", err)
	}
	defer p.rpool.put(rws)

	for i, v := range rhs {
		rws.rbuf[i] = float32(v - offset)
	}

	if err := rws.rfft.Forward(rws.rspec, rws.rbuf); err != nil {
		return fmt.Errorf("real FFT forward: %w", err)
	}

	if err := p.divideRealSpectrum(rws.rspec); err != nil {
		return err
	}

	if err := rws.rfft.Inverse(rws.rbuf, rws.rspec); err != nil {
		return fmt.Errorf("real FFT inverse: %w", err)
	}

	addMean := 0.0
	if p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range p.nx * p.ny {
		dst[i] = float64(rws.rbuf[i]) + addMean
	}

	return nil
}

func (p *Plan2DPeriodic) divideRealSpectrum(rspec []complex64) error {
	workers := clampWorkers(p.opts.Workers, p.nx)
	return parallelFor(workers, p.nx, func(_ int, start, end int) error {
		for i := start; i < end; i++ {
			base := i * p.rhalf
			for j := 0; j < p.rhalf; j++ {
				denom := p.eigX[i] + p.eigY[j]
				if denom == 0 {
					rspec[base+j] = 0
					continue
				}
				rspec[base+j] /= complex(float32(denom), 0)
			}
		}
		return nil
	})
}
