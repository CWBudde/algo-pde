package poisson

import (
	"fmt"
	"log"

	algofft "github.com/cwbudde/algo-fft"
	"github.com/MeKo-Tech/algo-pde/grid"
)

// real3DWorkspace bundles a real FFT plan with its buffers for one Solve call.
// PlanReal3D carries mutable internal scratch, so each concurrent Solve needs
// its own instance (algo-fft < v0.6.13 additionally had a Clone that shared a
// stateful width plan; full construction sidesteps that entirely).
type real3DWorkspace struct {
	rfft  *algofft.PlanReal3D
	rbuf  []float32
	rspec []complex64
}

// Plan3DPeriodic is a reusable plan for solving 3D periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing hx, hy, hz.
type Plan3DPeriodic struct {
	nx, ny, nz int
	hx, hy, hz float64
	eigX       []float64
	eigY       []float64
	eigZ       []float64
	fftX       *FFTPlan
	fftY       *FFTPlan
	fftZ       *FFTPlan
	work       *workspacePool
	rpool      *residentPool[real3DWorkspace]
	rhalf      int
	useR       bool
	opts       Options
	shape      grid.Shape
}

// NewPlan3DPeriodic creates a new 3D periodic Poisson plan.
func NewPlan3DPeriodic(nx, ny, nz int, hx, hy, hz float64, opts ...Option) (*Plan3DPeriodic, error) {
	if nx < 1 || ny < 1 || nz < 1 {
		return nil, ErrInvalidSize
	}

	if !validSpacing(hx) || !validSpacing(hy) || !validSpacing(hz) {
		return nil, ErrInvalidSpacing
	}

	options := ApplyOptions(DefaultOptions(), opts)
	options.Workers = effectiveWorkers(options.Workers)

	// A periodic problem always carries the constant nullspace, so
	// NullspaceError can never yield a usable Solve. Reject it up front.
	if options.Nullspace == NullspaceError {
		return nil, ErrNullspace
	}

	plan := &Plan3DPeriodic{
		nx:    nx,
		ny:    ny,
		nz:    nz,
		hx:    hx,
		hy:    hy,
		hz:    hz,
		eigX:  eigenvaluesPeriodic(nx, hx),
		eigY:  eigenvaluesPeriodic(ny, hy),
		eigZ:  eigenvaluesPeriodic(nz, hz),
		work:  newWorkspacePool(0, nx*ny*nz),
		opts:  options,
		shape: grid.NewShape3D(nx, ny, nz),
	}

	if options.UseRealFFT {
		if nz%2 != 0 || nz < 2 || !isPowerOfTwo(nx) || !isPowerOfTwo(ny) || !isPowerOfTwo(nz) {
			log.Printf("poisson: real FFT disabled for 3D plan (nx=%d, ny=%d, nz=%d): requires even nz and power-of-two sizes", nx, ny, nz)
		} else {
			plan.rhalf = nz/2 + 1
			rws, err := plan.newRealWorkspace()
			if err != nil {
				log.Printf("poisson: real FFT disabled for 3D plan (nx=%d, ny=%d, nz=%d): %v", nx, ny, nz, err)
			} else {
				plan.rpool = newResidentPool[real3DWorkspace](1)
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

		plan.fftZ, err = NewFFTPlanWithWorkers(nz, options.Workers)
		if err != nil {
			return nil, err
		}
	}

	return plan, nil
}

// Solve computes the solution into dst for a given RHS.
func (p *Plan3DPeriodic) Solve(dst, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	if len(dst) != p.nx*p.ny*p.nz || len(rhs) != p.nx*p.ny*p.nz {
		return ErrSizeMismatch
	}

	mean, maxAbs := meanAndMaxAbs(rhs)
	if p.opts.Nullspace == NullspaceZeroMode && !meanWithinTolerance(mean, maxAbs, meanRelTol(min(p.nx, p.ny, p.nz))) {
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

	if err := p.fftZ.TransformLines(workspace.Complex, p.shape, 2, false); err != nil {
		return fmt.Errorf("FFT forward axis 2: %w", err)
	}

	workers := clampWorkers(p.opts.Workers, p.nx)
	if err := parallelFor(workers, p.nx, func(_ int, start, end int) error {
		for i := start; i < end; i++ {
			baseXY := i * p.ny * p.nz
			for j := 0; j < p.ny; j++ {
				base := baseXY + j*p.nz
				xy := p.eigX[i] + p.eigY[j]
				for k := 0; k < p.nz; k++ {
					denom := xy + p.eigZ[k]
					if denom == 0 {
						workspace.Complex[base+k] = 0
						continue
					}
					workspace.Complex[base+k] /= complex(denom, 0)
				}
			}
		}
		return nil
	}); err != nil {
		return err
	}

	if err := p.fftZ.TransformLines(workspace.Complex, p.shape, 2, true); err != nil {
		return fmt.Errorf("FFT inverse axis 2: %w", err)
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

	for i := range p.nx * p.ny * p.nz {
		dst[i] = real(workspace.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan3DPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

// UsedRealFFT reports whether the plan runs the single-precision (float32)
// real-FFT path. It is false when WithRealFFT/WithFloat32 was not set or when
// the sizes did not qualify and the plan fell back to the float64 complex FFT.
func (p *Plan3DPeriodic) UsedRealFFT() bool {
	return p.useR
}

func (p *Plan3DPeriodic) newRealWorkspace() (*real3DWorkspace, error) {
	rfft, err := algofft.NewPlanReal3D(p.nx, p.ny, p.nz)
	if err != nil {
		return nil, err
	}

	return &real3DWorkspace{
		rfft:  rfft,
		rbuf:  make([]float32, p.nx*p.ny*p.nz),
		rspec: make([]complex64, p.nx*p.ny*p.rhalf),
	}, nil
}

func (p *Plan3DPeriodic) getRealWorkspace() (*real3DWorkspace, error) {
	if rws := p.rpool.get(); rws != nil {
		return rws, nil
	}
	return p.newRealWorkspace()
}

func (p *Plan3DPeriodic) solveReal(dst, rhs []float64, offset float64) error {
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

	for i := range p.nx * p.ny * p.nz {
		dst[i] = float64(rws.rbuf[i]) + addMean
	}

	return nil
}

func (p *Plan3DPeriodic) divideRealSpectrum(rspec []complex64) error {
	workers := clampWorkers(p.opts.Workers, p.nx)
	return parallelFor(workers, p.nx, func(_ int, start, end int) error {
		for i := start; i < end; i++ {
			baseXY := i * p.ny * p.rhalf
			for j := 0; j < p.ny; j++ {
				base := baseXY + j*p.rhalf
				xy := p.eigX[i] + p.eigY[j]
				for k := 0; k < p.rhalf; k++ {
					denom := xy + p.eigZ[k]
					if denom == 0 {
						rspec[base+k] = 0
						continue
					}
					rspec[base+k] /= complex(float32(denom), 0)
				}
			}
		}
		return nil
	})
}
