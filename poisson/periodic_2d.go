package poisson

import (
	"fmt"
	"log"

	algofft "github.com/MeKo-Christian/algo-fft"
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
	rfft   *algofft.PlanReal2D
	rbuf   []float32
	rspec  []complex64
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

	if hx <= 0 || hy <= 0 {
		return nil, ErrInvalidSpacing
	}

	options := ApplyOptions(DefaultOptions(), opts)
	options.Workers = effectiveWorkers(options.Workers)

	var (
		fftX  *FFTPlan
		fftY  *FFTPlan
		rfft  *algofft.PlanReal2D
		rbuf  []float32
		rspec []complex64
		rhalf int
		useR  bool
	)

	if options.UseRealFFT {
		if ny%2 != 0 || ny < 2 || !isPowerOfTwo(nx) || !isPowerOfTwo(ny) {
			log.Printf("poisson: real FFT disabled for 2D plan (nx=%d, ny=%d): requires even ny and power-of-two sizes", nx, ny)
		} else {
			plan, err := algofft.NewPlanReal2D(nx, ny)
			if err != nil {
				log.Printf("poisson: real FFT disabled for 2D plan (nx=%d, ny=%d): %v", nx, ny, err)
			} else {
				rfft = plan
				rhalf = ny/2 + 1
				rbuf = make([]float32, nx*ny)
				rspec = make([]complex64, nx*rhalf)
				useR = true
			}
		}
	}

	if !useR {
		var err error
		fftX, err = NewFFTPlanWithWorkers(nx, options.Workers)
		if err != nil {
			return nil, err
		}

		fftY, err = NewFFTPlanWithWorkers(ny, options.Workers)
		if err != nil {
			return nil, err
		}
	}

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
		rfft:  rfft,
		rbuf:  rbuf,
		rspec: rspec,
		rhalf: rhalf,
		useR:  useR,
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

	if p.useR {
		for i, v := range rhs {
			p.rbuf[i] = float32(v - offset)
		}

		if err := p.rfft.Forward(p.rspec, p.rbuf); err != nil {
			return fmt.Errorf("real FFT forward: %w", err)
		}

		workers := clampWorkers(p.opts.Workers, p.nx)
		if err := parallelFor(workers, p.nx, func(_ int, start, end int) error {
			for i := start; i < end; i++ {
				base := i * p.rhalf
				for j := 0; j < p.rhalf; j++ {
					denom := p.eigX[i] + p.eigY[j]
					if denom == 0 {
						p.rspec[base+j] = 0
						continue
					}
					p.rspec[base+j] /= complex(float32(denom), 0)
				}
			}
			return nil
		}); err != nil {
			return err
		}

		if err := p.rfft.Inverse(p.rbuf, p.rspec); err != nil {
			return fmt.Errorf("real FFT inverse: %w", err)
		}

		addMean := 0.0
		if p.opts.SolutionMean != nil {
			addMean = *p.opts.SolutionMean
		}

		for i := range p.nx * p.ny {
			dst[i] = float64(p.rbuf[i]) + addMean
		}

		return nil
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

	workers := clampWorkers(p.opts.Workers, p.nx)
	if err := parallelFor(workers, p.nx, func(_ int, start, end int) error {
		for i := start; i < end; i++ {
			base := i * p.ny
			for j := 0; j < p.ny; j++ {
				denom := p.eigX[i] + p.eigY[j]
				if denom == 0 {
					p.work.Complex[base+j] = 0
					continue
				}
				p.work.Complex[base+j] /= complex(denom, 0)
			}
		}
		return nil
	}); err != nil {
		return err
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
