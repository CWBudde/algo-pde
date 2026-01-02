package poisson

import (
	"fmt"
	"log"

	algofft "github.com/MeKo-Christian/algo-fft"
	"github.com/MeKo-Tech/algo-pde/grid"
)

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
	work       Workspace
	rfft       *algofft.PlanReal3D
	rbuf       []float32
	rspec      []complex64
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

	if hx <= 0 || hy <= 0 || hz <= 0 {
		return nil, ErrInvalidSpacing
	}

	options := ApplyOptions(DefaultOptions(), opts)

	var (
		fftX  *FFTPlan
		fftY  *FFTPlan
		fftZ  *FFTPlan
		rfft  *algofft.PlanReal3D
		rbuf  []float32
		rspec []complex64
		rhalf int
		useR  bool
	)

	if options.UseRealFFT {
		if nz%2 != 0 || nz < 2 || !isPowerOfTwo(nx) || !isPowerOfTwo(ny) || !isPowerOfTwo(nz) {
			log.Printf("poisson: real FFT disabled for 3D plan (nx=%d, ny=%d, nz=%d): requires even nz and power-of-two sizes", nx, ny, nz)
		} else {
			plan, err := algofft.NewPlanReal3D(nx, ny, nz)
			if err != nil {
				log.Printf("poisson: real FFT disabled for 3D plan (nx=%d, ny=%d, nz=%d): %v", nx, ny, nz, err)
			} else {
				rfft = plan
				rhalf = nz/2 + 1
				rbuf = make([]float32, nx*ny*nz)
				rspec = make([]complex64, nx*ny*rhalf)
				useR = true
			}
		}
	}

	if !useR {
		var err error
		fftX, err = NewFFTPlan(nx)
		if err != nil {
			return nil, err
		}

		fftY, err = NewFFTPlan(ny)
		if err != nil {
			return nil, err
		}

		fftZ, err = NewFFTPlan(nz)
		if err != nil {
			return nil, err
		}
	}

	return &Plan3DPeriodic{
		nx:    nx,
		ny:    ny,
		nz:    nz,
		hx:    hx,
		hy:    hy,
		hz:    hz,
		eigX:  eigenvaluesPeriodic(nx, hx),
		eigY:  eigenvaluesPeriodic(ny, hy),
		eigZ:  eigenvaluesPeriodic(nz, hz),
		fftX:  fftX,
		fftY:  fftY,
		fftZ:  fftZ,
		work:  NewWorkspace(0, nx*ny*nz),
		rfft:  rfft,
		rbuf:  rbuf,
		rspec: rspec,
		rhalf: rhalf,
		useR:  useR,
		opts:  options,
		shape: grid.NewShape3D(nx, ny, nz),
	}, nil
}

// Solve computes the solution into dst for a given RHS.
func (p *Plan3DPeriodic) Solve(dst, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	if len(dst) != p.nx*p.ny*p.nz || len(rhs) != p.nx*p.ny*p.nz {
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

		for i := range p.nx {
			baseXY := i * p.ny * p.rhalf
			for j := range p.ny {
				base := baseXY + j*p.rhalf
				xy := p.eigX[i] + p.eigY[j]
				for k := range p.rhalf {
					denom := xy + p.eigZ[k]
					if denom == 0 {
						p.rspec[base+k] = 0
						continue
					}
					p.rspec[base+k] /= complex(float32(denom), 0)
				}
			}
		}

		if err := p.rfft.Inverse(p.rbuf, p.rspec); err != nil {
			return fmt.Errorf("real FFT inverse: %w", err)
		}

		addMean := 0.0
		if p.opts.SolutionMean != nil {
			addMean = *p.opts.SolutionMean
		}

		for i := range p.nx * p.ny * p.nz {
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

	if err := p.fftZ.TransformLines(p.work.Complex, p.shape, 2, false); err != nil {
		return fmt.Errorf("FFT forward axis 2: %w", err)
	}

	for i := range p.nx {
		baseXY := i * p.ny * p.nz
		for j := range p.ny {
			base := baseXY + j*p.nz
			xy := p.eigX[i] + p.eigY[j]
			for k := range p.nz {
				denom := xy + p.eigZ[k]
				if denom == 0 {
					p.work.Complex[base+k] = 0
					continue
				}
				p.work.Complex[base+k] /= complex(denom, 0)
			}
		}
	}

	if err := p.fftZ.TransformLines(p.work.Complex, p.shape, 2, true); err != nil {
		return fmt.Errorf("FFT inverse axis 2: %w", err)
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

	for i := range p.nx * p.ny * p.nz {
		dst[i] = real(p.work.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *Plan3DPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}
