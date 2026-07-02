package poisson

import (
	"fmt"
	"log"
)

// ndWorkspace holds the per-solve buffers for PlanNDPeriodic: the complex
// working grid plus an index slice reused as the odometer counter by the
// (sequential) eigenvalue and axis-transform loops.
type ndWorkspace struct {
	complexBuf []complex128
	idx        []int
}

// PlanNDPeriodic is a reusable plan for solving N-dimensional periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing h per axis.
type PlanNDPeriodic struct {
	shape  Shape
	h      []float64
	eig    [][]float64
	fft    []*fftWorkerPool
	stride []int
	wsPool *residentPool[ndWorkspace]
	opts   Options

	axisDims  [][]int
	axisOther [][]int
}

// NewPlanNDPeriodic creates a new N-dimensional periodic Poisson plan.
func NewPlanNDPeriodic(shape Shape, h []float64, opts ...Option) (*PlanNDPeriodic, error) {
	if len(shape) == 0 {
		return nil, ErrInvalidSize
	}

	for _, n := range shape {
		if n < 1 {
			return nil, ErrInvalidSize
		}
	}

	if len(h) != len(shape) {
		return nil, &ValidationError{
			Field:   "h",
			Message: "length must match shape dimensions",
		}
	}

	for _, spacing := range h {
		if spacing <= 0 {
			return nil, ErrInvalidSpacing
		}
	}

	options := ApplyOptions(DefaultOptions(), opts)
	if options.UseRealFFT {
		log.Printf("poisson: real FFT disabled for ND plan: not supported for arbitrary dimensions")
	}

	dims := make(Shape, len(shape))
	copy(dims, shape)

	hCopy := make([]float64, len(h))
	copy(hCopy, h)

	eig := make([][]float64, len(dims))
	for i, n := range dims {
		eig[i] = eigenvaluesPeriodic(n, hCopy[i])
	}

	pools := make([]*fftWorkerPool, len(dims))
	for i, n := range dims {
		// The ND line loop is sequential, so one resident worker per axis.
		pool, err := newFFTWorkerPool(n, 1)
		if err != nil {
			return nil, fmt.Errorf("creating FFT plan for axis %d: %w", i, err)
		}
		pools[i] = pool
	}

	stride := make([]int, len(dims))
	step := 1
	for i := len(dims) - 1; i >= 0; i-- {
		stride[i] = step
		step *= dims[i]
	}

	axisDims := make([][]int, len(dims))
	axisOther := make([][]int, len(dims))
	for axis := range dims {
		reduced := make([]int, 0, len(dims)-1)
		other := make([]int, 0, len(dims)-1)
		for d := range dims {
			if d == axis {
				continue
			}
			reduced = append(reduced, dims[d])
			other = append(other, d)
		}
		axisDims[axis] = reduced
		axisOther[axis] = other
	}

	plan := &PlanNDPeriodic{
		shape:     dims,
		h:         hCopy,
		eig:       eig,
		fft:       pools,
		stride:    stride,
		wsPool:    newResidentPool[ndWorkspace](1),
		opts:      options,
		axisDims:  axisDims,
		axisOther: axisOther,
	}
	return plan, nil
}

// Solve computes the solution into dst for a given RHS.
func (p *PlanNDPeriodic) Solve(dst, rhs []float64) error {
	if dst == nil || rhs == nil {
		return ErrNilBuffer
	}

	size := p.shape.Size()
	if len(dst) != size || len(rhs) != size {
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

	workspace := p.getWorkspace()
	defer p.wsPool.put(workspace)

	for i, v := range rhs {
		workspace.complexBuf[i] = complex(v-offset, 0)
	}

	for axis := range p.fft {
		if err := p.transformAxis(axis, false, workspace.complexBuf, workspace.idx); err != nil {
			return fmt.Errorf("FFT forward axis %d: %w", axis, err)
		}
	}

	p.applyEigenvalues(workspace.complexBuf, workspace.idx)

	for axis := len(p.fft) - 1; axis >= 0; axis-- {
		if err := p.transformAxis(axis, true, workspace.complexBuf, workspace.idx); err != nil {
			return fmt.Errorf("FFT inverse axis %d: %w", axis, err)
		}
	}

	addMean := 0.0
	if p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i, v := range workspace.complexBuf {
		dst[i] = real(v) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *PlanNDPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

func (p *PlanNDPeriodic) getWorkspace() *ndWorkspace {
	if workspace := p.wsPool.get(); workspace != nil {
		return workspace
	}
	return &ndWorkspace{
		complexBuf: make([]complex128, p.shape.Size()),
		idx:        make([]int, len(p.shape)),
	}
}

func (p *PlanNDPeriodic) applyEigenvalues(data []complex128, idx []int) {
	indices := idx[:len(p.shape)]
	for i := range indices {
		indices[i] = 0
	}

	for i := range data {
		denom := 0.0
		for d, eig := range p.eig {
			denom += eig[indices[d]]
		}

		if denom == 0 {
			data[i] = 0
		} else {
			data[i] /= complex(denom, 0)
		}

		for d := len(indices) - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < p.shape[d] {
				break
			}
			indices[d] = 0
		}
	}
}

func (p *PlanNDPeriodic) transformAxis(axis int, inverse bool, data []complex128, idx []int) error {
	lineLen := p.shape[axis]
	lineStride := p.stride[axis]
	totalLines := p.shape.Size() / lineLen

	reducedDims := p.axisDims[axis]
	indices := idx[:len(reducedDims)]
	for i := range indices {
		indices[i] = 0
	}
	otherAxes := p.axisOther[axis]

	worker, err := p.fft[axis].get()
	if err != nil {
		return err
	}
	defer p.fft[axis].put(worker)

	for range totalLines {
		start := 0
		for i, d := range otherAxes {
			start += indices[i] * p.stride[d]
		}

		if err := fftTransformLine(worker, lineLen, data, start, lineStride, inverse); err != nil {
			return err
		}

		for i := len(indices) - 1; i >= 0; i-- {
			indices[i]++
			if indices[i] < reducedDims[i] {
				break
			}
			indices[i] = 0
		}
	}

	return nil
}
