package poisson

import (
	"context"
	"fmt"

	"github.com/MeKo-Tech/algo-pde/bc"
	"github.com/MeKo-Tech/algo-pde/grid"
)

// ndWorkspace holds the per-solve buffers for PlanNDPeriodic: the complex
// working grid plus one reusable multi-index scratch buffer per worker. The
// eigenvalue and axis-transform loops derive their per-line multi-indices into
// idx[worker] so parallel workers never share state and Solve stays
// allocation-free (the workspace itself is drawn from a pool per call).
type ndWorkspace struct {
	complexBuf []complex128
	idx        [][]int
}

// PlanNDPeriodic is a reusable plan for solving N-dimensional periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing h per axis.
type PlanNDPeriodic struct {
	shape  grid.Shape
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
func NewPlanNDPeriodic(shape grid.Shape, h []float64, opts ...Option) (*PlanNDPeriodic, error) {
	if shape.Dim() == 0 {
		return nil, ErrInvalidSize
	}

	for _, n := range shape.Dims() {
		if n < 1 {
			return nil, ErrInvalidSize
		}
	}

	if len(h) != shape.Dim() {
		return nil, &ValidationError{
			Field:   "h",
			Message: "length must match shape dimensions",
		}
	}

	for _, spacing := range h {
		if !validSpacing(spacing) {
			return nil, ErrInvalidSpacing
		}
	}

	options := ApplyOptions(DefaultOptions(), opts)
	options.Workers = effectiveWorkers(options.Workers)

	// A periodic problem always carries the constant nullspace, so
	// NullspaceError can never yield a usable Solve. Reject it up front.
	if options.Nullspace == NullspaceError {
		return nil, ErrNullspace
	}

	// Real FFT is not supported for arbitrary dimensions; the plan runs the
	// float64 complex path. UsedRealFFT reports this (always false for ND).

	// dims is read-only during construction; grid.NewShapeN below makes the one
	// defensive copy that the plan retains, so no separate copy is needed here.
	dims := shape.Dims()

	hCopy := make([]float64, len(h))
	copy(hCopy, h)

	eig := make([][]float64, len(dims))
	for i, n := range dims {
		eig[i] = bc.EigenvaluesPeriodic(n, hCopy[i])
	}

	pools := make([]*fftWorkerPool, len(dims))
	for i, n := range dims {
		// The line loop runs across options.Workers goroutines, so keep that
		// many resident FFT workers per axis.
		pool, err := newFFTWorkerPool(n, options.Workers)
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
		shape:     grid.NewShapeN(dims),
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

	workspace := p.getWorkspace()
	defer p.wsPool.put(workspace)

	for i, v := range rhs {
		workspace.complexBuf[i] = complex(v-offset, 0)
	}

	for axis := range p.fft {
		if err := p.transformAxis(axis, false, workspace); err != nil {
			return fmt.Errorf("FFT forward axis %d: %w", axis, err)
		}
	}

	if err := p.applyEigenvalues(workspace); err != nil {
		return err
	}

	for axis := len(p.fft) - 1; axis >= 0; axis-- {
		if err := p.transformAxis(axis, true, workspace); err != nil {
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
// PlanNDPeriodic reads the whole RHS into an internal complex workspace before
// writing any output, so passing the same slice as dst and rhs (as Solve does
// here) is always safe regardless of the WithInPlace option.
func (p *PlanNDPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

// UsedRealFFT reports whether the plan runs the single-precision real-FFT path.
// PlanNDPeriodic never uses real FFT (it is not supported for arbitrary
// dimensions), so this is always false.
func (p *PlanNDPeriodic) UsedRealFFT() bool {
	return false
}

func (p *PlanNDPeriodic) getWorkspace() *ndWorkspace {
	if workspace := p.wsPool.get(); workspace != nil {
		return workspace
	}
	// One index buffer per potential worker (opts.Workers is the clamp ceiling),
	// each long enough for the full multi-index; the axis loops slice it down.
	idx := make([][]int, p.opts.Workers)
	for w := range idx {
		idx[w] = make([]int, p.shape.Dim())
	}
	return &ndWorkspace{
		complexBuf: make([]complex128, p.shape.Size()),
		idx:        idx,
	}
}

// ndMultiIndex fills indices with the mixed-radix decomposition of the flat
// index flat over the given radices (rightmost varying fastest).
func ndMultiIndex(indices, radices []int, flat int) {
	for d := len(radices) - 1; d >= 0; d-- {
		indices[d] = flat % radices[d]
		flat /= radices[d]
	}
}

// ndIncrement advances the mixed-radix odometer indices by one step.
func ndIncrement(indices, radices []int) {
	for d := len(indices) - 1; d >= 0; d-- {
		indices[d]++
		if indices[d] < radices[d] {
			break
		}
		indices[d] = 0
	}
}

func (p *PlanNDPeriodic) applyEigenvalues(ws *ndWorkspace) error {
	data := ws.complexBuf
	size := p.shape.Size()
	workers := clampWorkers(p.opts.Workers, size)

	return parallelFor(workers, size, func(ctx context.Context, worker, start, end int) error {
		if err := ctx.Err(); err != nil {
			return err
		}

		radices := p.shape.Dims()
		indices := ws.idx[worker][:len(radices)]
		ndMultiIndex(indices, radices, start)

		for i := start; i < end; i++ {
			denom := 0.0
			for d, eig := range p.eig {
				denom += eig[indices[d]]
			}

			if denom == 0 {
				data[i] = 0
			} else {
				data[i] /= complex(denom, 0)
			}

			ndIncrement(indices, radices)
		}
		return nil
	})
}

func (p *PlanNDPeriodic) transformAxis(axis int, inverse bool, ws *ndWorkspace) error {
	data := ws.complexBuf
	lineLen := p.shape.N(axis)
	lineStride := p.stride[axis]
	totalLines := p.shape.Size() / lineLen

	reducedDims := p.axisDims[axis]
	otherAxes := p.axisOther[axis]
	workers := clampWorkers(p.opts.Workers, totalLines)

	return parallelFor(workers, totalLines, func(ctx context.Context, workerIdx, startLine, endLine int) error {
		worker, err := p.fft[axis].get()
		if err != nil {
			return err
		}
		defer p.fft[axis].put(worker)

		indices := ws.idx[workerIdx][:len(reducedDims)]
		ndMultiIndex(indices, reducedDims, startLine)

		for line := startLine; line < endLine; line++ {
			if err := ctx.Err(); err != nil {
				return err
			}

			start := 0
			for i, d := range otherAxes {
				start += indices[i] * p.stride[d]
			}

			if err := fftTransformLine(worker, lineLen, data, start, lineStride, inverse); err != nil {
				return err
			}

			ndIncrement(indices, reducedDims)
		}

		return nil
	})
}
