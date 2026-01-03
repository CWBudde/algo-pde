package poisson

import (
	"fmt"
	"log"

	algofft "github.com/MeKo-Christian/algo-fft"
)

// PlanNDPeriodic is a reusable plan for solving N-dimensional periodic Poisson problems.
// It solves -Δu = f on a periodic grid with spacing h per axis.
type PlanNDPeriodic struct {
	shape  Shape
	h      []float64
	eig    [][]float64
	fft    []*axisPlan
	stride []int
	work   Workspace
	opts   Options

	eigIndices []int
	axisDims   [][]int
	axisIdx    [][]int
	axisOther  [][]int
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

	plans := make([]*axisPlan, len(dims))
	for i, n := range dims {
		plan, err := newAxisPlan(n)
		if err != nil {
			return nil, fmt.Errorf("creating FFT plan for axis %d: %w", i, err)
		}
		plans[i] = plan
	}

	stride := make([]int, len(dims))
	step := 1
	for i := len(dims) - 1; i >= 0; i-- {
		stride[i] = step
		step *= dims[i]
	}

	axisDims := make([][]int, len(dims))
	axisIdx := make([][]int, len(dims))
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
		axisIdx[axis] = make([]int, len(reduced))
		axisOther[axis] = other
	}

	return &PlanNDPeriodic{
		shape:      dims,
		h:          hCopy,
		eig:        eig,
		fft:        plans,
		stride:     stride,
		work:       NewWorkspace(0, dims.Size()),
		opts:       options,
		eigIndices: make([]int, len(dims)),
		axisDims:   axisDims,
		axisIdx:    axisIdx,
		axisOther:  axisOther,
	}, nil
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

	for i, v := range rhs {
		p.work.Complex[i] = complex(v-offset, 0)
	}

	for axis := range p.fft {
		if err := p.transformAxis(axis, false); err != nil {
			return fmt.Errorf("FFT forward axis %d: %w", axis, err)
		}
	}

	p.applyEigenvalues(p.work.Complex)

	for axis := len(p.fft) - 1; axis >= 0; axis-- {
		if err := p.transformAxis(axis, true); err != nil {
			return fmt.Errorf("FFT inverse axis %d: %w", axis, err)
		}
	}

	addMean := 0.0
	if p.opts.SolutionMean != nil {
		addMean = *p.opts.SolutionMean
	}

	for i := range p.work.Complex {
		dst[i] = real(p.work.Complex[i]) + addMean
	}

	return nil
}

// SolveInPlace solves the system in-place, overwriting buf with the solution.
func (p *PlanNDPeriodic) SolveInPlace(buf []float64) error {
	return p.Solve(buf, buf)
}

func (p *PlanNDPeriodic) applyEigenvalues(data []complex128) {
	indices := p.eigIndices
	for i := range indices {
		indices[i] = 0
	}

	for idx := range data {
		denom := 0.0
		for d, eig := range p.eig {
			denom += eig[indices[d]]
		}

		if denom == 0 {
			data[idx] = 0
		} else {
			data[idx] /= complex(denom, 0)
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

func (p *PlanNDPeriodic) transformAxis(axis int, inverse bool) error {
	lineLen := p.shape[axis]
	lineStride := p.stride[axis]
	totalLines := p.shape.Size() / lineLen

	reducedDims := p.axisDims[axis]
	indices := p.axisIdx[axis]
	for i := range indices {
		indices[i] = 0
	}
	otherAxes := p.axisOther[axis]
	for range totalLines {
		start := 0
		for i, d := range otherAxes {
			start += indices[i] * p.stride[d]
		}

		if err := p.fft[axis].transformLine(p.work.Complex, start, lineStride, inverse); err != nil {
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

type axisPlan struct {
	n        int
	fftPlan  *algofft.Plan[complex128]
	scratchA []complex128
	scratchB []complex128
}

func newAxisPlan(n int) (*axisPlan, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	fftPlan, err := algofft.NewPlan64(n)
	if err != nil {
		return nil, err
	}

	return &axisPlan{
		n:        n,
		fftPlan:  fftPlan,
		scratchA: make([]complex128, n),
		scratchB: make([]complex128, n),
	}, nil
}

func (p *axisPlan) transformLine(data []complex128, start int, stride int, inverse bool) error {
	useOutOfPlace := !isPowerOfTwo(p.n)
	if !useOutOfPlace {
		return p.fftPlan.TransformStrided(data[start:], data[start:], stride, inverse)
	}

	if stride == 1 {
		line := data[start : start+p.n]
		var err error
		if inverse {
			err = p.fftPlan.Inverse(p.scratchB, line)
		} else {
			err = p.fftPlan.Forward(p.scratchB, line)
		}
		if err != nil {
			return err
		}
		copy(line, p.scratchB)
		return nil
	}

	for i := 0; i < p.n; i++ {
		p.scratchA[i] = data[start+i*stride]
	}

	var err error
	if inverse {
		err = p.fftPlan.Inverse(p.scratchB, p.scratchA)
	} else {
		err = p.fftPlan.Forward(p.scratchB, p.scratchA)
	}
	if err != nil {
		return err
	}

	for i := 0; i < p.n; i++ {
		data[start+i*stride] = p.scratchB[i]
	}

	return nil
}
