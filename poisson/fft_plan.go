package poisson

import (
	"fmt"

	algofft "github.com/MeKo-Christian/algo-fft"
	"github.com/MeKo-Tech/algo-pde/grid"
)

// FFTPlan wraps an algo-fft complex FFT plan for periodic (Fourier) transforms.
//
// It provides a convenience method to apply the 1D FFT along all lines of an
// N-dimensional grid stored in row-major order.
type FFTPlan struct {
	n        int
	workers  int
	plans    []*algofft.Plan[complex128]
	scratchA [][]complex128
	scratchB [][]complex128
}

// NewFFTPlan creates a new complex FFT plan for length n.
func NewFFTPlan(n int) (*FFTPlan, error) {
	return NewFFTPlanWithWorkers(n, 1)
}

// NewFFTPlanWithWorkers creates a new FFT plan with the requested worker count.
// workers <= 0 defaults to runtime.GOMAXPROCS.
func NewFFTPlanWithWorkers(n int, workers int) (*FFTPlan, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	workers = effectiveWorkers(workers)
	fftPlan, err := algofft.NewPlan64(n)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	plans := make([]*algofft.Plan[complex128], workers)
	plans[0] = fftPlan
	for i := 1; i < workers; i++ {
		plan, err := algofft.NewPlan64(n)
		if err != nil {
			return nil, fmt.Errorf("creating FFT plan: %w", err)
		}
		plans[i] = plan
	}

	scratchA := make([][]complex128, workers)
	scratchB := make([][]complex128, workers)
	for i := 0; i < workers; i++ {
		scratchA[i] = make([]complex128, n)
		scratchB[i] = make([]complex128, n)
	}

	return &FFTPlan{
		n:        n,
		workers:  workers,
		plans:    plans,
		scratchA: scratchA,
		scratchB: scratchB,
	}, nil
}

// Len returns the transform length.
func (p *FFTPlan) Len() int {
	return p.n
}

// TransformLines applies a forward or inverse FFT along all lines parallel to
// the given axis.
//
// data is modified in-place.
//
// For axis-wise transforms, this method relies on algo-fft's strided transform
// support and does not allocate.
func (p *FFTPlan) TransformLines(data []complex128, shape grid.Shape, axis int, inverse bool) error {
	if data == nil {
		return ErrNilBuffer
	}

	if len(data) != shape.Size() {
		return ErrSizeMismatch
	}

	if shape.N(axis) != p.n {
		return ErrSizeMismatch
	}

	useOutOfPlace := !isPowerOfTwo(p.n)
	lineStride := grid.RowMajorStride(shape)[axis]
	numLines := lineCount(shape, axis)
	workers := clampWorkers(p.workers, numLines)

	return parallelFor(workers, numLines, func(worker, startLine, endLine int) error {
		plan := p.plans[worker]
		scratchA := p.scratchA[worker]
		scratchB := p.scratchB[worker]
		for line := startLine; line < endLine; line++ {
			start := lineStartIndex(shape, axis, line)
			if err := p.transformLine(plan, scratchA, scratchB, data, start, lineStride, inverse, useOutOfPlace); err != nil {
				return err
			}
		}
		return nil
	})
}

func (p *FFTPlan) transformLine(
	plan *algofft.Plan[complex128],
	scratchA []complex128,
	scratchB []complex128,
	data []complex128,
	start int,
	stride int,
	inverse bool,
	useOutOfPlace bool,
) error {
	if !useOutOfPlace {
		return plan.TransformStrided(data[start:], data[start:], stride, inverse)
	}

	if stride == 1 {
		line := data[start : start+p.n]
		var err error
		if inverse {
			err = plan.Inverse(scratchB, line)
		} else {
			err = plan.Forward(scratchB, line)
		}
		if err != nil {
			return err
		}
		copy(line, scratchB)
		return nil
	}

	for i := 0; i < p.n; i++ {
		scratchA[i] = data[start+i*stride]
	}

	var err error
	if inverse {
		err = plan.Inverse(scratchB, scratchA)
	} else {
		err = plan.Forward(scratchB, scratchA)
	}
	if err != nil {
		return err
	}

	for i := 0; i < p.n; i++ {
		data[start+i*stride] = scratchB[i]
	}

	return nil
}

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}
