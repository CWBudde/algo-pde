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
	fftPlan  *algofft.Plan[complex128]
	scratchA []complex128
	scratchB []complex128
}

// NewFFTPlan creates a new complex FFT plan for length n.
func NewFFTPlan(n int) (*FFTPlan, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	fftPlan, err := algofft.NewPlan64(n)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	return &FFTPlan{
		n:        n,
		fftPlan:  fftPlan,
		scratchA: make([]complex128, n),
		scratchB: make([]complex128, n),
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
	it := grid.NewLineIterator(shape, axis)
	lineStride := it.LineStride()

	// Process first line (iterator starts at position 0)
	start := it.StartIndex()
	if err := p.transformLine(data, start, lineStride, inverse, useOutOfPlace); err != nil {
		return err
	}

	for it.Next() {
		start = it.StartIndex()
		if err := p.transformLine(data, start, lineStride, inverse, useOutOfPlace); err != nil {
			return err
		}
	}

	return nil
}

func (p *FFTPlan) transformLine(
	data []complex128,
	start int,
	stride int,
	inverse bool,
	useOutOfPlace bool,
) error {
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

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}
