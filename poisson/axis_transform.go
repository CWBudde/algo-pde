package poisson

import (
	"fmt"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/r2r"
)

type fftAxisTransform struct {
	plan *FFTPlan
}

func newFFTAxisTransform(n int) (AxisTransform, error) {
	plan, err := NewFFTPlan(n)
	if err != nil {
		return nil, err
	}

	return &fftAxisTransform{plan: plan}, nil
}

func (t *fftAxisTransform) Forward(data []complex128, shape grid.Shape, axis int) error {
	return t.plan.TransformLines(data, shape, axis, false)
}

func (t *fftAxisTransform) Inverse(data []complex128, shape grid.Shape, axis int) error {
	return t.plan.TransformLines(data, shape, axis, true)
}

func (t *fftAxisTransform) Length() int {
	return t.plan.Len()
}

func (t *fftAxisTransform) NormalizationFactor() float64 {
	return 1.0
}

type dstAxisTransform struct {
	plan    *r2r.DSTPlan
	realBuf []float64
	imagBuf []float64
}

func newDSTAxisTransform(n int) (AxisTransform, error) {
	plan, err := r2r.NewDSTPlan(n)
	if err != nil {
		return nil, err
	}

	return &dstAxisTransform{
		plan:    plan,
		realBuf: make([]float64, n),
		imagBuf: make([]float64, n),
	}, nil
}

func (t *dstAxisTransform) Forward(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, false)
}

func (t *dstAxisTransform) Inverse(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, true)
}

func (t *dstAxisTransform) Length() int {
	return t.plan.Len()
}

func (t *dstAxisTransform) NormalizationFactor() float64 {
	return t.plan.NormalizationFactor()
}

func (t *dstAxisTransform) transformLines(
	data []complex128,
	shape grid.Shape,
	axis int,
	inverse bool,
) error {
	if data == nil {
		return ErrNilBuffer
	}

	if len(data) != shape.Size() {
		return ErrSizeMismatch
	}

	if shape.N(axis) != t.plan.Len() {
		return ErrSizeMismatch
	}

	it := grid.NewLineIterator(shape, axis)
	lineLen := it.LineLength()
	lineStride := it.LineStride()

	for {
		start := it.StartIndex()
		if err := t.transformLine(data, start, lineLen, lineStride, inverse); err != nil {
			return err
		}

		if !it.Next() {
			break
		}
	}

	return nil
}

func (t *dstAxisTransform) transformLine(
	data []complex128,
	start int,
	length int,
	stride int,
	inverse bool,
) error {
	for i := 0; i < length; i++ {
		v := data[start+i*stride]
		t.realBuf[i] = real(v)
		t.imagBuf[i] = imag(v)
	}

	var err error
	if inverse {
		err = t.plan.Inverse(t.realBuf, t.realBuf)
	} else {
		err = t.plan.Forward(t.realBuf, t.realBuf)
	}
	if err != nil {
		return fmt.Errorf("DST real line: %w", err)
	}

	if inverse {
		err = t.plan.Inverse(t.imagBuf, t.imagBuf)
	} else {
		err = t.plan.Forward(t.imagBuf, t.imagBuf)
	}
	if err != nil {
		return fmt.Errorf("DST imag line: %w", err)
	}

	for i := 0; i < length; i++ {
		data[start+i*stride] = complex(t.realBuf[i], t.imagBuf[i])
	}

	return nil
}

type dctAxisTransform struct {
	plan    *r2r.DCT2Plan
	realBuf []float64
	imagBuf []float64
}

func newDCTAxisTransform(n int) (AxisTransform, error) {
	plan, err := r2r.NewDCT2Plan(n)
	if err != nil {
		return nil, err
	}

	return &dctAxisTransform{
		plan:    plan,
		realBuf: make([]float64, n),
		imagBuf: make([]float64, n),
	}, nil
}

func (t *dctAxisTransform) Forward(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, false)
}

func (t *dctAxisTransform) Inverse(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, true)
}

func (t *dctAxisTransform) Length() int {
	return t.plan.Len()
}

func (t *dctAxisTransform) NormalizationFactor() float64 {
	return t.plan.NormalizationFactor()
}

func (t *dctAxisTransform) transformLines(
	data []complex128,
	shape grid.Shape,
	axis int,
	inverse bool,
) error {
	if data == nil {
		return ErrNilBuffer
	}

	if len(data) != shape.Size() {
		return ErrSizeMismatch
	}

	if shape.N(axis) != t.plan.Len() {
		return ErrSizeMismatch
	}

	it := grid.NewLineIterator(shape, axis)
	lineLen := it.LineLength()
	lineStride := it.LineStride()

	for {
		start := it.StartIndex()
		if err := t.transformLine(data, start, lineLen, lineStride, inverse); err != nil {
			return err
		}

		if !it.Next() {
			break
		}
	}

	return nil
}

func (t *dctAxisTransform) transformLine(
	data []complex128,
	start int,
	length int,
	stride int,
	inverse bool,
) error {
	for i := 0; i < length; i++ {
		v := data[start+i*stride]
		t.realBuf[i] = real(v)
		t.imagBuf[i] = imag(v)
	}

	var err error
	if inverse {
		err = t.plan.Inverse(t.realBuf, t.realBuf)
	} else {
		err = t.plan.Forward(t.realBuf, t.realBuf)
	}
	if err != nil {
		return fmt.Errorf("DCT-II real line: %w", err)
	}

	if inverse {
		err = t.plan.Inverse(t.imagBuf, t.imagBuf)
	} else {
		err = t.plan.Forward(t.imagBuf, t.imagBuf)
	}
	if err != nil {
		return fmt.Errorf("DCT-II imag line: %w", err)
	}

	for i := 0; i < length; i++ {
		data[start+i*stride] = complex(t.realBuf[i], t.imagBuf[i])
	}

	return nil
}
