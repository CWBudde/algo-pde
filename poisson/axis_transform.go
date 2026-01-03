package poisson

import (
	"fmt"

	"github.com/MeKo-Tech/algo-pde/grid"
	"github.com/MeKo-Tech/algo-pde/r2r"
)

type fftAxisTransform struct {
	plan *FFTPlan
}

func newFFTAxisTransform(n int, workers int) (AxisTransform, error) {
	plan, err := NewFFTPlanWithWorkers(n, workers)
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
	plan     *r2r.DSTPlan
	realBuf  []float64
	imagBuf  []float64
	workers  int
	plans    []*r2r.DSTPlan
	realBufs [][]float64
	imagBufs [][]float64
}

func newDSTAxisTransform(n int, workers int) (AxisTransform, error) {
	plan, err := r2r.NewDSTPlan(n)
	if err != nil {
		return nil, err
	}

	workers = effectiveWorkers(workers)
	transform := &dstAxisTransform{
		plan:    plan,
		realBuf: make([]float64, n),
		imagBuf: make([]float64, n),
		workers: workers,
	}

	if workers == 1 {
		return transform, nil
	}

	plans := make([]*r2r.DSTPlan, workers)
	realBufs := make([][]float64, workers)
	imagBufs := make([][]float64, workers)
	plans[0] = plan
	realBufs[0] = transform.realBuf
	imagBufs[0] = transform.imagBuf
	for i := 1; i < workers; i++ {
		clone, err := r2r.NewDSTPlan(n)
		if err != nil {
			return nil, err
		}
		plans[i] = clone
		realBufs[i] = make([]float64, n)
		imagBufs[i] = make([]float64, n)
	}
	transform.plans = plans
	transform.realBufs = realBufs
	transform.imagBufs = imagBufs

	return transform, nil
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

	lineLen := shape.N(axis)
	lineStride := grid.RowMajorStride(shape)[axis]
	numLines := lineCount(shape, axis)
	workers := clampWorkers(t.workers, numLines)

	return parallelFor(workers, numLines, func(worker, startLine, endLine int) error {
		plan := t.plan
		realBuf := t.realBuf
		imagBuf := t.imagBuf
		if workers > 1 {
			plan = t.plans[worker]
			realBuf = t.realBufs[worker]
			imagBuf = t.imagBufs[worker]
		}

		for line := startLine; line < endLine; line++ {
			start := lineStartIndex(shape, axis, line)
			if err := t.transformLine(plan, realBuf, imagBuf, data, start, lineLen, lineStride, inverse); err != nil {
				return err
			}
		}
		return nil
	})
}

func (t *dstAxisTransform) transformLine(
	plan *r2r.DSTPlan,
	realBuf []float64,
	imagBuf []float64,
	data []complex128,
	start int,
	length int,
	stride int,
	inverse bool,
) error {
	for i := 0; i < length; i++ {
		v := data[start+i*stride]
		realBuf[i] = real(v)
		imagBuf[i] = imag(v)
	}

	var err error
	if inverse {
		err = plan.Inverse(realBuf, realBuf)
	} else {
		err = plan.Forward(realBuf, realBuf)
	}
	if err != nil {
		return fmt.Errorf("DST real line: %w", err)
	}

	if inverse {
		err = plan.Inverse(imagBuf, imagBuf)
	} else {
		err = plan.Forward(imagBuf, imagBuf)
	}
	if err != nil {
		return fmt.Errorf("DST imag line: %w", err)
	}

	for i := 0; i < length; i++ {
		data[start+i*stride] = complex(realBuf[i], imagBuf[i])
	}

	return nil
}

type dctAxisTransform struct {
	plan     *r2r.DCT2Plan
	realBuf  []float64
	imagBuf  []float64
	workers  int
	plans    []*r2r.DCT2Plan
	realBufs [][]float64
	imagBufs [][]float64
}

func newDCTAxisTransform(n int, workers int) (AxisTransform, error) {
	plan, err := r2r.NewDCT2Plan(n)
	if err != nil {
		return nil, err
	}

	workers = effectiveWorkers(workers)
	transform := &dctAxisTransform{
		plan:    plan,
		realBuf: make([]float64, n),
		imagBuf: make([]float64, n),
		workers: workers,
	}

	if workers == 1 {
		return transform, nil
	}

	plans := make([]*r2r.DCT2Plan, workers)
	realBufs := make([][]float64, workers)
	imagBufs := make([][]float64, workers)
	plans[0] = plan
	realBufs[0] = transform.realBuf
	imagBufs[0] = transform.imagBuf
	for i := 1; i < workers; i++ {
		clone, err := r2r.NewDCT2Plan(n)
		if err != nil {
			return nil, err
		}
		plans[i] = clone
		realBufs[i] = make([]float64, n)
		imagBufs[i] = make([]float64, n)
	}
	transform.plans = plans
	transform.realBufs = realBufs
	transform.imagBufs = imagBufs

	return transform, nil
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

	lineLen := shape.N(axis)
	lineStride := grid.RowMajorStride(shape)[axis]
	numLines := lineCount(shape, axis)
	workers := clampWorkers(t.workers, numLines)

	return parallelFor(workers, numLines, func(worker, startLine, endLine int) error {
		plan := t.plan
		realBuf := t.realBuf
		imagBuf := t.imagBuf
		if workers > 1 {
			plan = t.plans[worker]
			realBuf = t.realBufs[worker]
			imagBuf = t.imagBufs[worker]
		}

		for line := startLine; line < endLine; line++ {
			start := lineStartIndex(shape, axis, line)
			if err := t.transformLine(plan, realBuf, imagBuf, data, start, lineLen, lineStride, inverse); err != nil {
				return err
			}
		}
		return nil
	})
}

func (t *dctAxisTransform) transformLine(
	plan *r2r.DCT2Plan,
	realBuf []float64,
	imagBuf []float64,
	data []complex128,
	start int,
	length int,
	stride int,
	inverse bool,
) error {
	for i := 0; i < length; i++ {
		v := data[start+i*stride]
		realBuf[i] = real(v)
		imagBuf[i] = imag(v)
	}

	var err error
	if inverse {
		err = plan.Inverse(realBuf, realBuf)
	} else {
		err = plan.Forward(realBuf, realBuf)
	}
	if err != nil {
		return fmt.Errorf("DCT-II real line: %w", err)
	}

	if inverse {
		err = plan.Inverse(imagBuf, imagBuf)
	} else {
		err = plan.Forward(imagBuf, imagBuf)
	}
	if err != nil {
		return fmt.Errorf("DCT-II imag line: %w", err)
	}

	for i := 0; i < length; i++ {
		data[start+i*stride] = complex(realBuf[i], imagBuf[i])
	}

	return nil
}
