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

// dstWorker bundles a DST plan with line buffers for one goroutine. r2r plans
// carry mutable internal scratch, so concurrent goroutines must not share one.
type dstWorker struct {
	plan    *r2r.DSTPlan
	realBuf []float64
	imagBuf []float64
}

type dstAxisTransform struct {
	n       int
	workers int
	norm    float64
	pool    *residentPool[dstWorker]
}

func newDSTAxisTransform(n int, workers int) (AxisTransform, error) {
	workers = effectiveWorkers(workers)
	transform := &dstAxisTransform{
		n:       n,
		workers: workers,
		pool:    newResidentPool[dstWorker](workers),
	}

	worker, err := transform.newWorker()
	if err != nil {
		return nil, err
	}
	transform.norm = worker.plan.NormalizationFactor()
	transform.pool.put(worker)

	return transform, nil
}

func (t *dstAxisTransform) Forward(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, false)
}

func (t *dstAxisTransform) Inverse(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, true)
}

func (t *dstAxisTransform) Length() int {
	return t.n
}

func (t *dstAxisTransform) NormalizationFactor() float64 {
	return t.norm
}

func (t *dstAxisTransform) newWorker() (*dstWorker, error) {
	plan, err := r2r.NewDSTPlan(t.n)
	if err != nil {
		return nil, err
	}

	return &dstWorker{
		plan:    plan,
		realBuf: make([]float64, t.n),
		imagBuf: make([]float64, t.n),
	}, nil
}

func (t *dstAxisTransform) getWorker() (*dstWorker, error) {
	if worker := t.pool.get(); worker != nil {
		return worker, nil
	}
	return t.newWorker()
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

	if shape.N(axis) != t.n {
		return ErrSizeMismatch
	}

	lineLen := shape.N(axis)
	lineStride := grid.RowMajorStride(shape)[axis]
	numLines := lineCount(shape, axis)
	workers := clampWorkers(t.workers, numLines)

	return parallelFor(workers, numLines, func(_ int, startLine, endLine int) error {
		worker, err := t.getWorker()
		if err != nil {
			return err
		}
		defer t.pool.put(worker)

		for line := startLine; line < endLine; line++ {
			start := lineStartIndex(shape, axis, line)
			if err := transformDSTLine(worker, data, start, lineLen, lineStride, inverse); err != nil {
				return err
			}
		}
		return nil
	})
}

func transformDSTLine(
	w *dstWorker,
	data []complex128,
	start int,
	length int,
	stride int,
	inverse bool,
) error {
	for i := range length {
		v := data[start+i*stride]
		w.realBuf[i] = real(v)
		w.imagBuf[i] = imag(v)
	}

	var err error
	if inverse {
		err = w.plan.Inverse(w.realBuf, w.realBuf)
	} else {
		err = w.plan.Forward(w.realBuf, w.realBuf)
	}
	if err != nil {
		return fmt.Errorf("DST real line: %w", err)
	}

	if inverse {
		err = w.plan.Inverse(w.imagBuf, w.imagBuf)
	} else {
		err = w.plan.Forward(w.imagBuf, w.imagBuf)
	}
	if err != nil {
		return fmt.Errorf("DST imag line: %w", err)
	}

	for i := range length {
		data[start+i*stride] = complex(w.realBuf[i], w.imagBuf[i])
	}

	return nil
}

// dctWorker bundles a DCT-II plan with line buffers for one goroutine.
type dctWorker struct {
	plan    *r2r.DCT2Plan
	realBuf []float64
	imagBuf []float64
}

type dctAxisTransform struct {
	n       int
	workers int
	norm    float64
	pool    *residentPool[dctWorker]
}

func newDCTAxisTransform(n int, workers int) (AxisTransform, error) {
	workers = effectiveWorkers(workers)
	transform := &dctAxisTransform{
		n:       n,
		workers: workers,
		pool:    newResidentPool[dctWorker](workers),
	}

	worker, err := transform.newWorker()
	if err != nil {
		return nil, err
	}
	transform.norm = worker.plan.NormalizationFactor()
	transform.pool.put(worker)

	return transform, nil
}

func (t *dctAxisTransform) Forward(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, false)
}

func (t *dctAxisTransform) Inverse(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, true)
}

func (t *dctAxisTransform) Length() int {
	return t.n
}

func (t *dctAxisTransform) NormalizationFactor() float64 {
	return t.norm
}

func (t *dctAxisTransform) newWorker() (*dctWorker, error) {
	plan, err := r2r.NewDCT2Plan(t.n)
	if err != nil {
		return nil, err
	}

	return &dctWorker{
		plan:    plan,
		realBuf: make([]float64, t.n),
		imagBuf: make([]float64, t.n),
	}, nil
}

func (t *dctAxisTransform) getWorker() (*dctWorker, error) {
	if worker := t.pool.get(); worker != nil {
		return worker, nil
	}
	return t.newWorker()
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

	if shape.N(axis) != t.n {
		return ErrSizeMismatch
	}

	lineLen := shape.N(axis)
	lineStride := grid.RowMajorStride(shape)[axis]
	numLines := lineCount(shape, axis)
	workers := clampWorkers(t.workers, numLines)

	return parallelFor(workers, numLines, func(_ int, startLine, endLine int) error {
		worker, err := t.getWorker()
		if err != nil {
			return err
		}
		defer t.pool.put(worker)

		for line := startLine; line < endLine; line++ {
			start := lineStartIndex(shape, axis, line)
			if err := transformDCTLine(worker, data, start, lineLen, lineStride, inverse); err != nil {
				return err
			}
		}
		return nil
	})
}

func transformDCTLine(
	w *dctWorker,
	data []complex128,
	start int,
	length int,
	stride int,
	inverse bool,
) error {
	for i := range length {
		v := data[start+i*stride]
		w.realBuf[i] = real(v)
		w.imagBuf[i] = imag(v)
	}

	var err error
	if inverse {
		err = w.plan.Inverse(w.realBuf, w.realBuf)
	} else {
		err = w.plan.Forward(w.realBuf, w.realBuf)
	}
	if err != nil {
		return fmt.Errorf("DCT-II real line: %w", err)
	}

	if inverse {
		err = w.plan.Inverse(w.imagBuf, w.imagBuf)
	} else {
		err = w.plan.Forward(w.imagBuf, w.imagBuf)
	}
	if err != nil {
		return fmt.Errorf("DCT-II imag line: %w", err)
	}

	for i := range length {
		data[start+i*stride] = complex(w.realBuf[i], w.imagBuf[i])
	}

	return nil
}
