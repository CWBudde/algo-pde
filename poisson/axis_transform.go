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

// realLinePlan is the subset of the r2r type-I/II plan API used to transform a
// single real line in place. Both *r2r.DSTPlan (DST-I) and *r2r.DCT2Plan
// (DCT-II) satisfy it, letting DST and DCT axis transforms share one generic
// implementation.
type realLinePlan interface {
	Forward(dst, src []float64) error
	Inverse(dst, src []float64) error
}

// realLineWorker bundles a real-line plan with scratch buffers for one
// goroutine. r2r plans carry mutable internal scratch, so concurrent goroutines
// must not share one.
type realLineWorker[P realLinePlan] struct {
	plan    P
	realBuf []float64
	imagBuf []float64
}

// realAxisTransform adapts a real-valued r2r line plan (DST/DCT) into the
// complex-valued AxisTransform interface by transforming the real and
// imaginary parts of each line independently.
type realAxisTransform[P realLinePlan] struct {
	n       int
	workers int
	label   string
	newPlan func(n int) (P, error)
	pool    *residentPool[realLineWorker[P]]
}

func newRealAxisTransform[P realLinePlan](
	n, workers int,
	label string,
	newPlan func(n int) (P, error),
) (AxisTransform, error) {
	workers = effectiveWorkers(workers)
	transform := &realAxisTransform[P]{
		n:       n,
		workers: workers,
		label:   label,
		newPlan: newPlan,
		pool:    newResidentPool[realLineWorker[P]](workers),
	}

	// Construct one worker up front to surface plan-creation errors at
	// construction time and to prime the pool.
	worker, err := transform.newWorker()
	if err != nil {
		return nil, err
	}
	transform.pool.put(worker)

	return transform, nil
}

func newDSTAxisTransform(n int, workers int) (AxisTransform, error) {
	return newRealAxisTransform(n, workers, "DST", func(n int) (*r2r.DSTPlan, error) {
		return r2r.NewDSTPlan(n)
	})
}

func newDCTAxisTransform(n int, workers int) (AxisTransform, error) {
	return newRealAxisTransform(n, workers, "DCT-II", func(n int) (*r2r.DCT2Plan, error) {
		return r2r.NewDCT2Plan(n)
	})
}

func (t *realAxisTransform[P]) Forward(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, false)
}

func (t *realAxisTransform[P]) Inverse(data []complex128, shape grid.Shape, axis int) error {
	return t.transformLines(data, shape, axis, true)
}

func (t *realAxisTransform[P]) Length() int {
	return t.n
}

func (t *realAxisTransform[P]) newWorker() (*realLineWorker[P], error) {
	plan, err := t.newPlan(t.n)
	if err != nil {
		return nil, err
	}

	return &realLineWorker[P]{
		plan:    plan,
		realBuf: make([]float64, t.n),
		imagBuf: make([]float64, t.n),
	}, nil
}

func (t *realAxisTransform[P]) getWorker() (*realLineWorker[P], error) {
	if worker := t.pool.get(); worker != nil {
		return worker, nil
	}
	return t.newWorker()
}

func (t *realAxisTransform[P]) transformLines(
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
			if err := t.transformLine(worker, data, start, lineLen, lineStride, inverse); err != nil {
				return err
			}
		}
		return nil
	})
}

func (t *realAxisTransform[P]) transformLine(
	w *realLineWorker[P],
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

	if err := t.applyPart(w.plan, w.realBuf, inverse); err != nil {
		return fmt.Errorf("%s real line: %w", t.label, err)
	}

	if err := t.applyPart(w.plan, w.imagBuf, inverse); err != nil {
		return fmt.Errorf("%s imag line: %w", t.label, err)
	}

	for i := range length {
		data[start+i*stride] = complex(w.realBuf[i], w.imagBuf[i])
	}

	return nil
}

func (t *realAxisTransform[P]) applyPart(plan P, buf []float64, inverse bool) error {
	if inverse {
		return plan.Inverse(buf, buf)
	}
	return plan.Forward(buf, buf)
}
