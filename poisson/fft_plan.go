package poisson

import (
	"context"
	"fmt"

	"github.com/cwbudde/algo-pde/grid"
	algofft "github.com/cwbudde/algo-fft"
)

// fftWorker bundles an FFT plan with the scratch buffers one goroutine needs
// to transform lines. algo-fft plans carry mutable internal scratch, so each
// concurrently transforming goroutine must hold its own worker.
type fftWorker struct {
	plan     *algofft.Plan[complex128]
	scratchA []complex128
	scratchB []complex128
}

// fftWorkerPool hands out fftWorkers on demand. It is what makes transforms
// safe for concurrent callers: workers are acquired per goroutine per call
// instead of being indexed by worker slot, so two concurrent Solve calls can
// never share scratch state. The steady state keeps workers cached in the
// pool; a burst of concurrency transiently constructs extras.
type fftWorkerPool struct {
	n    int
	pool *residentPool[fftWorker]
}

// newFFTWorkerPool creates a worker pool for transforms of length n that keeps
// up to capacity workers resident (one per intra-solve parallel worker).
func newFFTWorkerPool(n int, capacity int) (*fftWorkerPool, error) {
	if n < 1 {
		return nil, ErrInvalidSize
	}

	workerPool := &fftWorkerPool{n: n, pool: newResidentPool[fftWorker](capacity)}
	worker, err := workerPool.newWorker()
	if err != nil {
		return nil, err
	}
	workerPool.pool.put(worker)

	return workerPool, nil
}

// newWorker builds a full plan per pool entry. algo-fft's Plan.Clone (safe
// since v0.6.13; earlier versions left stridedScratch unallocated) could
// share twiddle tables here as a cheaper alternative; full construction is
// kept for simplicity.
func (p *fftWorkerPool) newWorker() (*fftWorker, error) {
	plan, err := algofft.NewPlan64(p.n)
	if err != nil {
		return nil, fmt.Errorf("creating FFT plan: %w", err)
	}

	return &fftWorker{
		plan:     plan,
		scratchA: make([]complex128, p.n),
		scratchB: make([]complex128, p.n),
	}, nil
}

func (p *fftWorkerPool) get() (*fftWorker, error) {
	if worker := p.pool.get(); worker != nil {
		return worker, nil
	}
	return p.newWorker()
}

func (p *fftWorkerPool) put(worker *fftWorker) {
	p.pool.put(worker)
}

// fftTransformLine applies a forward or inverse FFT to one strided line using
// the given worker's plan and scratch.
func fftTransformLine(w *fftWorker, n int, data []complex128, start, stride int, inverse bool) error {
	if isPowerOfTwo(n) {
		return w.plan.TransformStrided(data[start:], data[start:], stride, inverse)
	}

	if stride == 1 {
		line := data[start : start+n]
		var err error
		if inverse {
			err = w.plan.Inverse(w.scratchB, line)
		} else {
			err = w.plan.Forward(w.scratchB, line)
		}
		if err != nil {
			return err
		}
		copy(line, w.scratchB)
		return nil
	}

	for i := range n {
		w.scratchA[i] = data[start+i*stride]
	}

	var err error
	if inverse {
		err = w.plan.Inverse(w.scratchB, w.scratchA)
	} else {
		err = w.plan.Forward(w.scratchB, w.scratchA)
	}
	if err != nil {
		return err
	}

	for i := range n {
		data[start+i*stride] = w.scratchB[i]
	}

	return nil
}

// FFTPlan wraps an algo-fft complex FFT plan for periodic (Fourier) transforms.
//
// It provides a convenience method to apply the 1D FFT along all lines of an
// N-dimensional grid stored in row-major order.
//
// FFTPlan is safe for concurrent TransformLines calls.
type FFTPlan struct {
	n          int
	workers    int
	workerPool *fftWorkerPool
}

// NewFFTPlan creates a new complex FFT plan for length n.
func NewFFTPlan(n int) (*FFTPlan, error) {
	return NewFFTPlanWithWorkers(n, 1)
}

// NewFFTPlanWithWorkers creates a new FFT plan with the requested worker count.
// workers <= 0 defaults to runtime.GOMAXPROCS.
func NewFFTPlanWithWorkers(n int, workers int) (*FFTPlan, error) {
	workers = effectiveWorkers(workers)

	workerPool, err := newFFTWorkerPool(n, workers)
	if err != nil {
		return nil, err
	}

	return &FFTPlan{
		n:          n,
		workers:    workers,
		workerPool: workerPool,
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
// Safe for concurrent calls on distinct data buffers: each parallel worker
// draws its plan and scratch from a pool instead of sharing plan state.
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

	lineStride := grid.RowMajorStride(shape)[axis]
	numLines := lineCount(shape, axis)
	workers := clampWorkers(p.workers, numLines)

	return parallelFor(workers, numLines, func(ctx context.Context, _ int, startLine, endLine int) error {
		worker, err := p.workerPool.get()
		if err != nil {
			return err
		}
		defer p.workerPool.put(worker)

		for line := startLine; line < endLine; line++ {
			if err := ctx.Err(); err != nil {
				return err
			}

			start := lineStartIndex(shape, axis, line)
			if err := fftTransformLine(worker, p.n, data, start, lineStride, inverse); err != nil {
				return err
			}
		}
		return nil
	})
}

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}
