package poisson

import (
	"context"
	"runtime"
	"sync"

	"github.com/cwbudde/algo-pde/grid"
)

// cancelPollMask sets how often a tight, per-element worker loop polls its
// context for cancellation: once every (mask+1) iterations. It keeps the
// synchronized ctx.Err() call out of the hot path while still letting workers
// abandon a long chunk shortly after a sibling reports an error.
const cancelPollMask = 1<<10 - 1

func effectiveWorkers(workers int) int {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers < 1 {
		workers = 1
	}
	return workers
}

func clampWorkers(workers, tasks int) int {
	if tasks < 1 {
		return 1
	}
	if workers < 1 {
		workers = 1
	}
	if workers > tasks {
		return tasks
	}
	return workers
}

// parallelFor splits tasks across workers, invoking fn once per worker with the
// half-open [start, end) chunk it owns. fn receives a context that is cancelled
// as soon as any worker returns an error, so cooperating workers can check
// ctx.Err() at coarse granularity (the top of a unit of work, between lines)
// and return early instead of running to completion. The first error observed
// is returned; ctx.Canceled from workers that bailed out never overwrites it.
func parallelFor(workers, tasks int, fn func(ctx context.Context, worker, start, end int) error) error {
	if tasks <= 0 {
		return nil
	}
	if workers <= 1 || tasks == 1 {
		return fn(context.Background(), 0, 0, tasks)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	chunk := (tasks + workers - 1) / workers
	var wg sync.WaitGroup
	var errOnce sync.Once
	var err error

	for w := range workers {
		start := w * chunk
		if start >= tasks {
			break
		}
		end := start + chunk
		if end > tasks {
			end = tasks
		}

		wg.Add(1)
		go func(worker, start, end int) {
			defer wg.Done()
			if e := fn(ctx, worker, start, end); e != nil {
				errOnce.Do(func() {
					err = e
					cancel()
				})
			}
		}(w, start, end)
	}

	wg.Wait()
	return err
}

// lineCount returns the number of lines parallel to axis: the product of the
// extents of every other declared axis. It iterates over shape.Dim() axes
// rather than a hardcoded 3 so it cannot miscount a shape with fewer (or a
// degenerate trailing) dimension.
func lineCount(shape grid.Shape, axis int) int {
	count := 1
	for d := range shape.Dim() {
		if d != axis {
			count *= shape.N(d)
		}
	}
	return count
}

// lineStartIndex returns the linear index of the first element of the given
// line parallel to axis. The line number is decomposed across the non-axis
// declared axes (lowest axis varying fastest), matching the row-major layout.
func lineStartIndex(shape grid.Shape, axis, line int) int {
	stride := grid.RowMajorStride(shape)
	start := 0
	for d := range shape.Dim() {
		if d == axis {
			continue
		}
		n := shape.N(d)
		if n <= 0 {
			return 0
		}
		start += (line % n) * stride[d]
		line /= n
	}
	return start
}
