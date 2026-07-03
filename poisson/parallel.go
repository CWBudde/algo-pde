package poisson

import (
	"context"
	"runtime"
	"sync"

	"github.com/MeKo-Tech/algo-pde/grid"
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

func lineCount(shape grid.Shape, axis int) int {
	other0, other1 := otherAxes(axis)
	return shape[other0] * shape[other1]
}

func lineStartIndex(shape grid.Shape, axis, line int) int {
	other0, other1 := otherAxes(axis)
	max0 := shape[other0]
	if max0 <= 0 {
		return 0
	}
	pos0 := line % max0
	pos1 := line / max0
	stride := grid.RowMajorStride(shape)
	return pos0*stride[other0] + pos1*stride[other1]
}

func otherAxes(axis int) (int, int) {
	switch axis {
	case 0:
		return 1, 2
	case 1:
		return 0, 2
	default:
		return 0, 1
	}
}
