package poisson

import (
	"runtime"
	"sync"

	"github.com/MeKo-Tech/algo-pde/grid"
)

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

func parallelFor(workers, tasks int, fn func(worker, start, end int) error) error {
	if tasks <= 0 {
		return nil
	}
	if workers <= 1 || tasks == 1 {
		return fn(0, 0, tasks)
	}

	chunk := (tasks + workers - 1) / workers
	var wg sync.WaitGroup
	var errOnce sync.Once
	var err error

	for w := 0; w < workers; w++ {
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
			if e := fn(worker, start, end); e != nil {
				errOnce.Do(func() {
					err = e
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
