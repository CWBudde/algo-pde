package poisson

import "sync"

// residentPool hands out per-call resources to make Solve safe for concurrent
// callers. Up to capacity items live in a GC-proof freelist channel so the
// steady state never re-allocates: capacity matches the resources a single
// Solve call needs (e.g. the plan's worker count), mirroring what the plan
// used to own eagerly. Anything beyond that — extra demand from concurrent
// Solve calls — overflows into a sync.Pool, so burst items are cached
// opportunistically and released by the GC when the burst is over.
type residentPool[T any] struct {
	resident chan *T
	pool     sync.Pool
}

func newResidentPool[T any](capacity int) *residentPool[T] {
	if capacity < 1 {
		capacity = 1
	}
	return &residentPool[T]{resident: make(chan *T, capacity)}
}

// get returns a pooled item, or nil if the caller must construct one.
func (p *residentPool[T]) get() *T {
	select {
	case v := <-p.resident:
		return v
	default:
	}
	v, _ := p.pool.Get().(*T)
	return v
}

func (p *residentPool[T]) put(v *T) {
	select {
	case p.resident <- v:
	default:
		p.pool.Put(v)
	}
}
