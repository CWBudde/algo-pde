# Benchmarks

Indicative solve-time and allocation numbers for the 2D Poisson solver, one
benchmark per boundary-condition type across three grid sizes. Each measures a
single `Solve` call on an already-constructed plan (plan setup is amortized, as
it is in real use where one plan serves many solves).

> **Caveat:** these are indicative numbers from a single machine and a single
> run, not guarantees. They were produced with `go1.25.0` (linux/amd64) on an
> `Intel(R) Xeon(R) Processor @ 2.80GHz` with `GOMAXPROCS=4`. Absolute timings
> will differ substantially on other hardware, Go versions, and load; treat
> them as relative comparisons between boundary conditions and sizes.

## How to reproduce

```bash
go test -run '^$' -bench 'BenchmarkSolve2D_' -benchmem -benchtime=20x ./poisson/
```

The benchmarks live in `poisson/bc_scaling_bench_test.go`.

## Results

### Periodic (FFT)

| Grid  | ns/op      | allocs/op |
| ----- | ---------- | --------- |
| 256²  | 1,986,716  | 88        |
| 512²  | 6,770,841  | 90        |
| 1024² | 33,577,569 | 91        |

### Dirichlet (DST-I)

| Grid  | ns/op       | allocs/op |
| ----- | ----------- | --------- |
| 256²  | 55,020,468  | 82        |
| 512²  | 231,230,540 | 79        |
| 1024² | 986,844,856 | 77        |

### Neumann (DCT-II)

| Grid  | ns/op      | allocs/op |
| ----- | ---------- | --------- |
| 256²  | 4,957,786  | 88        |
| 512²  | 20,579,885 | 91        |
| 1024² | 75,091,810 | 89        |

## Notes

- **Neumann is now linearithmic.** Phase A.2 replaced the naive O(N²) DCT-II
  inverse transform with an FFT-embedded O(N log N) path. Neumann solves now
  scale like the periodic path (roughly ×4 per doubling of each axis) instead of
  the old quadratic blow-up.
- **The Dirichlet path is slow at these particular sizes because of FFT size,
  not an algorithmic regression.** The DST-I transform embeds an `n`-length axis
  into a `2(n+1)`-point FFT. For `n ∈ {256, 512, 1024}` the transform length
  `2(n+1) ∈ {514, 1026, 2050}` has large prime factors, forcing algo-fft onto
  its Bluestein (chirp-z) path, which is a large constant-factor penalty over a
  power-of-two FFT. The complexity is still O(N log N); choosing `n` such that
  `2(n+1)` is FFT-friendly (e.g. a power of two) removes the penalty.
- All boundary conditions run at a small, size-independent allocation count per
  solve (steady-state solves reuse the plan's work buffers).
