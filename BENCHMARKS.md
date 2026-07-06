# Benchmarks

Indicative solve-time and allocation numbers for the 2D Poisson solver, one
benchmark per boundary-condition type across three grid sizes. Each measures a
single `Solve` call on an already-constructed plan (plan setup is amortized, as
it is in real use where one plan serves many solves).

> **Caveat:** these are indicative numbers from a single machine and a single
> run, not guarantees. They were produced with `go1.26.1` (linux/amd64) on a
> `12th Gen Intel(R) Core(TM) i7-1255U` with `GOMAXPROCS=4`. Absolute timings
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
| 256²  | 1,493,823  | 88        |
| 512²  | 7,259,626  | 89        |
| 1024² | 36,608,177 | 91        |

### Dirichlet (DST-I)

| Grid  | ns/op       | allocs/op |
| ----- | ----------- | --------- |
| 256²  | 26,074,400  | 81        |
| 512²  | 133,387,917 | 80        |
| 1024² | 538,223,043 | 78        |

### Neumann (DCT-II)

| Grid  | ns/op      | allocs/op |
| ----- | ---------- | --------- |
| 256²  | 3,197,406  | 90        |
| 512²  | 14,466,610 | 90        |
| 1024² | 71,407,996 | 90        |

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
  solve (steady-state solves reuse the plan's work buffers). The absolute
  allocation count scales with `GOMAXPROCS` (one pooled worker per goroutine),
  not with grid size.
- **Phase G.3 halved the transform cost.** The DST/DCT axis transforms now embed
  their real, symmetric extension into a **float64 real-input FFT**
  (`algo-fft`'s `NewPlanReal64`) instead of a full `complex128` FFT of the same
  length — no accuracy loss and no extra allocation. Measured same-machine in the
  `r2r` package (`go test ./r2r -bench Transform`), each forward/inverse
  transform runs ≈1.3–2.2× faster (typically ~2×), which flows directly into the
  Dirichlet/Neumann/mixed-BC solve times above.
