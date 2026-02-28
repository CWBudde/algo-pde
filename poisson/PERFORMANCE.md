# Poisson Solver Performance Analysis

This document summarizes profiling analysis of the spectral Poisson solver and explains architecture-specific optimization decisions.

## Profiling Summary

### Methodology

CPU profiling was performed using Go's built-in profiler:

```bash
go test -bench=BenchmarkPlanSolve2D_Dirichlet -cpuprofile=cpu.prof ./poisson/
go tool pprof -top cpu.prof
```

Test configuration: 128x128 grid, Dirichlet boundary conditions.

### Results

| Component                               | Cumulative Time | % of Total |
| --------------------------------------- | --------------- | ---------- |
| DST/FFT transforms (algo-fft)           | ~8.1s           | 95.2%      |
| Eigenvalue division                     | ~30-50ms        | 0.3%       |
| Other (setup, parallelization overhead) | ~0.3s           | ~4.5%      |

The FFT-based transforms completely dominate runtime, which is expected for spectral solvers with O(N log N) complexity.

### Eigenvalue Division Analysis

The `applyEigenvalues()` function performs spectral division:

```go
p.work.Complex[idx] /= complex(denom, 0)
```

Despite being O(N) for the total grid size, this operation accounts for less than 0.5% of total runtime because:

1. It's a simple division operation with good cache locality
2. The FFT transforms involve many more floating-point operations
3. Modern CPUs handle scalar complex division efficiently

## SIMD Optimization Decision

**Decision: SIMD optimization of eigenvalue division is not implemented.**

Rationale:

- Maximum potential gain: ~0.3% of total runtime
- Even a 10x speedup would yield only ~0.27% total improvement
- Maintenance cost of assembly code outweighs negligible benefit

## Architecture-Specific Optimizations

### FFT Operations (Handled by algo-fft)

The `algo-fft` dependency provides extensive SIMD optimizations:

- 134 assembly kernels for amd64 (AVX2) and arm64 (NEON)
- Automatic CPU feature detection
- Architecture-specific radix implementations

Since FFT operations consume 95%+ of runtime, optimizations in algo-fft have maximum impact.

### Eigenvalue Division (Pure Go)

The eigenvalue division loop uses pure Go and benefits from:

- Go compiler optimizations (bounds check elimination, inlining)
- Parallelization via `parallelFor()` for multi-core scaling
- Sequential memory access patterns for good cache behavior

## Benchmark Reference

Run benchmarks with:

```bash
just bench-pkg pkg=poisson
```

Example output (12th Gen Intel Core i7-1255U):

```
BenchmarkPlanSolve2D_Dirichlet-12    76    28281632 ns/op    8056 B/op    140 allocs/op
```

## Scaling Characteristics

The solver exhibits O(N log N) complexity dominated by FFT operations:

| Grid Size | Approximate Time |
| --------- | ---------------- |
| 64x64     | ~5ms             |
| 128x128   | ~20ms            |
| 256x256   | ~90ms            |
| 512x512   | ~400ms           |

Multi-threaded scaling improves performance for larger grids where parallel overhead is amortized.
