# algo-pde Implementation Plan

A fast spectral Poisson/Helmholtz solver library for Go, built on top of `algo-fft`.

---

## Phase 1: Grid Package (`grid/`) ✅ COMPLETE

- Shape/stride types, indexing helpers, and row-major stride computation.
- Iterators for lines and planes, plus strided copy utilities.
- Unit tests covering indexing and iterator behavior.

---

## Phase 2: Real-to-Real & Line Transforms (`r2r/`) ✅

- [x] **DST/DCT Core**: DST-I/II and DCT-I/II implemented via FFT embedding with full normalization and correctness tests.
- [x] **Plan API**: `DSTPlan`, `DCTPlan`, and `FFTPlan` for allocation-conscious, axis-wise transforms on N-D grids.
- [x] **Multi-D Support**: `ForwardLines`/`InverseLines` for all transform types with 2D/3D unit tests.
- [x] **Performance**: Verified O(N log N) scaling and optimized buffer management.

---

## Phase 3: Finite Difference Operators (`fd/`) - PARTIAL ✅

### 3.1 Laplacian eigenvalues

- [x] Implement `EigenvaluesPeriodic(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2\*cos(2πm/N)) / h²
- [x] Implement `EigenvaluesDirichlet(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2\*cos(πm/(N+1))) / h²
- [x] Implement `EigenvaluesNeumann(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2\*cos(πm/N)) / h²
- [x] Write tests comparing eigenvalues to explicit matrix eigendecomposition
- [x] Document eigenvalue formulas and grid conventions

### 3.2 Laplacian stencil (for testing/validation)

- [x] Implement `Apply1D(dst, src []float64, h float64, bc BCType)`
- [x] Implement `Apply2D(dst, src []float64, shape Shape, h [2]float64, bc [2]BCType)`
- [x] Implement `Apply3D(...)`
- [x] Write tests verifying Δu matches expected for known u

---

## Phase 4: Periodic Poisson Solver (`poisson/`)

### 4.1 1D Periodic Solver

- [x] Implement `Plan1DPeriodic` struct
- [x] Implement `NewPlan1DPeriodic(nx int, hx float64, opts ...Option) (*Plan1DPeriodic, error)`
- [x] Pre-compute eigenvalues in plan creation
- [x] Pre-allocate FFT plan from algo-fft
- [x] Pre-allocate work buffers
- [x] Implement `Plan.Solve(dst, rhs []float64) error`
- [x] Implement `Plan.SolveInPlace(buf []float64) error`
- [x] Implement zero-mode (mean) handling:
  - [x] Option: RequireMeanZeroRHS (error if not)
  - [x] Option: SubtractMean (auto-subtract before solve)
  - [x] Option: SetSolutionMean (set u's mean to specified value)
- [x] Write unit tests with manufactured solutions
- [x] Write benchmark tests

### 4.2 2D Periodic Solver

- [x] Implement `Plan2DPeriodic` struct
- [x] Implement `NewPlan2DPeriodic(nx, ny int, hx, hy float64, opts ...Option) (*Plan2DPeriodic, error)`
- [x] Use algo-fft's 2D real FFT plans where possible
- [x] Implement eigenvalue division in spectral space
- [x] Implement zero-mode handling (same options as 1D)
- [x] Write manufactured solution tests:
  - [x] u = sin(2πx/Lx) \* sin(2πy/Ly)
  - [x] u = cos(2πx/Lx) \* cos(4πy/Ly) (mixed frequencies)
- [x] Write convergence tests (error vs grid spacing)
- [x] Benchmark: 64², 128², 256², 512², 1024²

### 4.3 3D Periodic Solver

- [x] Implement `Plan3DPeriodic` struct
- [x] Implement `NewPlan3DPeriodic(nx, ny, nz int, hx, hy, hz float64, opts ...Option)`
- [x] Use algo-fft's 3D real FFT plans
- [x] Implement eigenvalue division
- [x] Implement zero-mode handling
- [x] Write manufactured solution tests
- [x] Benchmark: 32³, 64³, 128³

### 4.4 ND Periodic Solver (generic)

- [x] Implement `PlanNDPeriodic` for arbitrary dimensions
- [x] Implement `NewPlanNDPeriodic(shape Shape, h []float64, opts ...Option)`
- [x] Write tests for 4D case (stress test)

---

## Phase 5: Dirichlet/Neumann Poisson Solver

### 5.1 Unified Plan type with per-axis BC

- [x] Implement main `Plan` struct:
  ```go
  type Plan struct {
      dim      int
      n        [3]int
      h        [3]float64
      bc       [3]BCType
      eig      [3][]float64
      tr       [3]AxisTransform
      work     workspace
      opts     Options
  }
  ```
- [x] Implement `NewPlan(dim int, n []int, h []float64, bc []BCType, opts ...Option) (*Plan, error)`
- [x] Select appropriate eigenvalue formula per axis based on BC
- [x] Select appropriate transform (FFT/DST/DCT) per axis based on BC
- [x] Validate BC combinations (document restrictions if any)

### 5.2 1D Dirichlet Solver

- [x] Wire DST transform for Dirichlet BC
- [x] Compute Dirichlet eigenvalues
- [x] Write manufactured solution tests:
  - [x] u = sin(πx/L) (fundamental mode)
  - [x] u = sin(2πx/L) \* sin(πx/L) (combination)
- [x] Verify boundary values are exactly zero

### 5.3 1D Neumann Solver

- [x] Wire DCT transform for Neumann BC
- [x] Compute Neumann eigenvalues
- [x] Handle nullspace (constant mode has zero eigenvalue)
- [x] Write manufactured solution tests:
  - [x] u = cos(πx/L)
  - [x] u = cos(2πx/L)
- [x] Verify derivative at boundaries is zero (finite difference check)

### 5.4 2D Mixed BC Solver

- [x] Test Dirichlet-Dirichlet (both axes)
- [x] Test Neumann-Neumann (both axes)
- [x] Test Periodic-Dirichlet (mixed)
- [x] Test Dirichlet-Neumann (mixed)
- [x] Write manufactured solution tests for each combination
- [x] Document which combinations have nullspace issues

### 5.5 3D Mixed BC Solver

- [x] Implement 3D with arbitrary BC per axis
- [x] Test all 27 combinations (3³) or representative subset
- [x] Benchmark against periodic-only solver

---

## Phase 6: Inhomogeneous Boundary Conditions

### 6.1 Boundary value data structures

- [x] Define `BoundaryFace` enum (XLow, XHigh, YLow, YHigh, ZLow, ZHigh)
- [x] Define `BoundaryData` struct:
  ```go
  type BoundaryData struct {
      Face   BoundaryFace
      Type   BCType  // Dirichlet or Neumann
      Values []float64  // boundary values (shape of face)
  }
  ```
- [x] Define `BoundaryConditions` as collection of `BoundaryData`

### 6.2 RHS modification for inhomogeneous Dirichlet

- [x] Implement RHS contribution from boundary values
- [x] For each boundary cell: `rhs[i] -= u_boundary / h²`
- [x] Write tests with non-zero Dirichlet values
- [x] Verify solution matches boundary values at edges

### 6.3 RHS modification for inhomogeneous Neumann

- [x] Implement ghost point elimination or modified stencil
- [x] For each boundary cell: adjust RHS based on derivative condition
- [x] Write tests with non-zero Neumann values
- [x] Verify derivative at boundary matches specified value

### 6.4 Unified inhomogeneous API

- [x] Implement `Plan.SolveWithBC(dst, rhs []float64, bc BoundaryConditions) error`
- [x] Write comprehensive tests for 2D and 3D
- [x] Add examples to examples/ directory

---

## Phase 7: Helmholtz Solver Extension

### 7.1 Helmholtz operator: (α - Δ)u = f

- [x] Extend `Plan` with `alpha` parameter (shift)
- [x] Modify eigenvalue division: divide by `α + λ` instead of just `λ`
- [x] Implement `NewHelmholtzPlan(...)` constructor
- [x] Handle α = 0 case (reduces to Poisson with nullspace)
- [x] Write tests for positive α (well-posed problem)
- [x] Write tests for negative α (potential resonance issues - document)

### 7.2 Screened Poisson / reaction-diffusion steady state

- [x] Document use case: `u - νΔu = f` (implicit diffusion step)
- [x] Add example for diffusion time-stepping
- [x] Benchmark against iterative methods for comparison

---

## Phase 8: Performance Optimization

### 8.1 Zero-allocation solve path

- [ ] Audit `Solve()` for heap allocations using benchmarks
- [ ] Ensure all scratch memory is pre-allocated in Plan
- [ ] Add `Plan.WorkBytes()` method for memory introspection
- [ ] Write allocation benchmarks: `go test -bench=. -benchmem`
- [ ] Target: 0 allocs/op for Solve with pre-made plan

### 8.2 Parallelism support

- [ ] Pass through `Options{Workers: n}` to algo-fft plans
- [ ] Parallelize eigenvalue division loop (if beneficial)
- [ ] Parallelize line-wise DST/DCT transforms
- [ ] Benchmark single-threaded vs multi-threaded
- [ ] Document scaling characteristics

### 8.3 SIMD considerations

- [ ] Profile hot paths (eigenvalue division likely)
- [ ] Consider SIMD for eigenvalue division if beneficial
- [ ] Document any architecture-specific optimizations

### 8.4 Plan caching / reuse

- [ ] Document plan reuse patterns in README
- [ ] Consider `sync.Pool` for temporary buffers if needed
- [ ] Ensure thread-safety for concurrent Solve() calls on same plan

---

## Phase 9: Validation & Testing

### 9.1 Manufactured solution test suite

- [ ] Create `testdata/` with analytic test cases
- [ ] 1D periodic: u = sin(2πx/L)
- [ ] 1D Dirichlet: u = sin(πx/L)
- [ ] 1D Neumann: u = cos(πx/L) + x (linear + cosine)
- [ ] 2D periodic: u = sin(2πx/Lx) \* sin(2πy/Ly)
- [ ] 2D Dirichlet: u = sin(πx/Lx) \* sin(πy/Ly)
- [ ] 2D Neumann: u = cos(πx/Lx) \* cos(πy/Ly)
- [ ] 2D mixed: combinations of above
- [ ] 3D cases for each BC type

### 9.2 Convergence tests

- [ ] Implement `TestConvergence_*` tests
- [ ] Verify O(h²) error convergence for 2nd-order FD
- [ ] Plot convergence (log-log) in documentation

### 9.3 Reference solver comparison

- [ ] Implement naive dense solver for small grids (8x8, 16x16)
- [ ] Compare spectral solution against direct solve
- [ ] Maximum error should be O(machine epsilon \* condition number)

### 9.4 Fuzzing

- [ ] Add fuzz tests for robustness
- [ ] Fuzz input sizes, values, BC combinations
- [ ] Ensure no panics on edge cases

---

## Phase 10: Documentation & Examples

### 10.1 Package documentation

- [x] Write comprehensive doc.go for each package
- [ ] Document eigenvalue formulas with LaTeX/math
- [ ] Document memory layout conventions
- [ ] Document BC conventions and grid alignment

### 10.2 README.md

- [ ] Project overview and motivation
- [ ] Installation instructions
- [ ] Quick start example
- [ ] Performance characteristics
- [ ] Comparison with alternatives

### 10.3 Examples

- [ ] `examples/periodic1d/` - basic 1D periodic solve
- [ ] `examples/periodic2d/` - 2D periodic solve with visualization
- [ ] `examples/dirichlet2d/` - 2D Dirichlet problem
- [ ] `examples/neumann2d/` - 2D Neumann problem
- [ ] `examples/mixed2d/` - 2D mixed BC problem
- [ ] `examples/helmholtz/` - Helmholtz equation
- [ ] `examples/diffusion/` - implicit diffusion time-stepping

### 10.4 Benchmarks documentation

- [ ] Create BENCHMARKS.md with performance data
- [ ] Compare against gonum sparse solvers (if applicable)
- [ ] Document scaling: problem size vs time

---

## Phase 11: Future Extensions (Out of Scope for MVP)

### 11.1 Projection for incompressible flow

- [ ] Design API for pressure projection
- [ ] Implement divergence computation
- [ ] Implement gradient computation
- [ ] Write Navier-Stokes projection example

### 11.2 Variable coefficients

- [ ] Research preconditioned iterative methods
- [ ] Consider spectral method as preconditioner

### 11.3 Non-rectangular domains

- [ ] Research immersed boundary methods
- [ ] Consider mask-based approaches

---

## Implementation Order Summary

**MVP (Phases 0-4):** ~2-3 weeks of focused work

1. Phase 0: Setup ✅ COMPLETE
2. Phase 1: Grid package ✅ COMPLETE (core functionality)
3. Phase 2: R2R transforms ✅ PARTIAL (DST-I/DCT-I done, line-wise pending)
4. Phase 3: FD operators ✅ PARTIAL (eigenvalues done)
5. Phase 4: Periodic Poisson (1 week)

**Feature Complete (Phases 5-7):** ~2-3 weeks additional 6. Phase 5: Dirichlet/Neumann (1 week) 7. Phase 6: Inhomogeneous BC (3-4 days) 8. Phase 7: Helmholtz (2-3 days)

**Polish (Phases 8-10):** ~1-2 weeks 9. Phase 8: Optimization (ongoing) 10. Phase 9: Validation (ongoing, parallel with implementation) 11. Phase 10: Documentation (ongoing)

---

## Dependencies

- `github.com/MeKo-Christian/algo-fft` - FFT plans and execution
- Standard library only otherwise (no external deps)

## Testing Strategy

- Unit tests for every exported function
- Table-driven tests for BC combinations
- Manufactured solution tests for correctness
- Benchmark tests for performance regression
- Fuzz tests for robustness

## Success Criteria

1. **Correctness:** All manufactured solution tests pass with error < 1e-10 (for double precision, h → 0)
2. **Performance:** O(N log N) complexity verified empirically
3. **Memory:** Zero allocations in Solve() with pre-made plan
4. **API:** Clean, plan-based API consistent with algo-fft style
5. **Documentation:** All public APIs documented with examples
