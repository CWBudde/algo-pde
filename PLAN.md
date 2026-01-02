# algo-pde Implementation Plan

A fast spectral Poisson/Helmholtz solver library for Go, built on top of `algo-fft`.

---

## Phase 0: Project Setup & Foundation ✅ COMPLETE

### 0.1 Repository initialization
- [x] Initialize Go module (`go mod init github.com/MeKo-Tech/algo-pde`)
- [x] Add `algo-fft` as dependency
- [x] Create basic directory structure:
  ```
  algo-pde/
  ├── poisson/     # Main Poisson solver package
  ├── r2r/         # Real-to-real transforms (DST/DCT)
  ├── grid/        # Shape, stride, indexing helpers
  ├── fd/          # Finite difference operators
  ├── internal/    # Internal utilities
  └── examples/    # Usage examples
  ```
- [x] Set up justfile with common commands (test, bench, lint, coverage)
- [x] Configure golangci-lint (.golangci.toml)
- [ ] Create initial README.md with project overview

### 0.2 Core types and interfaces
- [x] Define `BCType` enum (Periodic, Dirichlet, Neumann)
- [x] Define `AxisBC` struct for per-axis boundary conditions
- [x] Define `Shape` type for N-dimensional grid dimensions
- [x] Define `Stride` type for memory layout
- [x] Define `Options` struct for solver configuration
- [x] Define `AxisTransform` interface for transform abstraction
- [x] Create `errors.go` with custom error types

---

## Phase 1: Grid Package (`grid/`) ✅ COMPLETE

### 1.1 Shape and indexing
- [x] Implement `Shape` type with 1D/2D/3D support
- [x] Implement `Shape.Size()` - total element count
- [x] Implement `Shape.Dim()` - dimensionality
- [x] Implement index conversion: `(i,j,k) <-> linear`
- [x] Implement row-major stride computation
- [x] Write unit tests for indexing correctness

### 1.2 Grid iteration helpers
- [x] Implement line iterator (for axis-wise transforms)
- [ ] Implement plane iterator (for 3D operations)
- [ ] Implement strided copy utilities
- [x] Write tests for iteration patterns

---

## Phase 2: Real-to-Real Transforms (`r2r/`) - PARTIAL ✅

### 2.1 DST (Discrete Sine Transform) via FFT
- [x] Research DST-I vs DST-II conventions for FD grids
- [x] Implement `DST1Forward(dst, src []float64)` using odd extension + FFT
- [x] Implement `DST1Inverse(dst, src []float64)`
- [ ] Implement `DST2Forward(dst, src []float64)` (alternative formulation)
- [ ] Implement `DST2Inverse(dst, src []float64)`
- [ ] Add normalization options (orthogonal vs unnormalized)
- [x] Write round-trip tests: `inverse(forward(x)) ≈ x`
- [x] Write correctness tests against known analytic values
- [ ] Benchmark against theoretical O(N log N)

### 2.2 DCT (Discrete Cosine Transform) via FFT
- [x] Research DCT-I vs DCT-II conventions for FD grids
- [x] Implement `DCT1Forward(dst, src []float64)` using even extension + FFT
- [x] Implement `DCT1Inverse(dst, src []float64)`
- [ ] Implement `DCT2Forward(dst, src []float64)` (alternative formulation)
- [ ] Implement `DCT2Inverse(dst, src []float64)`
- [ ] Add normalization options
- [x] Write round-trip tests
- [x] Write correctness tests against known analytic values
- [ ] Benchmark performance

### 2.3 Transform plan types
- [x] Create `DSTPlan` with pre-allocated buffers
- [x] Create `DCTPlan` with pre-allocated buffers
- [x] Implement `NewDSTPlan(n int, opts ...Option) *DSTPlan`
- [x] Implement `NewDCTPlan(n int, opts ...Option) *DCTPlan`
- [ ] Ensure plans are safe for concurrent `Execute()` calls
- [x] Add `Plan.Bytes()` for memory introspection

### 2.4 Line-wise transforms (for multi-D)
- [ ] Implement `DSTPlan.TransformLines(data []float64, shape Shape, axis int)`
- [ ] Implement `DCTPlan.TransformLines(data []float64, shape Shape, axis int)`
- [ ] Implement `FFTPlan.TransformLines(...)` wrapper for periodic
- [ ] Write tests for 2D/3D line-wise transforms

---

## Phase 3: Finite Difference Operators (`fd/`) - PARTIAL ✅

### 3.1 Laplacian eigenvalues
- [x] Implement `EigenvaluesPeriodic(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2*cos(2πm/N)) / h²
- [x] Implement `EigenvaluesDirichlet(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2*cos(πm/(N+1))) / h²
- [x] Implement `EigenvaluesNeumann(n int, h float64) []float64`
  - Formula: λ_m = (2 - 2*cos(πm/N)) / h²
- [x] Write tests comparing eigenvalues to explicit matrix eigendecomposition
- [x] Document eigenvalue formulas and grid conventions

### 3.2 Laplacian stencil (for testing/validation)
- [ ] Implement `Apply1D(dst, src []float64, h float64, bc BCType)`
- [ ] Implement `Apply2D(dst, src []float64, shape Shape, h [2]float64, bc [2]BCType)`
- [ ] Implement `Apply3D(...)`
- [ ] Write tests verifying Δu matches expected for known u

---

## Phase 4: Periodic Poisson Solver (`poisson/`)

### 4.1 1D Periodic Solver
- [ ] Implement `Plan1DPeriodic` struct
- [ ] Implement `NewPlan1DPeriodic(nx int, hx float64, opts ...Option) (*Plan1DPeriodic, error)`
- [ ] Pre-compute eigenvalues in plan creation
- [ ] Pre-allocate FFT plan from algo-fft
- [ ] Pre-allocate work buffers
- [ ] Implement `Plan.Solve(dst, rhs []float64) error`
- [ ] Implement `Plan.SolveInPlace(buf []float64) error`
- [ ] Implement zero-mode (mean) handling:
  - [ ] Option: RequireMeanZeroRHS (error if not)
  - [ ] Option: SubtractMean (auto-subtract before solve)
  - [ ] Option: SetSolutionMean (set u's mean to specified value)
- [ ] Write unit tests with manufactured solutions
- [ ] Write benchmark tests

### 4.2 2D Periodic Solver
- [ ] Implement `Plan2DPeriodic` struct
- [ ] Implement `NewPlan2DPeriodic(nx, ny int, hx, hy float64, opts ...Option) (*Plan2DPeriodic, error)`
- [ ] Use algo-fft's 2D real FFT plans where possible
- [ ] Implement eigenvalue division in spectral space
- [ ] Implement zero-mode handling (same options as 1D)
- [ ] Write manufactured solution tests:
  - [ ] u = sin(2πx/Lx) * sin(2πy/Ly)
  - [ ] u = cos(2πx/Lx) * cos(4πy/Ly) (mixed frequencies)
- [ ] Write convergence tests (error vs grid spacing)
- [ ] Benchmark: 64², 128², 256², 512², 1024²

### 4.3 3D Periodic Solver
- [ ] Implement `Plan3DPeriodic` struct
- [ ] Implement `NewPlan3DPeriodic(nx, ny, nz int, hx, hy, hz float64, opts ...Option)`
- [ ] Use algo-fft's 3D real FFT plans
- [ ] Implement eigenvalue division
- [ ] Implement zero-mode handling
- [ ] Write manufactured solution tests
- [ ] Benchmark: 32³, 64³, 128³

### 4.4 ND Periodic Solver (generic)
- [ ] Implement `PlanNDPeriodic` for arbitrary dimensions
- [ ] Implement `NewPlanNDPeriodic(shape Shape, h []float64, opts ...Option)`
- [ ] Write tests for 4D case (stress test)

---

## Phase 5: Dirichlet/Neumann Poisson Solver

### 5.1 Unified Plan type with per-axis BC
- [ ] Implement main `Plan` struct:
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
- [ ] Implement `NewPlan(dim int, n []int, h []float64, bc []BCType, opts ...Option) (*Plan, error)`
- [ ] Select appropriate eigenvalue formula per axis based on BC
- [ ] Select appropriate transform (FFT/DST/DCT) per axis based on BC
- [ ] Validate BC combinations (document restrictions if any)

### 5.2 1D Dirichlet Solver
- [ ] Wire DST transform for Dirichlet BC
- [ ] Compute Dirichlet eigenvalues
- [ ] Write manufactured solution tests:
  - [ ] u = sin(πx/L) (fundamental mode)
  - [ ] u = sin(2πx/L) * sin(πx/L) (combination)
- [ ] Verify boundary values are exactly zero

### 5.3 1D Neumann Solver
- [ ] Wire DCT transform for Neumann BC
- [ ] Compute Neumann eigenvalues
- [ ] Handle nullspace (constant mode has zero eigenvalue)
- [ ] Write manufactured solution tests:
  - [ ] u = cos(πx/L)
  - [ ] u = cos(2πx/L)
- [ ] Verify derivative at boundaries is zero (finite difference check)

### 5.4 2D Mixed BC Solver
- [ ] Test Dirichlet-Dirichlet (both axes)
- [ ] Test Neumann-Neumann (both axes)
- [ ] Test Periodic-Dirichlet (mixed)
- [ ] Test Dirichlet-Neumann (mixed)
- [ ] Write manufactured solution tests for each combination
- [ ] Document which combinations have nullspace issues

### 5.5 3D Mixed BC Solver
- [ ] Implement 3D with arbitrary BC per axis
- [ ] Test all 27 combinations (3³) or representative subset
- [ ] Benchmark against periodic-only solver

---

## Phase 6: Inhomogeneous Boundary Conditions

### 6.1 Boundary value data structures
- [ ] Define `BoundaryFace` enum (XLow, XHigh, YLow, YHigh, ZLow, ZHigh)
- [ ] Define `BoundaryData` struct:
  ```go
  type BoundaryData struct {
      Face   BoundaryFace
      Type   BCType  // Dirichlet or Neumann
      Values []float64  // boundary values (shape of face)
  }
  ```
- [ ] Define `BoundaryConditions` as collection of `BoundaryData`

### 6.2 RHS modification for inhomogeneous Dirichlet
- [ ] Implement RHS contribution from boundary values
- [ ] For each boundary cell: `rhs[i] -= u_boundary / h²`
- [ ] Write tests with non-zero Dirichlet values
- [ ] Verify solution matches boundary values at edges

### 6.3 RHS modification for inhomogeneous Neumann
- [ ] Implement ghost point elimination or modified stencil
- [ ] For each boundary cell: adjust RHS based on derivative condition
- [ ] Write tests with non-zero Neumann values
- [ ] Verify derivative at boundary matches specified value

### 6.4 Unified inhomogeneous API
- [ ] Implement `Plan.SolveWithBC(dst, rhs []float64, bc BoundaryConditions) error`
- [ ] Write comprehensive tests for 2D and 3D
- [ ] Add examples to examples/ directory

---

## Phase 7: Helmholtz Solver Extension

### 7.1 Helmholtz operator: (α - Δ)u = f
- [ ] Extend `Plan` with `alpha` parameter (shift)
- [ ] Modify eigenvalue division: divide by `α + λ` instead of just `λ`
- [ ] Implement `NewHelmholtzPlan(...)` constructor
- [ ] Handle α = 0 case (reduces to Poisson with nullspace)
- [ ] Write tests for positive α (well-posed problem)
- [ ] Write tests for negative α (potential resonance issues - document)

### 7.2 Screened Poisson / reaction-diffusion steady state
- [ ] Document use case: `u - νΔu = f` (implicit diffusion step)
- [ ] Add example for diffusion time-stepping
- [ ] Benchmark against iterative methods for comparison

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
- [ ] 2D periodic: u = sin(2πx/Lx) * sin(2πy/Ly)
- [ ] 2D Dirichlet: u = sin(πx/Lx) * sin(πy/Ly)
- [ ] 2D Neumann: u = cos(πx/Lx) * cos(πy/Ly)
- [ ] 2D mixed: combinations of above
- [ ] 3D cases for each BC type

### 9.2 Convergence tests
- [ ] Implement `TestConvergence_*` tests
- [ ] Verify O(h²) error convergence for 2nd-order FD
- [ ] Plot convergence (log-log) in documentation

### 9.3 Reference solver comparison
- [ ] Implement naive dense solver for small grids (8x8, 16x16)
- [ ] Compare spectral solution against direct solve
- [ ] Maximum error should be O(machine epsilon * condition number)

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

**Feature Complete (Phases 5-7):** ~2-3 weeks additional
6. Phase 5: Dirichlet/Neumann (1 week)
7. Phase 6: Inhomogeneous BC (3-4 days)
8. Phase 7: Helmholtz (2-3 days)

**Polish (Phases 8-10):** ~1-2 weeks
9. Phase 8: Optimization (ongoing)
10. Phase 9: Validation (ongoing, parallel with implementation)
11. Phase 10: Documentation (ongoing)

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
